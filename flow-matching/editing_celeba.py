import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from absl import app, flags
from ml_collections import config_flags

import sde_lib
import sampling
import datasets
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from models import ncsnpp, ddpm, ncsnv2

sys.path.append("../classifier")
from model import TimeCondResNet18

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "./configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py", "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "./logs/celebahq", "Work directory.")
flags.DEFINE_enum("mode", "train", ["train", "eval", "reflow"], "Running mode")
flags.DEFINE_string("eval_folder", "eval", "Folder name for storing evaluation results")

flags.DEFINE_string("target_attribute", "beard", "The attribute to add or remove")
flags.DEFINE_float("guidance_scale", 40.0, "The scale of the classifier guidance")
flags.DEFINE_string("classifier_path", "../classifier/checkpoints/time_resnet18_epoch_18.pth", "Path to the trained classifier weights")
flags.DEFINE_string("input_dir", "../dataset/celebaHQ/images", "Directory containing the source images")
flags.DEFINE_string("output_dir", "./results_editing_celebaHQ", "Directory where the edited images will be saved")

ATTRIBUTE_MAPPING = {
    'eyeglasses': 0, 'male': 1, 'female': 2, 'smiling': 3, 'hat': 4,
    'mustache': 5, 'beard': 6, 'pale_skin': 7, 'blond': 8, 'attractive': 9
}

IMAGE_NAMES = [
    "29613.jpg", "19993.jpg", "17964.jpg", "25247.jpg", "25230.jpg",
    "25979.jpg", "14741.jpg", "10115.jpg", "17411.jpg", "16858.jpg"]

def restore_checkpoint_inference(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        print(f"Error: No checkpoint found at {ckpt_dir}")
        return state
        
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state

def edit_samples(config, workdir):
    config.eval.batch_size = 5
    device = config.device
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    target_out_dir = os.path.join(FLAGS.output_dir, FLAGS.target_attribute)
    os.makedirs(target_out_dir, exist_ok=True)

    print("[INFO] Initializing generator model...")
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_10.pth")
    state = restore_checkpoint_inference(ckpt_path, state, device)
    ema.copy_to(score_model.parameters())
    score_model.eval()

    print("[INFO] Initializing classifier...")
    classifier = TimeCondResNet18(num_classes=10).to(device)
    classifier.load_state_dict(torch.load(FLAGS.classifier_path, map_location=device))
    classifier.eval()

    target_idx = ATTRIBUTE_MAPPING[FLAGS.target_attribute]

    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling.init_type, 
        noise_scale=config.sampling.init_noise_scale, 
        use_ode_sampler="rk45", 
        sigma_var=config.sampling.sigma_variance, 
        ode_tol=config.sampling.ode_tol, 
        sample_N=config.sampling.sample_N
    )

    print(f"\n[INFO] Starting editing pipeline for {len(IMAGE_NAMES)} images (Batch size: {config.eval.batch_size})")
    
    for i in range(0, len(IMAGE_NAMES), config.eval.batch_size):
        batch_names = IMAGE_NAMES[i:i + config.eval.batch_size]
        batch_tensors = []
        valid_names = []
        
        for img_name in batch_names:
            img_path = os.path.join(FLAGS.input_dir, img_name)
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert('RGB')
            batch_tensors.append(transform(img))
            valid_names.append(img_name)
            
        if not batch_tensors:
            continue
            
        print(f"\n[INFO] Processing batch: {valid_names}")
        
        img_test_batch = torch.stack(batch_tensors).to(device)
        current_batch_size = len(valid_names)
        current_shape = (current_batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)

        desampling_fn = sampling.get_rectified_flow_inversion_fn(sde, current_shape, device)
        sampling_fn = sampling.get_sampling_fn(
            config, sde, current_shape, inverse_scaler, eps=1e-3,
            classifier=classifier, 
            target_attr_idx=target_idx, 
            guidance_scale=FLAGS.guidance_scale
        )

        print("       -> Inverting batch to latent space...")
        z0, _ = desampling_fn(score_model, img_test_batch)

        print("       -> Applying classifier-guided generation...")
        edited_batch, _ = sampling_fn(score_model, z=z0)

        edited_batch_np = np.clip(edited_batch.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        original_batch_np = np.clip(inverse_scaler(img_test_batch).permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        
        for b_idx, img_name in enumerate(valid_names):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original_batch_np[b_idx])
            axes[0].axis('off')
            axes[0].set_title("Original")

            axes[1].imshow(edited_batch_np[b_idx])
            axes[1].axis('off')
            axes[1].set_title(f"Edited: + {FLAGS.target_attribute.capitalize()}")

            plt.tight_layout()
            
            save_path = os.path.join(target_out_dir, img_name)
            final_save_path = save_path.replace(".jpg", "") + "_" + FLAGS.target_attribute + ".jpg"
            
            plt.savefig(final_save_path, bbox_inches='tight')
            plt.close(fig)
            
            print(f"       -> Saved: {final_save_path}")

    print("\n[INFO] Editing pipeline completed successfully.")

def main(argv):
    edit_samples(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
    app.run(main)

