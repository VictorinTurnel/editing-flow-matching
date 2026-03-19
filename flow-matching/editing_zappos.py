import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from absl import app, flags
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model

import sde_lib
import sampling
import datasets
from generate_samples import restore_checkpoint_inference
from models.ema import ExponentialMovingAverage
from models import utils as mutils

sys.path.append("../classifier")
from classifier.model import TimeCondResNet18

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval", "reflow"], "Running mode")
flags.DEFINE_string("eval_folder", "eval", "Folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])

ATTRIBUTE_MAPPING = {
    'Boots': 0, 'Sandals': 1, 'Sneakers': 2, 'High_Heels': 3,
    'Flats': 4, 'Leather': 5, 'Laces': 6, 'Slip_on': 7,
    'Men': 8, 'Women': 9
}

TARGET_ATTRIBUTE = 'High_Heels' 
GUIDANCE_SCALE = 40.0

CLASSIFIER_PATH = "../classifier/checkpoints/time_resnet18_epoch_18.pth"
LORA_ZAPPOS_PATH = "lora_zappos_last.pth" 


INPUT_DIR = "/path/to/dataset/zappos/images"
OUTPUT_DIR = "/path/to/results"

IMAGE_NAMES = [
    "100627-255.jpg", "101026-3.jpg", "101093-342648.jpg", "101404-231.jpg", 
    "104730-35.jpg", "104733-1647.jpg", "105214-6.jpg", "107219-152359.jpg", 
    "107999-585.jpg", "115220-151.jpg"
]

def edit_samples(config, workdir):
    config.eval.batch_size = 5 
    device = config.device
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    target_out_dir = os.path.join(OUTPUT_DIR, TARGET_ATTRIBUTE)
    os.makedirs(target_out_dir, exist_ok=True)

    print("[INFO] Initializing generator model...")
    score_model = mutils.create_model(config)
    if isinstance(score_model, torch.nn.DataParallel):
        score_model = score_model.module

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_10.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    clean_state_dict = {k.replace("module.", ""): v for k, v in ckpt['ema'].items()}
    score_model.load_state_dict(clean_state_dict, strict=False)

    print("[INFO] Applying LoRA weights for domain transfer...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["Conv_0", "Conv_1", "Conv_2", "Dense_0"],
        lora_dropout=0.1, bias="none"
    )
    score_model = get_peft_model(score_model, lora_config)
    score_model.to(device)
    score_model.load_state_dict(torch.load(LORA_ZAPPOS_PATH, map_location=device))
    score_model.eval()

    print("[INFO] Initializing classifier...")
    classifier = TimeCondResNet18(num_classes=10).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.eval()

    target_idx = ATTRIBUTE_MAPPING[TARGET_ATTRIBUTE]

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
            img_path = os.path.join(INPUT_DIR, img_name)
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
            guidance_scale=GUIDANCE_SCALE
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
            axes[1].set_title(f"Edited: + {TARGET_ATTRIBUTE.replace('_', ' ')}")

            plt.tight_layout()
            
            save_path = os.path.join(target_out_dir, img_name)
            final_save_path = save_path.replace(".jpg", "") + "_" + TARGET_ATTRIBUTE + ".jpg"
            
            os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
            plt.savefig(final_save_path, bbox_inches='tight')
            plt.close(fig)
            
            print(f"       -> Saved: {final_save_path}")

    print("\n[INFO] Editing pipeline completed successfully.")

def main(argv):
    edit_samples(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
    app.run(main)