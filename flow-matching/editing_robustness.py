import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
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

flags.DEFINE_string("input_dir", "../dataset/celebaHQ/images", "Directory containing input images")
flags.DEFINE_string("output_base_dir", "./robustness", "Base directory for robustness results")
flags.DEFINE_string("classifier_path", "../classifier/checkpoints/time_resnet18_epoch_2.pth", "Path to the classifier checkpoint")
flags.DEFINE_string("test_json", "test_dataset.json", "Path to the JSON file containing test data mapping")


ATTRIBUTE_MAPPING = {
    'eyeglasses': 0, 'male': 1, 'female': 2, 'smiling': 3, 'hat': 4,
    'mustache': 5, 'beard': 6, 'pale_skin': 7, 'blond': 8, 'attractive': 9
}

GUIDANCE_SCALES = {
    'eyeglasses': 40.0, 'male': 40.0, 'female': 40.0, 'smiling': 30.0,
    'hat': 40.0, 'mustache': 40.0, 'beard': 40.0, 'pale_skin': 40.0,
    'blond': 40.0, 'attractive': 20.0
}

TRANSFORMATIONS = ['rotate', 'shift', 'zoom']

def restore_checkpoint_inference(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        print(f"Error: No checkpoint found at {ckpt_dir}")
        return state
        
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state

def apply_robustness_transform(img, t_type):
    if t_type == 'rotate':
        return TF.rotate(img, 30)
    elif t_type == 'shift':
        return TF.affine(img, angle=0, translate=[30, 30], scale=1.0, shear=0)
    elif t_type == 'zoom':
        return TF.affine(img, angle=0, translate=[0, 0], scale=1.3, shear=0)
    return img

def edit_samples(config, workdir):
    config.eval.batch_size = 5
    device = config.device
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    output_dir_original = os.path.join(FLAGS.output_base_dir, "original_transformed")
    output_dir_edited = os.path.join(FLAGS.output_base_dir, "edited_transformed")

    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print(f"[INFO] Loading {FLAGS.test_json}...")
    with open(FLAGS.test_json, 'r') as f:
        test_data = json.load(f)

    print("[INFO] Initializing generator model...")
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    ckpt_path = os.path.join(workdir, "checkpoints", "checkpoint_10.pth")
    state = restore_checkpoint_inference(ckpt_path, state, device)
    ema.copy_to(score_model.parameters())
    score_model.eval()

    print(f"[INFO] Initializing classifier from {FLAGS.classifier_path}...")
    classifier = TimeCondResNet18(num_classes=10).to(device)
    classifier.load_state_dict(torch.load(FLAGS.classifier_path, map_location=device))
    classifier.eval()

    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling.init_type, 
        noise_scale=config.sampling.init_noise_scale, 
        use_ode_sampler="rk45", 
        sigma_var=config.sampling.sigma_variance, 
        ode_tol=config.sampling.ode_tol, 
        sample_N=config.sampling.sample_N
    )

    for target_attribute, image_names in test_data.items():
        if target_attribute not in ATTRIBUTE_MAPPING:
            continue
            
        target_idx = ATTRIBUTE_MAPPING[target_attribute]
        current_guidance = GUIDANCE_SCALES.get(target_attribute, 30.0)
        
        target_out_dir_orig = os.path.join(output_dir_original, target_attribute)
        target_out_dir_edit = os.path.join(output_dir_edited, target_attribute)
        os.makedirs(target_out_dir_orig, exist_ok=True)
        os.makedirs(target_out_dir_edit, exist_ok=True)
        
        batch_names = image_names[:5]
        if not batch_names:
            continue

        print(f"\n[INFO] Target: {target_attribute.upper()} | Guidance: {current_guidance} | Samples: 5")

        for t_type in TRANSFORMATIONS:
            print(f"       -> Transformation: {t_type.upper()}")
            
            batch_tensors = []
            valid_names = []
            
            for img_name in batch_names:
                img_path = os.path.join(FLAGS.input_dir, img_name)
                if not os.path.exists(img_path):
                    continue
                    
                img = Image.open(img_path).convert('RGB')
                img = apply_robustness_transform(img, t_type)
                batch_tensors.append(base_transform(img))
                valid_names.append(img_name)
                
            if not batch_tensors:
                continue
                
            img_test_batch = torch.stack(batch_tensors).to(device)
            current_shape = (len(valid_names), config.data.num_channels, config.data.image_size, config.data.image_size)

            desampling_fn = sampling.get_rectified_flow_inversion_fn(sde, current_shape, device)
            sampling_fn = sampling.get_sampling_fn(
                config, sde, current_shape, inverse_scaler, eps=1e-3,
                classifier=classifier, target_attr_idx=target_idx, guidance_scale=current_guidance
            )

            z0, _ = desampling_fn(score_model, img_test_batch)
            edited_batch, _ = sampling_fn(score_model, z=z0)

            original_batch_np = np.clip(inverse_scaler(img_test_batch).permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            edited_batch_np = np.clip(edited_batch.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            
            for b_idx, img_name in enumerate(valid_names):
                base_name = img_name.replace(".jpg", "")
                
                orig_img = Image.fromarray(original_batch_np[b_idx])
                orig_save_path = os.path.join(target_out_dir_orig, f"{base_name}_{t_type}.jpg")
                orig_img.save(orig_save_path)

                edit_img = Image.fromarray(edited_batch_np[b_idx])
                edit_save_path = os.path.join(target_out_dir_edit, f"{base_name}_{target_attribute}_{t_type}.jpg")
                edit_img.save(edit_save_path)

    print("\n[INFO] Robustness evaluation completed successfully.")

def main(argv):
    edit_samples(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
    app.run(main)

