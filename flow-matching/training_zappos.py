import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision
import numpy as np

from absl import app, flags
from ml_collections import config_flags

from models import utils as mutils
from models.ncsnpp import NCSNpp
import losses
import datasets
import sde_lib
import sampling

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "./configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py", "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "./logs/zappos", "Work directory.")
flags.DEFINE_enum("mode", "train", ["train", "eval", "reflow"], "Running mode")

flags.DEFINE_string("data_dir", "../dataset/zappos/images", "Path to the Zappos dataset directory")
flags.DEFINE_string("base_model_ckpt", "./logs/celebahq/checkpoints/checkpoint_10.pth", "Path to the base model checkpoint")
flags.DEFINE_string("output_model_path", "lora_zappos_last.pth", "Path to save the fine-tuned LoRA model")
flags.DEFINE_string("sample_output_dir", "samples_zappos", "Directory to save generated samples during training")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs")
flags.DEFINE_integer("train_batch_size", 1, "Training batch size")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")


class ZapposDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = []
        
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, f))

        print(f"[INFO] Zappos dataset initialized: {len(self.image_files)} images found.")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        return {'image': self.transform(img)}


def train_lora(config, workdir):
    device = config.device if hasattr(config, 'device') else torch.device("cuda")

    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high')

    print("[INFO] Initializing base generator model...")
    score_model = mutils.create_model(config) 

    if isinstance(score_model, torch.nn.DataParallel):
        score_model = score_model.module

    print(f"[INFO] Loading base weights from {FLAGS.base_model_ckpt}...")
    ckpt = torch.load(FLAGS.base_model_ckpt, map_location=device, weights_only=False)
    
    if 'model' in ckpt:
        clean_state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    elif 'ema' in ckpt:
        clean_state_dict = {k.replace("module.", ""): v for k, v in ckpt['ema'].items()}
    else:
        clean_state_dict = ckpt

    score_model.load_state_dict(clean_state_dict, strict=False)

    print("[INFO] Injecting LoRA layers...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["Conv_0", "Conv_1", "Conv_2", "Dense_0"],
        lora_dropout=0.1, bias="none"
    )
    model = get_peft_model(score_model, lora_config)
    model.to(device)

    trainable_p, all_p = model.get_nb_trainable_parameters()
    print(f"[INFO] LoRA Parameters | Trainable: {trainable_p} | Total: {all_p} | Percentage: {100 * trainable_p / all_p:.2f}%")

    train_ds = ZapposDataset(data_dir=FLAGS.data_dir)
    train_loader = DataLoader(
        train_ds, 
        batch_size=FLAGS.train_batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,          
        prefetch_factor=4,        
        persistent_workers=True   
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.learning_rate)

    inverse_scaler = lambda x: (x + 1.) / 2.

    sde = sde_lib.RectifiedFlow(
        init_type=config.sampling.init_type, 
        noise_scale=config.sampling.init_noise_scale, 
        use_ode_sampler="rk45", 
        sigma_var=config.sampling.sigma_variance, 
        ode_tol=config.sampling.ode_tol, 
        sample_N=config.sampling.sample_N
    )

    sampling_shape = (4, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, eps=1e-3)

    os.makedirs(FLAGS.sample_output_dir, exist_ok=True)

    print(f"[INFO] Starting training loop on GPU for {FLAGS.num_epochs} epochs...")
    eps = 1e-4
    accumulation_steps = 8

    for epoch in range(FLAGS.num_epochs):
        model.train() 
        optimizer.zero_grad(set_to_none=True) 
        
        for step, batch_data in enumerate(train_loader):
            x1 = batch_data['image'].to(device, non_blocking=True).float()
            
            z0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device='cpu').to(device, non_blocking=True).view(-1, 1, 1, 1)
            t = t * (1.0 - eps) + eps
            
            xt = t * x1 + (1.0 - t) * z0
            target = x1 - z0
            
            prediction = model(xt, t.view(-1) * 999)
            
            loss = torch.mean(torch.square(prediction - target))
            loss_accum = loss / accumulation_steps
            
            loss_accum.backward()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) 
            
            if step % 10 == 0:
                print(f"[INFO] Epoch [{epoch:02d}/{FLAGS.num_epochs}] | Step {step:04d} | Loss: {loss.item():.4e}")
                
        print(f"[INFO] Saving model weights to {FLAGS.output_model_path}...")
        torch.save(model.state_dict(), FLAGS.output_model_path)

        model.eval()
        with torch.no_grad():
            samples, nfe = sampling_fn(model)
            samples = inverse_scaler(samples)
            samples = torch.clamp(samples, 0.0, 1.0)
            
            save_path = os.path.join(FLAGS.sample_output_dir, f"sample_epoch_{epoch:02d}.png")
            torchvision.utils.save_image(samples, save_path, nrow=2, padding=2)
            print(f"[INFO] Sample successfully generated: {save_path} (NFE: {nfe})")

    print("[INFO] Training completed successfully.")


def main(argv):
    train_lora(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
    app.run(main)
