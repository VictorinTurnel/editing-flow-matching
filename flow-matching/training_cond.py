import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision
import numpy as np
from tqdm import tqdm

from models import utils as mutils
from models.ncsnpp import NCSNpp
from configs.rectified_flow.celeba_cond import get_config
import sde_lib
import sampling

sys.path.append("../classifier")
from dataset import TimeCondCelebADataset

class GeneratorCelebADataset(TimeCondCelebADataset):
    def __getitem__(self, idx):
        img_name, img_attr = self.images_info[idx]

        img_path = os.path.join(self.data_dir, str(int(img_name.split(".")[0])) + ".jpg")
        x_1 = Image.open(img_path).convert('RGB')
        x_1 = self.transform(x_1)

        labels = torch.zeros(10, dtype=torch.float32)
        labels[0] = img_attr[self.attr_idx["eyeglasses"]]
        labels[1] = img_attr[self.attr_idx["male"]]
        labels[2] = 1.0 - img_attr[self.attr_idx["male"]]
        labels[3] = img_attr[self.attr_idx["smiling"]]
        labels[4] = img_attr[self.attr_idx["hat"]]
        labels[5] = img_attr[self.attr_idx["mustache"]]
        labels[6] = 1.0 - img_attr[self.attr_idx["beard"]]
        labels[7] = img_attr[self.attr_idx["pale_Skin"]]
        labels[8] = img_attr[self.attr_idx["blond"]]
        labels[9] = img_attr[self.attr_idx["attractive"]]

        return {'image': x_1, 'label': labels}

config = get_config()
device = torch.device("cuda")
batch_size = 1
lr = 1e-4

DATASET = "../dataset"
DATA_DIR = os.path.join(DATASET,"data")
ATTR_FILE = os.path.join(DATASET,"list_attr_celeba_hq.txt")

dataset = GeneratorCelebADataset(DATA_DIR, ATTR_FILE)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

score_model = mutils.create_model(config) 

if isinstance(score_model, torch.nn.DataParallel):
    score_model = score_model.module

ckpt = torch.load("logs/celebahq/checkpoints/checkpoint_10.pth", map_location=device, weights_only=False)
clean_state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
score_model.load_state_dict(clean_state_dict, strict=False)

lora_config = LoraConfig(
    r=4, lora_alpha=32,
    target_modules=["Conv_0", "Conv_1", "Conv_2", "Dense_0"],
    lora_dropout=0.1, bias="none"
)
model = get_peft_model(score_model, lora_config)
model.to(device)

for name, param in model.named_parameters():
    if "attr_mlp" in name:
        param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

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

os.makedirs("temp", exist_ok=True)

eps = 1e-4
accumulation_steps = 8

for epoch in range(50):
    model.train() 
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
    
    for step, batch_data in enumerate(pbar):
        x1 = batch_data['image'].to(device, non_blocking=True).float()
        attrs = batch_data['label'].to(device, non_blocking=True).float()

        if torch.rand(1).item() < 0.1:
            attrs = torch.zeros_like(attrs)
        
        z0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device='cpu').to(device, non_blocking=True).view(-1, 1, 1, 1)
        t = t * (1.0 - eps) + eps
        
        xt = t * x1 + (1.0 - t) * z0
        target = x1 - z0
        
        prediction = model(xt, t.view(-1) * 999, attrs=attrs)
        
        loss = torch.mean(torch.square(prediction - target))
        loss_accum = loss / accumulation_steps
        
        loss_accum.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 
        
        pbar.set_postfix({"Loss": f"{loss.item():.4e}"})
        
    torch.save(model.state_dict(), f"lora_cond_last.pth")

    model.eval()
    with torch.no_grad():
        samples, nfe = sampling_fn(model)
        samples = inverse_scaler(samples)
        samples = torch.clamp(samples, 0.0, 1.0)
        
        save_path = os.path.join("temp", f"sample_epoch_{epoch:02d}.png")
        torchvision.utils.save_image(samples, save_path, nrow=2, padding=2)