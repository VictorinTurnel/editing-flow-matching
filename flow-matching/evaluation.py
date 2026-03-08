import os
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import sys

sys.path.append("../classifier")
from model import TimeCondResNet18

INPUT_DIR = "../dataset/data/"
BASE_OUTPUT_DIR = "/home/victorin/Documents/MVA/DELIRE/flow-matching-clf-guid/results_test_linear/"
CLASSIFIER_PATH = "../classifier/checkpoints/time_resnet18_epoch_18.pth"

ATTRIBUTE_MAPPING = {
    'eyeglasses': 0, 'male': 1, 'female': 2, 'smiling': 3, 'hat': 4,
    'mustache': 5, 'beard': 6, 'pale_skin': 7, 'blond': 8, 'attractive': 9
}

def evaluate_all_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    print("[INFO] Initializing models and metrics...")
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    classifier = TimeCondResNet18(num_classes=10).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.eval()

    total_images_evaluated = 0

    print(f"\n{'ATTRIBUTE':<15} | {'LPIPS':<8} | {'ACCURACY':<9} | {'COUNT'}")
    print("-" * 50)

    for attr_name, attr_idx in ATTRIBUTE_MAPPING.items():
        attr_folder = os.path.join(BASE_OUTPUT_DIR, attr_name)
        
        if not os.path.exists(attr_folder):
            continue
            
        generated_files = [f for f in os.listdir(attr_folder) if f.endswith(('.png', '.jpg'))]
        if not generated_files:
            continue

        lpips_scores = []
        classifier_scores_fake = []

        with torch.no_grad():
            for file_name in generated_files:
                original_name = file_name.split('_')[0] + ".jpg" 
                path_fake = os.path.join(attr_folder, file_name)
                path_real = os.path.join(INPUT_DIR, original_name)

                if not os.path.exists(path_real):
                    print(f"[WARNING] Real image not found for {file_name}, skipping.")
                    print(f"Expected real image path: {path_real}")
                    continue

                img_fake_pil = Image.open(path_fake).convert('RGB')
                img_real_pil = Image.open(path_real).convert('RGB')

                img_fake_tensor = transform(img_fake_pil).unsqueeze(0).to(device)
                img_real_tensor = transform(img_real_pil).unsqueeze(0).to(device)

                lpips_val = lpips_metric(img_real_tensor, img_fake_tensor)
                lpips_scores.append(lpips_val.item())

                t_zero = torch.zeros(1, device=device)
                logits_fake = classifier(img_fake_tensor, t_zero)
                prob_fake = torch.sigmoid(logits_fake)[0, attr_idx].item()
                classifier_scores_fake.append(prob_fake)

                img_fake_fid = (transform_fid(img_fake_pil).unsqueeze(0) * 255).byte().to(device)
                img_real_fid = (transform_fid(img_real_pil).unsqueeze(0) * 255).byte().to(device)
                fid_metric.update(img_real_fid, real=True)
                fid_metric.update(img_fake_fid, real=False)
                
                total_images_evaluated += 1

        avg_lpips = np.mean(lpips_scores)
        success_rate = np.mean([1 if p >= 0.5 else 0 for p in classifier_scores_fake]) * 100
        
        print(f"{attr_name.upper():<15} | {avg_lpips:.4f}   | {success_rate:>5.1f}%    | {len(generated_files)}")

    print("-" * 50)
    print(f"TOTAL IMAGES: {total_images_evaluated}")
    
    if total_images_evaluated >= 50:
        print("[INFO] Computing Global FID...")
        fid_score = fid_metric.compute().item()
        print(f"GLOBAL FID: {fid_score:.2f}")
    else:
        print("GLOBAL FID: N/A (Insufficient data)")

if __name__ == "__main__":
    evaluate_all_results()