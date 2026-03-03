import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode


class TimeCondCelebADataset(Dataset):

    def __init__(self, data_dir, attr_file):
        self.data_dir = data_dir

        with open(attr_file, 'r') as f:
            lines = f.readlines()

        headers = lines[1].strip().split()
        self.attr_idx = {
            'eyeglasses' : headers.index('Eyeglasses'),
            'male' : headers.index('Male'),
            'smiling' : headers.index('Smiling'),
            'hat' : headers.index('Wearing_Hat'),
            'mustache' : headers.index('Mustache'),
            'beard' : headers.index('No_Beard'),
            'pale_Skin' : headers.index('Pale_Skin'),
            'blond' : headers.index('Blond_Hair'),
            'attractive' : headers.index('Attractive')
        }

        self.images_info = []
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            attrs = [1 if x == "1" else 0 for x in parts[1:]]
            self.images_info.append((filename, attrs))


        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, idx):
        img_name, img_attr = self.images_info[idx]

        img_path = os.path.join(self.data_dir, str(int(img_name.split(".")[0])) + ".jpg")
        x_1 = Image.open(img_path).convert('RGB')
        x_1 = self.transform(x_1)

        labels = torch.zeros(10, dtype=torch.float32)
        labels[0] = img_attr[self.attr_idx["eyeglasses"]]
        labels[1] = img_attr[self.attr_idx["male"]]
        labels[2] = 1.0-img_attr[self.attr_idx["male"]]
        labels[3] = img_attr[self.attr_idx["smiling"]]
        labels[4] = img_attr[self.attr_idx["hat"]]
        labels[5] = img_attr[self.attr_idx["mustache"]]
        labels[6] = 1.0-img_attr[self.attr_idx["beard"]]
        labels[7] = img_attr[self.attr_idx["pale_Skin"]]
        labels[8] = img_attr[self.attr_idx["blond"]]
        labels[9] = img_attr[self.attr_idx["attractive"]]

        t = torch.rand(1).item() * 0.999 + 0.001
        x_0 = torch.randn_like(x_1)
        x_t = t * x_1 + (1 - t)*x_0

        return x_t, torch.tensor([t], dtype=torch.float32), labels
    

class TimeCondZapposDataset(Dataset):

    def __init__(self, data_dir, attr_file):
        self.data_dir = data_dir

        with open(attr_file, 'r') as f:
            lines = f.readlines()

        headers = lines[1].strip().split()
        
        self.attr_idx = {
            'Boots' : headers.index('Boots'),
            'Sandals' : headers.index('Sandals'),
            'Sneakers' : headers.index('Sneakers'),
            'High_Heels' : headers.index('High_Heels'),
            'Flats' : headers.index('Flats'),
            'Leather' : headers.index('Leather'),
            'Laces' : headers.index('Laces'),
            'Slip_on' : headers.index('Slip_on'),
            'Men' : headers.index('Men'),
            'Women' : headers.index('Women')
        }

        self.images_info = []
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            # Convertit "1" en 1 et "-1" en 0
            attrs = [1 if x == "1" else 0 for x in parts[1:]]
            self.images_info.append((filename, attrs))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, idx):
        img_name, img_attr = self.images_info[idx]

        img_path = os.path.join(self.data_dir, img_name)
        x_1 = Image.open(img_path).convert('RGB')
        x_1 = self.transform(x_1)

        labels = torch.zeros(10, dtype=torch.float32)
        labels[0] = img_attr[self.attr_idx["Boots"]]
        labels[1] = img_attr[self.attr_idx["Sandals"]]
        labels[2] = img_attr[self.attr_idx["Sneakers"]]
        labels[3] = img_attr[self.attr_idx["High_Heels"]]
        labels[4] = img_attr[self.attr_idx["Flats"]]
        labels[5] = img_attr[self.attr_idx["Leather"]]
        labels[6] = img_attr[self.attr_idx["Laces"]]
        labels[7] = img_attr[self.attr_idx["Slip_on"]]
        labels[8] = img_attr[self.attr_idx["Men"]]
        labels[9] = img_attr[self.attr_idx["Women"]]

        t = torch.rand(1).item() * 0.999 + 0.001
        x_0 = torch.randn_like(x_1)
        x_t = t * x_1 + (1 - t) * x_0

        return x_t, torch.tensor([t], dtype=torch.float32), labels