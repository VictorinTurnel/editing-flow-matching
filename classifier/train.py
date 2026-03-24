import os
import torch
import torch.nn as nn
from model import TimeCondResNet18
from dataset import TimeCondCelebADataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

EPOCHS = 20
NUM_CLASSES = 10
DATASET = "../dataset/celebaHQ"
DATA_DIR = os.path.join(DATASET,"images")
ATTR_FILE = os.path.join(DATASET,"list_attr_celeba_hq.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device : {device}")

dataset = TimeCondCelebADataset(DATA_DIR, ATTR_FILE)
train_size = int(0.9 * dataset.__len__())
val_size = dataset.__len__() - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

model= TimeCondResNet18(num_classes=NUM_CLASSES).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):

    # Training
    model.train()
    progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training...")

    total_train_loss = 0
    for x_t, t, labels in progress_bar_train:
        x_t, t, labels = x_t.to(device), t.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(x_t,t)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss+= loss.item()
        progress_bar_train.set_postfix({"Loss":loss.item()})

    average_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Validation...")

    total_val_loss = 0
    total_correct_predictions = 0
    total_prediction = 0
    with torch.no_grad():

        for x_t, t, labels in progress_bar_val:
            x_t, t, labels = x_t.to(device), t.to(device), labels.to(device)

            outputs = model(x_t, t)
            loss = criterion(outputs, labels)
            total_val_loss+= loss.item()

            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            total_correct_predictions += (predicted_labels == labels).sum().item()
            total_prediction += labels.numel()

    average_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_correct_predictions / total_prediction

    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    torch.save(model.state_dict(), f"time_resnet18_epoch_{epoch+1}.pth")
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {average_train_loss:.4f} - Val Loss: {average_val_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - LR: {current_lr:.6f}")



