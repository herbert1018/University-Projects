import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import copy
import time

def get_dataloader(dataset_path, batch_size=8):  # 減小batch size增加隨機性
    # 大幅增強數據擴充
    train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomRotation(degrees=15),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = ImageFolder(root=f"{dataset_path}/train", transform=train_transform)
    test_dataset = ImageFolder(root=f"{dataset_path}/test", transform=eval_transform)
    val_dataset = ImageFolder(root=f"{dataset_path}/val", transform=eval_transform)
    
    print(f"類別: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)

    return train_loader, test_loader, val_loader

class ResNetModel(nn.Module):
    def __init__(self, num_classes=5, weights=ResNet18_Weights.DEFAULT):
        super(ResNetModel, self).__init__()
        # 載入 ResNet18
        self.model = resnet18(weights=weights)
        self.num_classes = num_classes
        # 取出最後全連接層的輸入維度
        in_features = self.model.fc.in_features
        # 替換分類層
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def f1_score_metric(preds, targets, num_classes=5):
    preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return f1_score(targets, preds, average='macro', zero_division=0)

def train_model(model, train_loader, criterion, optimizer, num_epochs, update_training_progress=None, val_loader=None, patience=15, device='cuda'):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-6)

    val_loss_history, val_acc_history, val_f1_history, train_loss_history = [], [], [], []
    model.to(device)
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_batches = len(train_loader)
        all_preds_epoch = []
        all_labels_epoch = []

        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            if update_training_progress and batch_idx % 2 == 0:
                update_training_progress(epoch, num_epochs, batch_idx, total_batches, None, None, None, None, False)
            all_preds_epoch.append(preds.cpu())
            all_labels_epoch.append(labels.cpu())

        val_loss, val_acc, val_f1 = None, None, None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).long()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    all_preds.append(outputs)
                    all_labels.append(labels)
            val_loss = val_running_loss / len(val_loader)
            val_acc = correct / total if total > 0 else 0
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            val_f1 = f1_score_metric(all_preds, all_labels, model.num_classes)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            val_f1_history.append(val_f1)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        if update_training_progress:
            update_training_progress(epoch, num_epochs, len(train_loader), len(train_loader), epoch_loss, val_loss, val_acc, val_f1, True)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\033[96mEpoch\033[0m {epoch:>2}/{num_epochs}, \033[96mLR:\033[0m {current_lr:.6f}, '
              f'\033[38;5;180mTrain Loss\033[0m={epoch_loss:.4f}, \033[38;5;180mAcc\033[0m={epoch_acc:.4f}, '
              f'\033[38;5;180mVal Loss\033[0m={val_loss:.4f}, \033[38;5;180mVal Acc\033[0m={val_acc:.4f}, \033[38;5;180mVal F1\033[0m={val_f1:.4f}')
        
        if epoch % 5 == 0:
            all_preds_epoch = torch.cat(all_preds_epoch)
            all_labels_epoch = torch.cat(all_labels_epoch)
            pred_unique, pred_counts = np.unique(all_preds_epoch.numpy(), return_counts=True)
            label_unique, label_counts = np.unique(all_labels_epoch.numpy(), return_counts=True)
            print(f"Train Pred: {dict(zip(pred_unique, pred_counts))}")
            print(f"Train True: {dict(zip(label_unique, label_counts))}")

        if val_f1 is not None and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif val_f1 is not None:
            patience_counter += 1
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch} (patience={patience})')
            break
        torch.cuda.empty_cache()

    time_elapsed = time.time() - start_time
    minutes = int(time_elapsed // 60)
    seconds = int(time_elapsed % 60)
    print(f'\nTraining completed in {minutes}m {seconds}s')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, best_val_acc, minutes, seconds, val_loss_history, val_acc_history, val_f1_history, train_loss_history

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Model loaded from {path}")
    return model