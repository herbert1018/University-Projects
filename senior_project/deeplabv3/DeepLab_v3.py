import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import glob
import os
import resnet
from torch.utils.data import get_worker_info
import copy
import time
import random
import timm

# 在頂層定義 SegmentationDataset
class SegmentationDataset(torch.utils.data.Dataset):
    _print_once = False
    def __init__(self, img_paths, mask_paths, transform=None, mask_transform=None):
        # 支援多種圖片格式
        self.img_paths = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            self.img_paths.extend(glob.glob(os.path.join(img_paths, f'*{ext}')))
        self.img_paths = sorted(self.img_paths)
        
        # 檢查對應的遮罩檔案
        self.mask_paths = []
        for img_path in self.img_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_paths, f"{base_name}.png")  # 遮罩都是PNG
            if os.path.exists(mask_path):
                self.mask_paths.append(mask_path)
            else:
                print(f"警告: 找不到遮罩檔案 {mask_path}")
        
        # 過濾掉沒有對應遮罩的圖片
        valid_pairs = list(zip(self.img_paths, self.mask_paths))
        if len(valid_pairs) == 0:
            raise ValueError(f"在 {img_paths} 中找不到有效的圖片-遮罩配對")
            
        self.img_paths, self.mask_paths = zip(*valid_pairs)
        self.transform = transform
        
        if not SegmentationDataset._print_once and get_worker_info() is None:
            print(f"載入了 {len(self.img_paths)} 對圖片-遮罩配對")
            SegmentationDataset._print_once = True

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            img, mask = self.transform(img, mask)

        # 將 mask 轉成 LongTensor 並確保值合法
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).long()  # 只要大於0視為1

        mask = mask.squeeze()
        return img, mask


def transform_img_mask(img, mask):
    """建立圖片與遮罩的共同轉換，確保隨機增強一致"""
    # Resize
    img = TF.resize(img, (512, 512))
    mask = TF.resize(mask, (512, 512), interpolation=transforms.InterpolationMode.NEAREST)
    
    """
    if random.random() > 0.5:
        angle = random.uniform(-7, 7)
        img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
    
    # 隨機翻轉
    if random.random() > 0.5:
        img = TF.hflip(img); mask = TF.hflip(mask)
    """
    # 隨機亮度/對比度
    if random.random() > 0.5:
        img = TF.adjust_brightness(img, 0.9 + 0.2 * random.random())
    if random.random() > 0.5:
        img = TF.adjust_contrast(img, 0.9 + 0.2 * random.random())
    
    # To Tensor
    img = TF.to_tensor(img)
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    # Normalize (只對 image)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])

    return img, mask

def eval_transform(img, mask):
    """評估階段的轉換，不含隨機增強"""
    img = TF.resize(img, (512, 512))
    mask = TF.resize(mask, (512, 512), interpolation=transforms.InterpolationMode.NEAREST)
    img  = TF.to_tensor(img)
    img  = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    return img, mask

def get_dataloader(train_images, train_masks, val_images, val_masks, test_images, test_masks, batch_size):
    """建立資料加載器"""

    train_dataset = SegmentationDataset(train_images, train_masks, transform=transform_img_mask)
    val_dataset   = SegmentationDataset(val_images, val_masks, transform=eval_transform)
    test_dataset  = SegmentationDataset(test_images, test_masks, transform=eval_transform)

    return {
        'train': data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True),
        'val':   data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True),
        'test':  data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    }

class ASPPConv(nn.Sequential):
    """卷積模組: 不同尺度特徵提取"""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    """池化模組: 捕捉全局上下文"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """整合多尺度特徵的模組"""
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        
        # 1x1 convolution branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        
        # Atrous convolution branches
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global average pooling branch
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Project the concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabDecoder(nn.Module):
    """解碼器: 特徵融合與分割"""
    def __init__(self, low_level_channels, num_classes):
        super(DeepLabDecoder, self).__init__()
        
        # 保持原始層命名以匹配預訓練權重
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level_feat):
        # 低層特徵預處理（官方設計的關鍵步驟）
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        # 上採樣高層特徵以匹配低層特徵尺寸
        x = F.interpolate(x, size=low_level_feat.shape[2:], 
                         mode='bilinear', align_corners=False)
        
        # 特徵拼接 (256 + 48 = 304)
        x = torch.cat((x, low_level_feat), dim=1)
        
        return self.output(x)

class XceptionBackbone(nn.Module):
    """修改的 Xception Backbone，支援多層輸出"""
    def __init__(self, output_stride=16):
        super(XceptionBackbone, self).__init__()
        
        # 使用 timm 創建 Xception
        self.backbone = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        
        # 獲取特徵層的輸出通道數
        feature_info = self.backbone.feature_info
        self.low_level_channels = feature_info[1]['num_chs']  # 通常是第2層作為低層特徵
        self.high_level_channels = feature_info[-1]['num_chs']  # 最後一層作為高層特徵
        
    def forward(self, x):
        features = self.backbone(x)
        # 返回低層特徵（index=1）和高層特徵（index=-1）
        return {
            'low_level': features[1],    # 低層特徵，尺寸約為 H/4, W/4
            'out': features[-1]          # 高層特徵，尺寸約為 H/16, W/16
        }

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ 主架構 (可選 backbone)"""
    def __init__(self, num_classes=2, output_stride=16, backbone_type='resnet', resnet_number=50):
        super(DeepLabV3Plus, self).__init__()
        
        # 根據 backbone_type 選擇骨幹網路
        if backbone_type == 'resnet':
            self.backbone = resnet.ResnetChoose(output_stride=output_stride, number=resnet_number)
            # 根據常見 ResNet 結構自動設置通道數
            if resnet_number in [50, 101, 152]:
                self.low_level_channels = 256  # layer1 輸出
                self.high_level_channels = 2048  # layer4 輸出
            elif resnet_number in [34, 18]:
                self.low_level_channels = 64  # layer1 輸出
                self.high_level_channels = 512  # layer4 輸出
            else:
                raise ValueError(f"不支援的 resnet_number: {resnet_number}")
        elif backbone_type == 'xception':
            self.backbone = XceptionBackbone(output_stride=output_stride)
            self.low_level_channels = self.backbone.low_level_channels
            self.high_level_channels = self.backbone.high_level_channels
        else:
            raise ValueError("backbone_type 必須是 'resnet' 或 'xception'")
        
        # ASPP 空洞率配置
        if output_stride == 16:
            aspp_dilate = [6, 12, 18]
        elif output_stride == 8:
            aspp_dilate = [12, 24, 36]
        else:
            raise ValueError("output_stride must be 8 or 16")

        # ASPP 模組
        self.aspp = ASPP(in_channels=self.high_level_channels, atrous_rates=aspp_dilate, out_channels=256)
        
        # 解碼器
        self.decoder = DeepLabDecoder(low_level_channels=self.low_level_channels, num_classes=num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """權重初始化 - 使用 He 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 骨幹網路特徵提取
        features = self.backbone(x)
        if isinstance(features, dict):
            low_level_feat = features["low_level"]
            high_level_feat = features["out"]
        else:
            # 假設 ResNet 回傳 list: [layer1, layer2, layer3, layer4]
            low_level_feat = features[0]  # layer1
            high_level_feat = features[-1]  # layer4
        
        # ASPP 多尺度特徵提取
        aspp_out = self.aspp(high_level_feat)     # [B, 256, H/16, W/16]
        
        # 解碼器特徵融合
        out = self.decoder(aspp_out, low_level_feat)
        
        # 最終上採樣到輸入尺寸
        out = F.interpolate(out, size=input_shape, 
                           mode='bilinear', align_corners=False)
        
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs,
                update_callback=None, val_loader=None, patience=5,
                device='cuda'):
    
    best_dice = 0.0  # 改为跟踪最佳 Dice Score
    start_time = time.time()
    scaler = torch.amp.GradScaler(enabled=(device=="cuda"))
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)

    for epoch in range(1, num_epochs + 1):
        # ====== 訓練階段 ======
        model.train()
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            # Batch callback
            if update_callback:
                update_callback(epoch, num_epochs, batch_idx, len(train_loader))

        epoch_train_loss = running_loss / len(train_loader)

        # ====== 驗證階段 ======
        val_loss = None
        val_acc = None
        val_dice = None
        epoch_iou = 0.0

        if val_loader is not None:
            model.eval()
            val_ious = []
            val_running_loss = 0.0
            val_batches = 0
            correct = 0
            total = 0
            dice_scores = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_running_loss += loss.item()
                    val_batches += 1
                    preds = outputs.argmax(1)

                    # Accuracy
                    correct += (preds == masks).sum().item()
                    total += masks.numel()

                    # Dice Score (foreground=1)
                    intersection = ((preds == 1) & (masks == 1)).sum().item()
                    pred_sum = (preds == 1).sum().item()
                    true_sum = (masks == 1).sum().item()
                    dice = (2.0 * intersection) / (pred_sum + true_sum) if (pred_sum + true_sum) > 0 else 0
                    dice_scores.append(dice)

                    # IoU
                    for pred, target in zip(preds, masks):
                        inter = ((pred == 1) & (target == 1)).sum().item()
                        uni = ((pred == 1) | (target == 1)).sum().item()
                        if uni > 0:
                            val_ious.append(inter / uni)

            val_loss = val_running_loss / val_batches if val_batches > 0 else None
            val_acc = correct / total if total > 0 else None
            val_dice = np.mean(dice_scores) if dice_scores else 0.0  # 确保有默认值
            epoch_iou = np.mean(val_ious) if val_ious else 0

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f'\033[96mEpoch\033[0m {epoch:>2}, \033[96mLR:\033[0m {current_lr:.6f}: '
                  f'\033[38;5;180mTrain Loss\033[0m={epoch_train_loss:.4f}, '
                  f'\033[38;5;180mVal Loss\033[0m={val_loss:.4f}, '
                  f'\033[38;5;180mVal IoU\033[0m={epoch_iou:.4f}, '
                  f'\033[38;5;180mVal Dice\033[0m={val_dice:.4f}, '
                  f'\033[38;5;180mVal Acc\033[0m={val_acc*100:.2f}%,')

            # Early Stopping - 改为使用 Dice Score
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch:>2}. "
                      f"Best Dice Score: {best_dice:.4f} at epoch {best_epoch}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        # Epoch callback: 只更新 epoch_train_loss/val_loss/acc/dice 曲線
        if update_callback:
            update_callback(epoch, num_epochs, len(train_loader), len(train_loader),
                            train_loss=epoch_train_loss, val_loss=val_loss, val_acc=val_acc, val_dice=val_dice)

    total_time = time.time() - start_time
    return model, best_dice, int(total_time // 60), int(total_time % 60)


def save_model(model, path):
    """儲存模型"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """載入模型"""
    model.load_state_dict(torch.load(path))
    return model