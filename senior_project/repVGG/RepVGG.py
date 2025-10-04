import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
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

class RepVGGBlock(nn.Module):
    """
    RepVGG 基本塊 - 官方實現版本
    訓練時有三個分支：3x3 conv, 1x1 conv, identity
    推理時合併為單一 3x3 conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=True, 
                                       padding_mode=padding_mode)

        else:
            # Identity branch (只有當 in_channels == out_channels 且 stride == 1 時才有)
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            # Dense 3x3 branch
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, 
                                   groups=groups)

            # 1x1 branch
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, 
                                 kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """獲取等效的kernel和bias"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """轉換到推理模式"""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, 
                                   out_channels=self.rbr_dense.conv.out_channels,
                                   kernel_size=self.rbr_dense.conv.kernel_size, 
                                   stride=self.rbr_dense.conv.stride,
                                   padding=self.rbr_dense.conv.padding, 
                                   dilation=self.rbr_dense.conv.dilation, 
                                   groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    """Conv + BN 組合"""
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding, 
                                      dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGModel(nn.Module):
    def __init__(self, num_classes=5, deploy=False):
        super(RepVGGModel, self).__init__()
        self.deploy = deploy
        self.num_classes = num_classes
        
        # stage0: 48 channels
        self.in_planes = 48
        self.stage0 = RepVGGBlock(3, self.in_planes, kernel_size=3, stride=2, padding=1, deploy=deploy)

        # stage1~4 - 使用檢查點中的通道數配置
        self.stage1 = self._make_stage(48, 2, stride=2)    # 48 channels, 2 blocks
        self.stage2 = self._make_stage(96, 4, stride=2)    # 96 channels, 4 blocks  
        self.stage3 = self._make_stage(192, 14, stride=2)  # 192 channels, 14 blocks
        self.stage4 = self._make_stage(1280, 1, stride=2)  # 1280 channels, 1 block

        # Global Average Pooling + FC
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(1280, num_classes)
        self._initialize_weights()

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(self.in_planes, planes, kernel_size=3, stride=stride, padding=1, deploy=self.deploy))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def switch_to_deploy(self):
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()

def f1_score_metric(preds, targets, num_classes=5):
    preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return f1_score(targets, preds, average='macro', zero_division=0)

def train_model(model, train_loader, criterion, optimizer, num_epochs, update_training_progress=None, val_loader=None, patience=15, device='cuda'):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=5)

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