import os
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
dataset_dir = './dataset'


# 1. 数据收集和准备
class ImageCollector:
    """图像收集器，用于下载和组织数据"""

    def __init__(self, save_dir='./dataset'):
        self.save_dir = save_dir
        os.makedirs(f"{save_dir}/positive", exist_ok=True)
        os.makedirs(f"{save_dir}/negative", exist_ok=True)

    def download_image(self, url, save_path):
        """下载单张图片"""
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(save_path)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False

    def collect_sample_data(self):
        """收集示例数据（这里使用本地图片路径作为示例）"""
        print("请将正类图片放在 ./dataset/positive 目录")
        print("请将负类图片放在 ./dataset/negative 目录")

        # 如果需要自动下载，可以添加URL列表
        # positive_urls = [...]
        # negative_urls = [...]


# 2. 自定义数据集类
class BinaryImageDataset(Dataset):
    """二分类图像数据集"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# 3. 数据预处理
def get_data_transforms():
    """获取数据转换"""
    # 训练时的数据增强
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    # 验证/测试时的转换
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    return train_transform, val_transform


# 4. 构建ResNet二分类模型
class ResNetBinaryClassifier(nn.Module):
    """基于ResNet的二分类器"""

    def __init__(self, pretrained=True, freeze_backbone=True):
        super(ResNetBinaryClassifier, self).__init__()

        # 加载预训练的ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # 获取特征维度
        num_features = self.backbone.fc.in_features

        # 替换最后的全连接层为二分类层
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),  # 二分类，输出1个值
            nn.Sigmoid()  # 输出概率
            )

        # 是否冻结主干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 解冻最后的分类层
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """解冻主干网络用于微调"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# 5. 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# 6. 验证函数
def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# 7. 完整的训练流程
def train_model(
        model, train_loader, val_loader, num_epochs, device,
        learning_rate=0.001, freeze_epochs=5
        ):
    """
    完整的训练流程
    freeze_epochs: 冻结主干网络训练的epochs数
    """
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # 在指定epoch后解冻主干网络
        if epoch == freeze_epochs and freeze_epochs > 0:
            print("\n解冻主干网络进行微调...")
            model.unfreeze_backbone()
            # 使用较小的学习率进行微调
            optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 学习率调整
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
        }


# 8. 可视化训练过程
def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['val_accs'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 9. 预测函数
def predict_image(model, image_path, transform, device):
    """对单张图片进行预测"""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0

    return prediction, probability


def predict_batch(model, dataset_dir, transform, device):
    """对批量图片进行预测"""
    print("\n测试预测其他图片...")
    test_images = [os.path.join(dataset_dir, 'test', img) for img in os.listdir(os.path.join(dataset_dir, 'test')) if
                   img.endswith(('.jpg', '.jpeg', '.png'))]
    for test_image in test_images:
        prediction, probability = predict_image(model, test_image, transform, device)
        print(f"图片: {test_image}")
        print(f"预测类别: {'正类' if prediction == 1 else '负类'}")
        print(f"置信度: {probability:.4f}")


# 10. 主函数
def main():
    # 设置参数
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    FREEZE_EPOCHS = 10  # 前10个epoch冻结主干网络
    LEARNING_RATE = 0.001

    # 准备数据
    print("准备数据...")

    # 收集图片路径和标签
    image_paths = []
    labels = []

    # 正类（标签为1）
    positive_dir = os.path.join(dataset_dir, 'positive')
    if os.path.exists(positive_dir):
        for img_name in os.listdir(positive_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(positive_dir, img_name))
                labels.append(1)

    # 负类（标签为0）
    negative_dir = os.path.join(dataset_dir, 'negative')
    if os.path.exists(negative_dir):
        for img_name in os.listdir(negative_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(negative_dir, img_name))
                labels.append(0)

    if len(image_paths) == 0:
        print("未找到图片！请确保在./dataset/positive和./dataset/negative目录中放置图片")
        return

    print(f"找到 {len(image_paths)} 张图片（正类: {sum(labels)}, 负类: {len(labels) - sum(labels)}）")

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

    # 获取数据转换
    train_transform, val_transform = get_data_transforms()

    # 创建数据集和数据加载器
    train_dataset = BinaryImageDataset(X_train, y_train, train_transform)
    val_dataset = BinaryImageDataset(X_val, y_val, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 创建模型
    print("\n创建模型...")
    model = ResNetBinaryClassifier(pretrained=True, freeze_backbone=True).to(device)

    # 训练模型
    print("\n开始训练...")
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        learning_rate=LEARNING_RATE,
        freeze_epochs=FREEZE_EPOCHS
        )

    # 绘制训练历史
    plot_training_history(history)

    # 保存模型
    torch.save(model.state_dict(), 'resnet_binary_classifier.pth')
    print("\n模型已保存为 resnet_binary_classifier.pth")

    # 测试预测
    print("\n测试预测...")
    test_image = X_val[0]  # 使用验证集的第一张图片测试
    prediction, probability = predict_image(model, test_image, val_transform, device)
    print(f"图片: {test_image}")
    print(f"预测类别: {'正类' if prediction == 1 else '负类'}")
    print(f"置信度: {probability:.4f}")

    # 批量预测


if __name__ == "__main__":
    # main()
    model_path = 'resnet_binary_classifier.pth'
    model = ResNetBinaryClassifier(pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    val_transform = get_data_transforms()[1]  # 获取验证转换
    print("\n批量预测测试集图片...")
    predict_batch(model, dataset_dir, val_transform, device)
