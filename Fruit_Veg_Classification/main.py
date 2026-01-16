import torch
import os
from PIL import Image
from torchvision import transforms

# 超参数
batch_size = 30
learning_rate = 0.001
num_epochs = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load data
folder_path = r"trainData"
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
NumClasses = len(subdirs)
print(f"Number of classes: {NumClasses}")


# 定义图像预处理操作（根据你的 CNN 输入要求调整）
transform = transforms.Compose([
    transforms.Resize((224, 224)),        # 调整图像大小（例如 ResNet 输入为 224x224）
    transforms.ToTensor(),                # 转换为 Tensor，并自动将像素值归一化到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化（可选）
])

# 读取图片
tensor_TrainData_list  = []
tensor_TrainLabel_list = []
CountImages = 0
for subdir in subdirs:
    subdir_path = os.path.join(folder_path, subdir)
    image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"子文件夹: {subdir}, 图片数量: {len(image_files)}")
    for image_file in image_files:
        image_path = os.path.join(subdir_path, image_file)
        image = Image.open(image_path).convert('RGB')  # 确保图像是 RGB 格式
        image = transform(image)  # 应用预处理
        tensor_TrainData_list.append(image)
        tensor_TrainLabel_list.append(CountImages)  # 这里可以根据需要存储标签
    CountImages += 1

# 将列表堆叠成一个大 tensor: [N, 3, H, W]
if tensor_TrainData_list:
    all_imagesData_tensor = torch.stack(tensor_TrainData_list, dim=0)  # shape: [N, 3, 224, 224]
    print("all_imagesData_tensor 形状:", all_imagesData_tensor.shape)
else:
    raise ValueError("没有成功加载任何图片！")

if tensor_TrainLabel_list:
    all_imagesLabel_tensor = torch.tensor(tensor_TrainLabel_list, dtype=torch.long)  # shape: [N]
    one_hot = torch.nn.functional.one_hot(all_imagesLabel_tensor, num_classes=NumClasses)
    one_hot = one_hot.to(torch.float32)
    print("all_imagesLabel_tensor 形状:", one_hot.shape)
else:
    raise ValueError("没有成功加载任何图片！")


from torch.utils.data import TensorDataset, DataLoader, random_split
# 创建 TensorDataset
dataset = TensorDataset(all_imagesData_tensor, one_hot)
# 划分训练集和验证集（例如 80% 训练，20% 验证）
train_size = int(0.8 * len(dataset))    
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 创建 SimpleCNN 模型实例
from models.SimpleCNN import SimpleCNN
model = SimpleCNN(input_channels=3, num_classes=NumClasses)
model = model.to(device)
import torch.optim as optim
criterion = torch.nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # 在验证集上评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            _, labelsValue = torch.max(labels.data, 1)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labelsValue).sum().item()
        val_accuracy = 100 * correct / total
        print(f'Validation Accuracy after epoch {epoch+1}: {val_accuracy:.2f}%')