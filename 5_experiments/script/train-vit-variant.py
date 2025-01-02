import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import os
import time

# Parser để cấu hình tham số đầu vào
parser = argparse.ArgumentParser(description="Train or Test Pretrained Vision Transformer on CIFAR-100")
parser.add_argument('--train', type=int, default=1, help="1 for training, 0 for testing")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--model', type=str, default="vit_base_patch16_224", help="Pretrained model from timm")
parser.add_argument('--image_sz', type=int, default=224, help="Image size for resizing input")
parser.add_argument('--save', type=str, default="checkpoints", help="Base directory to save checkpoints")
parser.add_argument('--name', type=str, default="default", help="Unique name for this experiment")
parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume training")
parser.add_argument('--save_freq', type=int, default=1, help="Frequency of saving checkpoints (in epochs)")
parser.add_argument('--device', type=str, default='0', help="GPU devices to use, e.g., '0', '0,1', or 'cpu'")
args = parser.parse_args()

# Xử lý thiết bị từ tham số --device
if args.device == 'cpu':
    device = torch.device('cpu')
    device_ids = None
else:
    device_ids = [int(d) for d in args.device.split(',')]
    device = torch.device(f'cuda:{device_ids[0]}')

# Kiểm tra thiết bị
print(f"Using device(s): {args.device}")

# Tạo thư mục lưu trữ riêng cho mỗi trường hợp
save_dir = os.path.join(args.save, args.name)
os.makedirs(save_dir, exist_ok=True)
print(f"Results will be saved to: {save_dir}")

# Augmentation cho dữ liệu
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((args.image_sz, args.image_sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # CIFAR-100 mean và std
])

test_transform = transforms.Compose([
    transforms.Resize((args.image_sz, args.image_sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# Load dữ liệu
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Tạo mô hình pretrained từ timm
model = create_model(args.model, pretrained=True, num_classes=100)  # Pretrained, số lớp là 100 cho CIFAR-100

# Nếu có nhiều GPU, sử dụng DataParallel
if device_ids and len(device_ids) > 1:
    print(f"Using DataParallel with devices: {device_ids}")
    model = nn.DataParallel(model, device_ids=device_ids)

model = model.to(device)

# Hàm train
def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_accuracy = 0.0  # Lưu accuracy tốt nhất

    # Resume từ checkpoint nếu có
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resumed training from epoch {start_epoch}")

    print("Starting training...")
    total_training_time = 0  # Tổng thời gian huấn luyện
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start_time = time.time()  # Bắt đầu đo thời gian epoch
        # Sử dụng tqdm cho progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Cập nhật loss và accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Hiển thị loss và accuracy trên progress bar
            train_loader_tqdm.set_postfix(
                loss=running_loss/len(train_loader), 
                accuracy=100*correct/total,
                memory=f"{torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
            )

        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - epoch_start_time  # Thời gian epoch
        total_training_time += epoch_time
        print(f"Epoch [{epoch+1}/{args.epochs}] - Accuracy: {epoch_accuracy:.2f}%, Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")

        # Lưu checkpoint theo save_freq
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Lưu mô hình tốt nhất
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_path = os.path.join(save_dir, "best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }, best_path)
            print(f"Saved best model to {best_path} with accuracy: {best_accuracy:.2f}%")

    print(f"Total training time: {total_training_time:.2f}s")

# Hàm test
def test_model():
    # Load mô hình tốt nhất
    best_path = os.path.join(save_dir, "best.pt")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_path} with accuracy: {checkpoint['best_accuracy']:.2f}%")
    else:
        print("Best model not found. Please train the model first.")
        return

    model.eval()
    correct = 0
    total = 0

    # Sử dụng tqdm cho progress bar
    test_loader_tqdm = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Hiển thị accuracy trên progress bar
            test_loader_tqdm.set_postfix(
                accuracy=100*correct/total,
                memory=f"{torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
            )

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Chạy chương trình theo tham số --train
if __name__ == "__main__":
    if args.train == 1:
        train_model()
    else:
        test_model()
