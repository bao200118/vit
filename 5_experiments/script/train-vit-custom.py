import argparse
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Parser để cấu hình tham số đầu vào
parser = argparse.ArgumentParser(description="Train or Test Custom Vision Transformer on CIFAR-100")
parser.add_argument('--train', type=int, default=1, help="1 for training, 0 for testing")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--patch_size', type=int, default=16, help="Patch size for Vision Transformer")
parser.add_argument('--embed_dim', type=int, default=768, help="Embedding dimension for Vision Transformer")
parser.add_argument('--num_heads', type=int, default=12, help="Number of attention heads")
parser.add_argument('--depth', type=int, default=12, help="Number of transformer blocks")
parser.add_argument('--mlp_ratio', type=float, default=4.0, help="MLP ratio in the Transformer blocks")
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
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])  # CIFAR-100 mean và std
])

test_transform = transforms.Compose([
    transforms.Resize((args.image_sz, args.image_sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

# Load dữ liệu
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Tạo mô hình tùy chỉnh ViT
def create_model_instance():
    model_instance = VisionTransformer(
        img_size=args.image_sz,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=100,  # CIFAR-100 có 100 lớp
        qkv_bias=True
    )
    return model_instance

model = create_model_instance()

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

    # Theo dõi kết quả để vẽ biểu đồ
    training_accuracies = []
    training_losses = []

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
                loss=running_loss / len(train_loader),
                accuracy=100 * correct / total,
                memory=f"{torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
            )

        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time  # Thời gian epoch
        total_training_time += epoch_time
        training_accuracies.append(epoch_accuracy)
        training_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}] - Accuracy: {epoch_accuracy:.2f}%, "
              f"Loss: {epoch_loss:.4f}, Time: {str(timedelta(seconds=int(epoch_time)))}, "
              f"Memory: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")

        # Lưu checkpoint theo save_freq
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'depth': args.depth,
                'patch_size': args.patch_size,
                'image_sz': args.image_sz,
                'mlp_ratio': args.mlp_ratio
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
                'best_accuracy': best_accuracy,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'depth': args.depth,
                'patch_size': args.patch_size,
                'image_sz': args.image_sz,
                'mlp_ratio': args.mlp_ratio
            }, best_path)
            print(f"Saved best model to {best_path} with accuracy: {best_accuracy:.2f}%")

    print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
    # Lưu kết quả huấn luyện để vẽ biểu đồ
    np.save(os.path.join(save_dir, 'training_accuracies.npy'), training_accuracies)
    np.save(os.path.join(save_dir, 'training_losses.npy'), training_losses)

# Hàm test
def test_model():
    best_path = os.path.join(save_dir, "best.pt")
    if not os.path.exists(best_path):
        print("Best model not found. Please train the model first.")
        return

    checkpoint = torch.load(best_path, map_location=device)
    print(f"Loaded best model from {best_path} with accuracy: {checkpoint['best_accuracy']:.2f}%")

    # Lấy cấu hình mô hình từ checkpoint
    embed_dim = checkpoint.get('embed_dim', args.embed_dim)
    num_heads = checkpoint.get('num_heads', args.num_heads)
    depth = checkpoint.get('depth', args.depth)
    patch_size = checkpoint.get('patch_size', args.patch_size)
    image_sz = checkpoint.get('image_sz', args.image_sz)
    mlp_ratio = checkpoint.get('mlp_ratio', args.mlp_ratio)

    # Tạo lại mô hình với cấu hình phù hợp
    model = VisionTransformer(
        img_size=image_sz,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=100,  # CIFAR-100
        qkv_bias=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Theo dõi test loss cho mỗi batch
    test_losses = []

    # Sử dụng tqdm cho progress bar
    test_loader_tqdm = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            test_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    mean_loss = total_loss / len(test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%, Mean Loss: {mean_loss:.4f}")

    # Lưu kết quả test để vẽ biểu đồ
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)
    np.save(os.path.join(save_dir, 'test_accuracy.npy'), [accuracy])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Classification report
    class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
    average_accuracy = class_report['accuracy']
    print(f"Average Accuracy: {average_accuracy:.2f}")

    # Top 10 and Bottom 10 performing classes
    class_accuracies = {class_name: class_report[class_name]['precision'] for class_name in test_dataset.classes}
    sorted_class_accuracies = sorted(class_accuracies.items(), key=lambda item: item[1], reverse=True)

    top_10 = sorted_class_accuracies[:10]
    bottom_10 = sorted_class_accuracies[-10:]

    print("Top 10 Performing Classes:")
    for class_name, acc in top_10:
        print(f"{class_name}: {acc:.2f}")

    print("\nBottom 10 Performing Classes:")
    for class_name, acc in bottom_10:
        print(f"{class_name}: {acc:.2f}")

    # Loss distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_losses, bins=50, alpha=0.75)
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Test Loss Distribution')
    plt.savefig(os.path.join(save_dir, 'loss_distribution.png'))
    plt.close()

    # Save class accuracies
    with open(os.path.join(save_dir, 'class_accuracies.txt'), 'w') as f:
        f.write("Class Accuracies:\n")
        for class_name, acc in sorted_class_accuracies:
            f.write(f"{class_name}: {acc:.2f}\n")

    return accuracy, mean_loss, cm, class_report

# Hàm plot kết quả
def plot_results():
    training_accuracies = np.load(os.path.join(save_dir, 'training_accuracies.npy'))
    training_losses = np.load(os.path.join(save_dir, 'training_losses.npy'))
    test_losses = np.load(os.path.join(save_dir, 'test_losses.npy'))
    test_accuracy = np.load(os.path.join(save_dir, 'test_accuracy.npy'))[0]

    epochs = range(1, len(training_accuracies) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_accuracies, label="Training Accuracy")
    plt.hlines(test_accuracy, xmin=1, xmax=len(epochs), colors='r', linestyles='dashed', label="Test Accuracy")
    plt.title("Training and Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.hlines(np.mean(test_losses), xmin=1, xmax=len(epochs), colors='r', linestyles='dashed', label="Test Loss")
    plt.title("Training and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

# Hàm visualize attention maps
def visualize_attention():
    best_path = os.path.join(save_dir, "best.pt")
    if not os.path.exists(best_path):
        print("Best model not found. Please train the model first.")
        return

    checkpoint = torch.load(best_path, map_location=device)
    print(f"Loaded best model from {best_path} for attention visualization.")

    # Lấy cấu hình mô hình từ checkpoint
    embed_dim = checkpoint.get('embed_dim', args.embed_dim)
    num_heads = checkpoint.get('num_heads', args.num_heads)
    depth = checkpoint.get('depth', args.depth)
    patch_size = checkpoint.get('patch_size', args.patch_size)
    image_sz = checkpoint.get('image_sz', args.image_sz)
    mlp_ratio = checkpoint.get('mlp_ratio', args.mlp_ratio)

    # Tạo lại mô hình với cấu hình phù hợp
    model = VisionTransformer(
        img_size=image_sz,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=100,  # CIFAR-100
        qkv_bias=True,
        return_attn=True  # Thêm tham số để trả về attention
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Lấy một mẫu từ test set
    input_image, _ = test_dataset[0]
    input_image = input_image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_image)
        attn_weights = outputs['attn_weights']  # Assuming model returns attention weights

    # Chỉ visualize attention của lớp đầu tiên
    attn = attn_weights[0][0]  # [layers, batch_size, num_heads, tokens, tokens]

    num_heads = attn.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
    for i in range(num_heads):
        attn_map = attn[i].cpu().numpy()
        sns.heatmap(attn_map, ax=axes[i], cmap='viridis', cbar=False)
        axes[i].set_title(f'Head {i+1}')
        axes[i].axis('off')
    plt.suptitle('Attention Maps of the First Layer')
    plt.savefig(os.path.join(save_dir, 'attention_maps.png'))
    plt.close()

# Chạy chương trình theo tham số --train
if __name__ == "__main__":
    if args.train == 1:
        train_model()
        test_model()
        plot_results()
    else:
        test_model()
        plot_results()
        visualize_attention()

































####################################################
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from timm.models.vision_transformer import VisionTransformer
# from tqdm import tqdm
# import os
# import time

# # Parser để cấu hình tham số đầu vào
# parser = argparse.ArgumentParser(description="Train or Test Custom Vision Transformer on CIFAR-100")
# parser.add_argument('--train', type=int, default=1, help="1 for training, 0 for testing")
# parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
# parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
# parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
# parser.add_argument('--patch_size', type=int, default=16, help="Patch size for Vision Transformer")
# parser.add_argument('--embed_dim', type=int, default=768, help="Embedding dimension for Vision Transformer")
# parser.add_argument('--num_heads', type=int, default=12, help="Number of attention heads")
# parser.add_argument('--depth', type=int, default=12, help="Number of transformer blocks")
# parser.add_argument('--mlp_ratio', type=float, default=4.0, help="MLP ratio in the Transformer blocks")
# parser.add_argument('--image_sz', type=int, default=224, help="Image size for resizing input")
# parser.add_argument('--save', type=str, default="checkpoints", help="Base directory to save checkpoints")
# parser.add_argument('--name', type=str, default="default", help="Unique name for this experiment")
# parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume training")
# parser.add_argument('--save_freq', type=int, default=1, help="Frequency of saving checkpoints (in epochs)")
# parser.add_argument('--device', type=str, default='0', help="GPU devices to use, e.g., '0', '0,1', or 'cpu'")
# args = parser.parse_args()

# # Xử lý thiết bị từ tham số --device
# if args.device == 'cpu':
#     device = torch.device('cpu')
#     device_ids = None
# else:
#     device_ids = [int(d) for d in args.device.split(',')]
#     device = torch.device(f'cuda:{device_ids[0]}')

# # Kiểm tra thiết bị
# print(f"Using device(s): {args.device}")

# # Tạo thư mục lưu trữ riêng cho mỗi trường hợp
# save_dir = os.path.join(args.save, args.name)
# os.makedirs(save_dir, exist_ok=True)
# print(f"Results will be saved to: {save_dir}")

# # Augmentation cho dữ liệu
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize((args.image_sz, args.image_sz)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # CIFAR-100 mean và std
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((args.image_sz, args.image_sz)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
# ])

# # Load dữ liệu
# train_dataset = datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
# test_dataset = datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# # Tạo mô hình tùy chỉnh ViT
# model = VisionTransformer(
#     img_size=args.image_sz,
#     patch_size=args.patch_size,
#     embed_dim=args.embed_dim,
#     depth=args.depth,
#     num_heads=args.num_heads,
#     mlp_ratio=args.mlp_ratio,
#     num_classes=100,  # CIFAR-100 có 100 lớp
#     qkv_bias=True
# )

# # Nếu có nhiều GPU, sử dụng DataParallel
# if device_ids and len(device_ids) > 1:
#     print(f"Using DataParallel with devices: {device_ids}")
#     model = nn.DataParallel(model, device_ids=device_ids)

# model = model.to(device)

# # Hàm train
# def train_model():
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     start_epoch = 0
#     best_accuracy = 0.0  # Lưu accuracy tốt nhất

#     # Resume từ checkpoint nếu có
#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
#         best_accuracy = checkpoint.get('best_accuracy', 0.0)
#         print(f"Resumed training from epoch {start_epoch}")

#     print("Starting training...")
#     total_training_time = 0  # Tổng thời gian huấn luyện
#     for epoch in range(start_epoch, args.epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         epoch_start_time = time.time()  # Bắt đầu đo thời gian epoch
#         # Sử dụng tqdm cho progress bar
#         train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
#         for inputs, labels in train_loader_tqdm:
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # Cập nhật loss và accuracy
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             # Hiển thị loss và accuracy trên progress bar
#             train_loader_tqdm.set_postfix(
#                 loss=running_loss/len(train_loader), 
#                 accuracy=100*correct/total,
#                 memory=f"{torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
#             )

#         epoch_accuracy = 100 * correct / total
#         epoch_time = time.time() - epoch_start_time  # Thời gian epoch
#         total_training_time += epoch_time
#         print(f"Epoch [{epoch+1}/{args.epochs}] - Accuracy: {epoch_accuracy:.2f}%, Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")

#         # Lưu checkpoint theo save_freq
#         if (epoch + 1) % args.save_freq == 0:
#             checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_accuracy': best_accuracy
#             }, checkpoint_path)
#             print(f"Saved checkpoint to {checkpoint_path}")

#         # Lưu mô hình tốt nhất
#         if epoch_accuracy > best_accuracy:
#             best_accuracy = epoch_accuracy
#             best_path = os.path.join(save_dir, "best.pt")
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_accuracy': best_accuracy
#             }, best_path)
#             print(f"Saved best model to {best_path} with accuracy: {best_accuracy:.2f}%")

#     print(f"Total training time: {total_training_time:.2f}s")

# # Hàm test
# def test_model():
#     # Load mô hình tốt nhất
#     best_path = os.path.join(save_dir, "best.pt")
#     if os.path.exists(best_path):
#         checkpoint = torch.load(best_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Loaded best model from {best_path} with accuracy: {checkpoint['best_accuracy']:.2f}%")
#     else:
#         print("Best model not found. Please train the model first.")
#         return

#     model.eval()
#     correct = 0
#     total = 0

#     # Sử dụng tqdm cho progress bar
#     test_loader_tqdm = tqdm(test_loader, desc="Testing")
#     with torch.no_grad():
#         for inputs, labels in test_loader_tqdm:
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             # Hiển thị accuracy trên progress bar
#             test_loader_tqdm.set_postfix(
#                 accuracy=100*correct/total,
#                 memory=f"{torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
#             )

#     print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# # Chạy chương trình theo tham số --train
# if __name__ == "__main__":
#     if args.train == 1:
#         train_model()
#     else:
#         test_model()
