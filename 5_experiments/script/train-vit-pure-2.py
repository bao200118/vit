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
from timm import create_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Parser to configure input parameters
parser = argparse.ArgumentParser(description="Train or Test Pretrained ViT on CIFAR-100")
parser.add_argument('--train', type=int, default=1, help="1 for training, 0 for testing")
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and testing")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--model_name', type=str, default="vit_base_patch16_224", help="Pretrained ViT model name from timm")
parser.add_argument('--image_sz', type=int, default=224, help="Image size for resizing input")
parser.add_argument('--save', type=str, default="checkpoints", help="Base directory to save checkpoints")
parser.add_argument('--name', type=str, default="vit_pretrained", help="Unique name for this experiment")
parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume training")
parser.add_argument('--save_freq', type=int, default=1, help="Frequency of saving checkpoints (in epochs)")
parser.add_argument('--device', type=str, default='0', help="GPU devices to use, e.g., '0', '0,1', or 'cpu'")
args = parser.parse_args()

# Handle device from --device parameter
if args.device == 'cpu':
    device = torch.device('cpu')
    device_ids = None
else:
    device_ids = [int(d) for d in args.device.split(',')]
    device = torch.device(f'cuda:{device_ids[0]}')

# Check device
print(f"Using device(s): {args.device}")

# Create unique save directory for the experiment
save_dir = os.path.join(args.save, args.name)
os.makedirs(save_dir, exist_ok=True)
print(f"Results will be saved to: {save_dir}")

# Create metrics directory
metrics_dir = os.path.join(save_dir, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((args.image_sz, args.image_sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])  # CIFAR-100 mean and std
])

test_transform = transforms.Compose([
    transforms.Resize((args.image_sz, args.image_sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

# Load datasets
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Global variable to store attention weights
attention_weights = []

# Hook function to capture attention weights
def get_attention_weights(module, input, output):
    # For ViT models in timm, output[1] may contain attention weights
    # Depending on the model, you might need to adjust this
    attention_weights.append(output.attn_probs.cpu().detach())

# Create pretrained ViT model
def create_model_instance():
    model_instance = create_model(
        args.model_name,
        pretrained=True,
        num_classes=100  # CIFAR-100 has 100 classes
    )
    return model_instance

model = create_model_instance()

# If multiple GPUs, use DataParallel
if device_ids and len(device_ids) > 1:
    print(f"Using DataParallel with devices: {device_ids}")
    model = nn.DataParallel(model, device_ids=device_ids)

model = model.to(device)

# Training function
def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_accuracy = 0.0  # To save the best accuracy

    # Track metrics for plotting
    training_accuracies = []
    training_losses = []

    # Resume from checkpoint if available
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resumed training from epoch {start_epoch}")

    print("Starting training...")
    total_training_time = 0  # Total training time
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start_time = time.time()  # Start epoch timer
        # Use tqdm for progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Display loss and accuracy on progress bar
            train_loader_tqdm.set_postfix(
                loss=running_loss / len(train_loader),
                accuracy=100 * correct / total,
                memory=f"{torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
            )

        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time  # Epoch time
        total_training_time += epoch_time
        training_accuracies.append(epoch_accuracy)
        training_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}] - Accuracy: {epoch_accuracy:.2f}%, "
              f"Loss: {epoch_loss:.4f}, Time: {str(timedelta(seconds=int(epoch_time)))}, "
              f"Memory: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")

        # Save checkpoint per save_freq
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'model_name': args.model_name,
                'image_sz': args.image_sz
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_path = os.path.join(save_dir, "best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'model_name': args.model_name,
                'image_sz': args.image_sz
            }, best_path)
            print(f"Saved best model to {best_path} with accuracy: {best_accuracy:.2f}%")

    print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
    # Save training results for plotting
    np.save(os.path.join(metrics_dir, 'training_accuracies.npy'), training_accuracies)
    np.save(os.path.join(metrics_dir, 'training_losses.npy'), training_losses)

# Testing function
def test_model():
    best_path = os.path.join(save_dir, "best.pt")
    if not os.path.exists(best_path):
        print("Best model not found. Please train the model first.")
        return

    checkpoint = torch.load(best_path, map_location=device)
    print(f"Loaded best model from {best_path} with accuracy: {checkpoint['best_accuracy']:.2f}%")

    # Get model configuration from checkpoint
    model_name = checkpoint.get('model_name', args.model_name)
    image_sz = checkpoint.get('image_sz', args.image_sz)

    # Recreate the model with the correct configuration
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=100
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

    # Track test loss per batch
    test_losses = []

    # Use tqdm for progress bar
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

    # Save test results for plotting
    np.save(os.path.join(metrics_dir, 'test_losses.npy'), test_losses)
    np.save(os.path.join(metrics_dir, 'test_accuracy.npy'), [accuracy])

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(metrics_dir, 'confusion_matrix.png'))
    plt.close()

    # Classification report
    class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
    average_accuracy = class_report['accuracy']
    mean_loss_formatted = f"{mean_loss:.4f}"
    print(f"Average Accuracy: {average_accuracy:.2f}")

    # Top 10 and Bottom 10 performing classes
    class_accuracies = {class_name: class_report[class_name]['precision'] for class_name in test_dataset.classes}
    sorted_class_accuracies = sorted(class_accuracies.items(), key=lambda item: item[1], reverse=True)

    top_10 = sorted_class_accuracies[:10]
    bottom_10 = sorted_class_accuracies[-10:]

    # Format class accuracies for saving
    class_accuracies_text = "Model Evaluation Results\n"
    class_accuracies_text += "=" * 30 + "\n\n"
    class_accuracies_text += f"Model path: {best_path}\n\n"
    class_accuracies_text += "Overall Metrics:\n"
    class_accuracies_text += "-" * 30 + "\n"
    class_accuracies_text += f"Total Accuracy: {accuracy/100:.4f}\n"
    class_accuracies_text += f"Mean Loss: {mean_loss_formatted}\n\n"

    class_accuracies_text += "Top 10 Performing Classes:\n"
    for idx, (class_name, acc) in enumerate(top_10, 1):
        class_accuracies_text += f"{idx}. {class_name}: {acc:.4f}\n"

    class_accuracies_text += "\nBottom 10 Performing Classes:\n"
    for idx, (class_name, acc) in enumerate(bottom_10, 1):
        class_accuracies_text += f"{idx}. {class_name}: {acc:.4f}\n"

    # Save class accuracies
    with open(os.path.join(metrics_dir, 'class_accuracies.txt'), 'w') as f:
        f.write(class_accuracies_text)

    # Loss distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_losses, bins=50, alpha=0.75)
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Test Loss Distribution')
    plt.savefig(os.path.join(metrics_dir, 'loss_distribution.png'))
    plt.close()

    return accuracy, mean_loss, cm, class_report

# Plotting function
def plot_results():
    training_accuracies = np.load(os.path.join(metrics_dir, 'training_accuracies.npy'))
    training_losses = np.load(os.path.join(metrics_dir, 'training_losses.npy'))
    test_losses = np.load(os.path.join(metrics_dir, 'test_losses.npy'))
    test_accuracy = np.load(os.path.join(metrics_dir, 'test_accuracy.npy'))[0]

    epochs = range(1, len(training_accuracies) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_accuracies, label="Training Accuracy")
    plt.hlines(test_accuracy, xmin=1, xmax=len(epochs), colors='r', linestyles='dashed', label="Test Accuracy")
    plt.title("Training and Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, 'accuracy_plot.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.hlines(np.mean(test_losses), xmin=1, xmax=len(epochs), colors='r', linestyles='dashed', label="Mean Test Loss")
    plt.title("Training and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, 'loss_plot.png'))
    plt.close()

# Attention map visualization function
def visualize_attention():
    best_path = os.path.join(save_dir, "best.pt")
    if not os.path.exists(best_path):
        print("Best model not found. Please train the model first.")
        return

    checkpoint = torch.load(best_path, map_location=device)
    print(f"Loaded best model from {best_path} for attention visualization.")

    # Get model configuration from checkpoint
    model_name = checkpoint.get('model_name', args.model_name)
    image_sz = checkpoint.get('image_sz', args.image_sz)

    # Recreate the model with the correct configuration
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=100
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Register hook to capture attention weights
    global attention_weights
    attention_weights = []  # Reset attention weights

    def get_attn(module, input, output):
        # For ViT models in timm, the attention probabilities are stored in module.attn_drop
        # We need to access the attention probabilities from the output
        attn = output[1]  # The output contains (x, attn), where attn is the attention weights
        attention_weights.append(attn.detach().cpu())

    # Register hooks on the attention layers
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            module.register_forward_hook(get_attn)

    # Create attention map directory
    attention_dir = os.path.join(save_dir, 'attention_map')
    os.makedirs(attention_dir, exist_ok=True)

    # Get one sample from the test set
    input_image, _ = test_dataset[0]
    input_image = input_image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_image)

    # Now attention_weights list contains the attention weights from the transformer blocks
    # For simplicity, we'll visualize the attention from the last transformer block
    if len(attention_weights) == 0:
        print("Attention weights were not captured. Please ensure that the model's attention layers output the attention weights.")
        return

    # Assuming the last attention weight is the one from the last transformer block
    attn_weights = attention_weights[-1]  # Shape: [num_heads, seq_len, seq_len]

    # If the shape is [batch_size, num_heads, seq_len, seq_len], take the first sample
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # Shape: [num_heads, seq_len, seq_len]

    num_heads = attn_weights.shape[0]
    for i in range(num_heads):
        attn_map = attn_weights[i].numpy()  # Shape: [seq_len, seq_len]
        # You might need to process attn_map to match the image dimensions
        plt.figure(figsize=(5, 5))
        sns.heatmap(attn_map, cmap='viridis')
        plt.title(f'Head {i+1}')
        plt.axis('off')
        plt.savefig(os.path.join(attention_dir, f'head_{i+1}.png'))
        plt.close()

    print("Attention maps have been saved to:", attention_dir)

# Main execution based on --train parameter
if __name__ == "__main__":
    if args.train == 1:
        train_model()
    else:
        test_model()
        plot_results()
        visualize_attention()
