### Import thư viện
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import os
import argparse
from torchvision.transforms import functional as F
import collections
import traceback
import math

#### CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


### Thân hàm main

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer on CIFAR100')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--img-size', type=int, default=32,
                      help='Input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of data loading workers (default: 4)')
    parser.add_argument('--name', type=str, default='default',
                      help='Experiment name (default: default)')
    parser.add_argument('--gpu_id', type=int, default=0,
                      help='ID of GPU to use (default: None, use CPU)')
    parser.add_argument('--visualize', action='store_true', help='Enable attention visualization')

    return parser.parse_args()

def main():
    """Main evaluation function"""
    args = parse_args()
    print(vars(args))

    # Create save directory
    save_dir = os.path.join('./results', f'eval_{args.name}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nResults will be saved to: {save_dir}")
    
    # Setup device
    device = setup_device(args.gpu_id)
    
    # Initialize data module
    print("\nInitializing data module...")
    data_module = DataModule(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    _, test_loader = data_module.get_loaders()
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    print(f"Loading model from: {args.model}")
    evaluator = Evaluator(args.model, save_dir, device)
    
    # Ensure visualization mode is off for evaluation
    evaluator.model.set_visualization_mode(False)

    # Perform evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate(test_loader)
    
    # Handle visualization if requested
    if hasattr(args, 'visualize') and args.visualize:
        print("\nGenerating attention visualizations...")
        visualizer = AttentionVisualizer(save_dir)
        
        # Get a batch of images for attention visualization
        images, labels = next(iter(test_loader))
        print(f"Sample batch shape: {images.shape}")
        
        # Enable visualization mode temporarily
        evaluator.model.set_visualization_mode(True)
        
        print("Generating attention maps and overlays...")
        visualizer.visualize_attention(evaluator.model, images, labels, device)
        print(f"Attention visualizations saved to:")
        print(f"- Attention Maps: {os.path.join(save_dir, 'attention_maps')}")
        print(f"- Attention Overlays: {os.path.join(save_dir, 'attention_overlays')}")
        
        # Disable visualization mode after visualization
        evaluator.model.set_visualization_mode(False)
    else:
        print("\nSkipping attention visualization (use --visualize flag to enable)")
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    print(f"Model: {os.path.basename(args.model)}")
    print(f"Total Accuracy: {results['total_accuracy']:.4f}")
    print(f"Mean Loss: {results['mean_loss']:.4f}")
    
    # Get top 5 and bottom 5 classes
    sorted_accs = dict(sorted(results['class_accuracies'].items(), 
                            key=lambda x: x[1], reverse=True))
    
    print("\nTop 5 Classes:")
    print("-" * 30)
    for i, (class_name, acc) in enumerate(list(sorted_accs.items())[:5], 1):
        print(f"{i}. {class_name:<15}: {acc:.4f}")
    
    print("\nBottom 5 Classes:")
    print("-" * 30)
    for i, (class_name, acc) in enumerate(list(sorted_accs.items())[-5:], 1):
        print(f"{i}. {class_name:<15}: {acc:.4f}")
    
    print(f"\nDetailed results saved to: {save_dir}")

    # print("\nEvaluation Results Summary:")
    # print("=" * 50)
    # print(f"Detailed evaluation results: {os.path.join(save_dir, 'metrics', 'evaluation_results.txt')}")
    # print(f"Loss distribution plot: {os.path.join(save_dir, 'metrics', 'loss_distribution.png')}")
    # print(f"Confusion matrix: {os.path.join(save_dir, 'metrics', 'confusion_matrix.png')}")
    # print(f"Class accuracies plot: {os.path.join(save_dir, 'metrics', 'class_accuracies.png')}")


### Class và các functions

class DataModule:
    """Data module for CIFAR-100 dataset"""
    def __init__(self, img_size=224, batch_size=64, num_workers=4):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                              (0.2673, 0.2564, 0.2762))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                              (0.2673, 0.2564, 0.2762))
        ])
    
    def get_loaders(self):
        """Get train and test data loaders"""
        train_set = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, 
            transform=self.transform_train
        )
        test_set = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, 
            transform=self.transform_test
        )
        
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader

class AttentionVisualizer:
    """
    Class for handling attention visualization of Vision Transformer models.
    Provides tools to visualize attention maps and their overlays on original images.
    """
    def __init__(self, save_dir):
        """
        Initialize the visualizer
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, 'attention_maps'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'attention_overlays'), exist_ok=True)
    
    @staticmethod
    def get_attention_map(attention_weights):
        """
        Convert attention weights to attention map
        Args:
            attention_weights: shape [batch_size, num_heads, num_patches+1, num_patches+1]
        Returns:
            attention maps shape [batch_size, num_heads, grid_size, grid_size]
        """
        print(f"\nOriginal attention weights shape: {attention_weights.shape}")

        # Remove CLS token
        attn = attention_weights[:, :, 1:, 1:]  # Remove CLS token
        print(f"After removing CLS token: {attn.shape}")  # Should be [1, 4, 64, 64]

        # First, reshape to separate patches
        batch_size, num_heads, seq_len, _ = attn.shape
        height = width = int(np.sqrt(seq_len))  # Should be 8
        print(f"Calculated height/width: {height}")

        try:
            # Reshape the last two dimensions into a grid
            attn = attn.reshape(batch_size * num_heads, seq_len, seq_len)
            attn = attn.reshape(batch_size * num_heads, height, width, height, width)
            # Take mean over the patch dimensions
            attn = attn.mean(dim=(3, 4))
            # Reshape back to include batch and heads
            attn = attn.reshape(batch_size, num_heads, height, width)
            print(f"Final attention map shape: {attn.shape}")  # Should be [1, 4, 8, 8]
            return attn
        except Exception as e:
            print(f"Error in reshape: {str(e)}")
            print(f"Total elements in tensor: {attn.numel()}")
            raise
    
    def visualize_attention(self, model, images, labels, device, num_images=4):
        """
        Generate and save attention visualizations
        
        Args:
            model (nn.Module): Vision Transformer model
            images (torch.Tensor): Batch of images
            labels (torch.Tensor): Ground truth labels
            device (torch.device): Device to run model on
            num_images (int): Number of images to visualize
        """
        model.eval()
        images = images.to(device)
        labels = labels.to(device)
        
        # Debug: Print model structure
        print("\nModel structure:")
        for name, module in model.named_modules():
            print(f"Module name: {name}")
        
        # Forward pass, lấy predictions và attention weights trực tiếp
        with torch.no_grad():
            outputs, attention_weights = model(images)
            _, predicted = torch.max(outputs, 1)
        
        if not attention_weights:
            print("No attention weights were collected.")
            return
        
        print(f"\nCollected attention weights from {len(attention_weights)} layers")
        print("Attention weight shapes:")
        for i, attn in enumerate(attention_weights):
            print(f"Layer {i + 1}: {attn.shape}")
        
        try:
            self._plot_attention_details(images, labels, predicted, attention_weights, num_images)
            self._plot_attention_overlays(images, labels, predicted, attention_weights, num_images)
            print("\nVisualization completed successfully!")
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _plot_attention_details(self, images, labels, predicted, attention_maps, num_images):
        """Plot detailed attention maps"""
        num_layers = len(attention_maps)
        num_heads = attention_maps[0].size(1)
     
        for img_idx in range(min(num_images, len(images))):
            try:
                fig = plt.figure(figsize=(20, 12))
                gs = fig.add_gridspec(3, num_layers + 1, hspace=0.3, wspace=0.3)
                
                # Plot original image
                self._plot_original_image(fig, gs, images[img_idx], labels[img_idx], predicted[img_idx])
                
                # Plot attention maps for each layer
                for layer_idx in range(num_layers):
                    print(f"\nProcessing layer {layer_idx + 1}")
                    single_attention = attention_maps[layer_idx][img_idx:img_idx+1]
                    print(f"Single image attention shape: {single_attention.shape}")
                    
                    att_map = self.get_attention_map(single_attention)  # [1, num_heads, 8, 8]
                    print(f"Attention map shape after processing: {att_map.shape}")
                    
                    # Average attention map
                    ax = fig.add_subplot(gs[0, layer_idx + 1])
                    avg_att_map = att_map[0].mean(dim=0).cpu()  # Average over heads [8, 8]
                    print(f"Average attention map shape: {avg_att_map.shape}")
                    im = ax.imshow(avg_att_map, cmap='viridis')
                    ax.set_title(f'Layer {layer_idx + 1}\nAvg Attention')
                    ax.axis('off')
                    
                    # Individual head maps
                    for head_idx in range(min(4, num_heads)):
                        ax = fig.add_subplot(gs[1 + head_idx//2, layer_idx + 1])
                        head_att_map = att_map[0, head_idx].cpu()  # [8, 8]
                        im = ax.imshow(head_att_map, cmap='viridis')
                        ax.set_title(f'Head {head_idx + 1}')
                        ax.axis('off')
                
                plt.suptitle(f'Attention Maps Analysis - Image {img_idx + 1}', fontsize=16)
                save_path = os.path.join(self.save_dir, 'attention_maps', f'attention_maps_img{img_idx + 1}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"Saved attention maps to: {save_path}")
                plt.close()
                
            except Exception as e:
                print(f"Error plotting attention details for image {img_idx}: {str(e)}")
                traceback.print_exc()
                continue

    def _plot_attention_overlays(self, images, labels, predicted, attention_maps, num_images):
        """Plot attention overlays"""
        num_layers = len(attention_maps)

        for img_idx in range(min(num_images, len(images))):
            try:
                fig = plt.figure(figsize=(20, 4))

                for layer_idx in range(num_layers):
                    plt.subplot(1, num_layers, layer_idx + 1)

                    print(f"\nProcessing overlay for layer {layer_idx + 1}")
                    single_attention = attention_maps[layer_idx][img_idx:img_idx+1]
                    att_map = self.get_attention_map(single_attention)  # [1, num_heads, 8, 8]
                    avg_att_map = att_map[0].mean(dim=0).cpu()  # [8, 8]

                    print(f"Average map shape before resize: {avg_att_map.shape}")
                    # Resize to match image size
                    att_map_resized = F.resize(
                        avg_att_map.unsqueeze(0).unsqueeze(0).float(),
                        size=(images.size(2), images.size(3))
                    ).squeeze().numpy()
                    print(f"Resized map shape: {att_map_resized.shape}")

                    self._plot_overlay(images[img_idx], att_map_resized, layer_idx)

                true_class = CIFAR100_CLASSES[labels[img_idx].cpu().item()]
                pred_class = CIFAR100_CLASSES[predicted[img_idx].cpu().item()]
                plt.suptitle(
                    f'Attention Overlay - Image {img_idx + 1}\nTrue: {true_class}, Pred: {pred_class}',
                    fontsize=16
                )

                save_path = os.path.join(self.save_dir, 'attention_overlays', f'attention_overlay_img{img_idx + 1}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"Saved overlay to: {save_path}")
                plt.close()

            except Exception as e:
                print(f"Error plotting attention overlays for image {img_idx}: {str(e)}")
                traceback.print_exc()
                continue
    
    
    def _plot_original_image(self, fig, gs, image, label, predicted):
        """
        Plot original image with true and predicted labels
        
        Args:
            fig (matplotlib.figure.Figure): Figure object
            gs (matplotlib.gridspec.GridSpec): GridSpec object
            image (torch.Tensor): Image tensor
            label (torch.Tensor): True label
            predicted (torch.Tensor): Predicted label
        """
        ax = fig.add_subplot(gs[:, 0])
        
        # Denormalize image
        img = image.cpu()
        img = img * torch.tensor([0.2673, 0.2564, 0.2762]).view(3, 1, 1) + \
              torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        true_class = CIFAR100_CLASSES[label.cpu().item()]
        pred_class = CIFAR100_CLASSES[predicted.cpu().item()]
        ax.set_title(f'Original Image\nTrue: {true_class}\nPred: {pred_class}')
        ax.axis('off')
    
    def _plot_overlay(self, image, attention_map, layer_idx):
        """
        Plot attention map overlay on original image
        
        Args:
            image (torch.Tensor): Image tensor
            attention_map (numpy.ndarray): Attention map
            layer_idx (int): Index of the current layer
        """
        # Denormalize image
        img = image.cpu()
        img = img * torch.tensor([0.2673, 0.2564, 0.2762]).view(3, 1, 1) + \
              torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.imshow(attention_map, cmap='hot', alpha=0.5)
        plt.title(f'Layer {layer_idx + 1} Attention')
        plt.axis('off')

class Evaluator:
    """Class for model evaluation"""
    def __init__(self, model_path, save_dir, device):
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = device
        os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
        
        self.model = self._load_model()
        self.model.set_visualization_mode(False)  # Tắt visualization mode khi đánh giá
        self.criterion = nn.CrossEntropyLoss()
        
    def _load_model(self):
        """Load and prepare model for evaluation"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Kiểm tra xem checkpoint có phải là OrderedDict không
            if isinstance(checkpoint, collections.OrderedDict):
                # Vì mô hình là state_dict nên cần tạo model trước
                from demo_mod2 import VisionTransformer  # model class của ta

                # truyền tham số của /storageStudents/ncsmmlab/tungufm/VitFromScratch/final/demo_v2.sh
                model = VisionTransformer(
                    img_size=64, patch_size=8, in_chans=3, 
                    num_classes=100, embed_dim=128, depth=6, 
                    num_heads=4,  mlp_ratio=2 , dropout=0.1, drop_emb=False).to(self.device) 
                
                # Lọc bỏ các key không mong muốn từ state dict
                filtered_state_dict = {
                    k: v for k, v in checkpoint.items()
                    if not any(unwanted in k for unwanted in ['total_ops', 'total_params'])
                }
                
                model.load_state_dict(filtered_state_dict)
                print("Successfully loaded filtered state dict")
            else:
                model = checkpoint
            
            model = model.to(self.device)
            model.eval()
            return model
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def evaluate(self, test_loader):
        """Perform comprehensive model evaluation"""
        print(f"\nEvaluating model on {self.device}")
        
        metrics = {
            'test_losses': [],
            'predictions': [],
            'targets': [],
            'class_correct': [0] * 100,
            'class_total': [0] * 100
        }
        
        # Đảm bảo visualization mode tắt trong quá trình evaluate
        self.model.set_visualization_mode(False)

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                metrics['test_losses'].append(loss.item())
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                metrics['predictions'].extend(predicted.cpu().numpy())
                metrics['targets'].extend(target.cpu().numpy())
                
                # Update class-wise accuracy
                correct = (predicted == target)
                for i in range(len(target)):
                    label = target[i].item()
                    metrics['class_correct'][label] += correct[i].item()
                    metrics['class_total'][label] += 1
        
        return self._compute_metrics(metrics)
    
    def _compute_metrics(self, metrics):
        """Compute comprehensive evaluation metrics"""
        results = {}
        
        # Basic metrics
        results['mean_loss'] = np.mean(metrics['test_losses'])
        results['losses'] = metrics['test_losses']
        
        # Accuracy metrics
        class_accuracies = {
            CIFAR100_CLASSES[i]: metrics['class_correct'][i]/metrics['class_total'][i]
            for i in range(100) if metrics['class_total'][i] > 0
        }
        results['class_accuracies'] = class_accuracies
        results['total_accuracy'] = sum(metrics['class_correct']) / sum(metrics['class_total'])
        
        # Confusion matrix and classification report
        results['confusion_matrix'] = confusion_matrix(
            metrics['targets'], metrics['predictions'])
        results['classification_report'] = classification_report(
            metrics['targets'], metrics['predictions'],
            target_names=CIFAR100_CLASSES, output_dict=True)
        
        self._save_results(results)
        return results
    
    def _save_results(self, results):
        """Save evaluation results"""
        metrics_path = os.path.join(self.save_dir, 'metrics')
        
        # Save detailed metrics to text file
        with open(os.path.join(metrics_path, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Model Evaluation Results\n")
            f.write(f"=======================\n\n")
            f.write(f"Model path: {self.model_path}\n\n")
            
            f.write(f"Overall Metrics:\n")
            f.write(f"--------------\n")
            f.write(f"Total Accuracy: {results['total_accuracy']:.4f}\n")
            f.write(f"Mean Loss: {results['mean_loss']:.4f}\n\n")
            
            # Save top/bottom class performances
            self._write_class_performances(f, results['class_accuracies'])
        
        # Save visualizations
        self._plot_metrics(results, metrics_path)
    
    def _write_class_performances(self, f, class_accuracies):
        """Write class-wise performance details"""
        sorted_classes = dict(sorted(
            class_accuracies.items(), key=lambda x: x[1], reverse=True))
        
        f.write("Top 10 Performing Classes:\n")
        for i, (class_name, acc) in enumerate(list(sorted_classes.items())[:10]):
            f.write(f"{i+1}. {class_name}: {acc:.4f}\n")
        
        f.write("\nBottom 10 Performing Classes:\n")
        for i, (class_name, acc) in enumerate(list(sorted_classes.items())[-10:]):
            f.write(f"{i+1}. {class_name}: {acc:.4f}\n")
    
    def _plot_metrics(self, results, save_path):
        """Generate and save metric visualizations"""
        # Loss distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['losses'], bins=50, alpha=0.7, color='blue')
        plt.axvline(results['mean_loss'], color='red', linestyle='dashed',
                   linewidth=2, label=f'Mean Loss: {results["mean_loss"]:.4f}')
        plt.title('Loss Distribution')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'loss_distribution.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Confusion matrix (top 10 classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(results['confusion_matrix'][:10, :10], annot=True,
                   fmt='d', cmap='Blues',
                   xticklabels=CIFAR100_CLASSES[:10],
                   yticklabels=CIFAR100_CLASSES[:10])
        plt.title('Confusion Matrix (Top 10 Classes)')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Class accuracies
        plt.figure(figsize=(15, 6))
        sorted_accs = dict(sorted(results['class_accuracies'].items(), 
                                key=lambda x: x[1], reverse=True))
        plt.bar(range(len(sorted_accs)), list(sorted_accs.values()))
        plt.title('Class-wise Accuracies')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(sorted_accs)), list(sorted_accs.keys()), 
                  rotation=90)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'class_accuracies.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

def setup_device(gpu_id=None):
    """
    Setup the device for computation
    Args:
        gpu_id (int): ID of GPU to use. If None, use CPU
    Returns:
        device (torch.device): Device to use for computation
    """
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {gpu_id} not found. Available GPUs: {torch.cuda.device_count()}")
            print("Using CPU instead.")
            return torch.device('cpu')
        
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

if __name__ == "__main__":
    main()