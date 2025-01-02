import torch
import torchvision
from torch.utils.data import DataLoader
import argparse
import math
import os
import time
from datetime import timedelta
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from thop import profile  # Library for calculating FLOPs and parameters
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for environments without a display
import matplotlib.pyplot as plt
import json

def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Vision Transformer for CIFAR100')
    parser.add_argument('--name', type=str, default='experiment', help='Name of the experiment')
    parser.add_argument('--depth', type=int, default=12, help='Depth of the network (number of transformer layers)')
    parser.add_argument('--drop_emb', action='store_true', help='Whether to drop the embedding')
    parser.add_argument('--heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension')
    parser.add_argument('--mlp_size', type=int, default=3072, help='MLP size')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')  # Added img_size parameter
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_sz', type=int, default=64, help='Batch size')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--save_checkpoint', type=str, default='checkpoint.pth', help='Path to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for dataset')
    parser.add_argument('--log_interval', type=int, default=10, help='Batches between logging training status')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Epochs between saving checkpoints')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of GPU to use (0-based index)')
    parser.add_argument('--max_memory_fraction', type=float, default=0.8, help='Maximum GPU memory fraction to allocate (0.0 - 1.0)')
    return parser.parse_args()

def get_data_loaders(img_size=224, batch_size=64, num_workers=2):
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762))
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, criterion, epoch, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100,
                        bar_format='{desc}: {percentage:3.0f}%|{bar:12}| [{elapsed}<{remaining}] {postfix}')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update scheduler
        scheduler.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Monitor memory
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)

        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{running_loss / total:.4f}',
            'Acc': f'{100. * correct / total:.4f}%',
            'GN': f'{total_norm:.2f}',
            'LR': f'{current_lr:.6f}',
            'Mem': f'{memory_allocated:.1f}MB/{memory_reserved:.1f}MB'

        })

    epoch_loss = running_loss / total
    epoch_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} \tLoss: {epoch_loss:.4f} \tAccuracy: {epoch_accuracy:.4f}%')
    return epoch_loss, epoch_accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False,
                                 bar_format='{desc}: {percentage:3.0f}%|{bar:9}| [{elapsed}<{remaining}]'):
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_loss /= total
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}%\n')
    return test_loss, accuracy

def save_checkpoint(model, optimizer, epoch, name, path='./checkpoints'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(path, f'{name}_checkpoint{epoch}.pt'))
    print(f'Checkpoint saved at epoch {epoch}')

def visualize_results(train_losses, test_losses, train_accuracies, test_accuracies, name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.savefig(f'./results/metrics_{name}.png')
    plt.close()

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, drop_emb=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.drop_emb = drop_emb

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=int(embed_dim * mlp_ratio), dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()
        self.visualization_mode = False  # Add this flag

    def set_visualization_mode(self, mode=False):
        """Toggle visualization mode"""
        self.visualization_mode = mode
        return self

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        if self.drop_emb:
            x = self.pos_drop(x)

        attention_weights = []

        for layer in self.layers:
            if self.visualization_mode:
                x, attn = layer(x)
                attention_weights.append(attn)
            else:
                x, _ = layer(x)  # Discard attention weights

        x = self.norm(x)
        cls_output = x[:, 0]
        output = self.head(cls_output)

        if self.visualization_mode:
            return output, attention_weights
        return output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=3072, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        # Add variable to store attention weights
        self.attention_weights = None

    def forward(self, src):
        B = src.shape[0]  # Batch size
        N = src.shape[1]  # Sequence length

        # Self-Attention with Gumbel-Softmax
        Q = self.q_linear(src)
        K = self.k_linear(src)
        V = self.v_linear(src)

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(B, N, self.nhead, -1).transpose(1, 2)  # [B, num_heads, N, d_k]
        K = K.view(B, N, self.nhead, -1).transpose(1, 2)
        V = V.view(B, N, self.nhead, -1).transpose(1, 2)

        # Compute attention scores
        d_k = self.d_model // self.nhead
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        gumbel_scores = F.gumbel_softmax(scores, tau=1, hard=False)

        # Store attention weights for visualization
        self.attention_weights = gumbel_scores

        # Compute attention output
        attn_output = torch.matmul(gumbel_scores, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.d_model)

        src = src + self.dropout_attn(attn_output)
        src = self.norm1(src)

        # Feedforward with GELU activation
        ff_output = self.linear2(self.dropout1(F.gelu(self.linear1(src))))

        src = src + self.dropout_ffn(ff_output)
        src = self.norm2(src)

        # Return output and attention weights
        return src, self.attention_weights

def main():
    ###### Training ##########
    # Create necessary directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    args = parse_args()
    print(vars(args))

    # Set device based on GPU availability and gpu_id
    if torch.cuda.is_available():
        assert 0 <= args.gpu_id < torch.cuda.device_count(), \
            f"GPU {args.gpu_id} does not exist. Server has {torch.cuda.device_count()} GPUs (0-{torch.cuda.device_count()-1})"
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

        # Set GPU memory limit
        torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, device=device)
        print(f"GPU memory limit: {args.max_memory_fraction * 100}%")

    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU")

    train_loader, test_loader = get_data_loaders(img_size=args.img_size, batch_size=args.batch_sz)

    model = VisionTransformer(img_size=args.img_size, patch_size=args.patch_size, in_chans=3, num_classes=100,
                              embed_dim=args.hidden_dim, depth=args.depth, num_heads=args.heads,
                              mlp_ratio=args.mlp_size / args.hidden_dim, dropout=args.dropout_rate,
                              drop_emb=args.drop_emb).to(device)

    # Calculate and display number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Use thop to calculate FLOPs and parameters
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"Total MACs (FLOPs): {macs}")
    print(f"Total Params (from thop): {params}")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    # Initialize OneCycleLR scheduler
    max_lr = args.learning_rate
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    criterion = nn.CrossEntropyLoss()

    # Start training - start timing
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch, scheduler)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Save metrics to a dictionary
        metrics = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        }

        # Save metrics to a JSON file after each epoch
        with open(f'./results/metrics_{args.name}.json', 'w') as f:
            json.dump(metrics, f)

        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, args.name)
        if epoch % 10 == 0:
            visualize_results(train_losses, test_losses, train_accuracies, test_accuracies, args.name)

    # Call visualize_results after training completes
    visualize_results(train_losses, test_losses, train_accuracies, test_accuracies, args.name)
    torch.save(model.state_dict(), f"{args.name}_final.pth")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {str(timedelta(seconds=int(elapsed_time)))}")
    print("..Training finished")

    # Calculate average test loss and accuracy
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")

if __name__ == '__main__':
    main()
