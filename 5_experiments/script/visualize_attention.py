# Import thư viện cần thiết
import torch
import numpy as np
from model.vision_transformer import VisionTransformer
import matplotlib.pyplot as plt
import argparse
import os
from torch import nn
from torchvision import datasets, transforms
import random

# Hàm main
def main():
    # Parser cho tham số dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size for Vision Transformer")
    parser.add_argument('--embed_dim', type=int, default=768, help="Embedding dimension for Vision Transformer")
    parser.add_argument('--num_heads', type=int, default=12, help="Number of attention heads")
    parser.add_argument('--depth', type=int, default=12, help="Number of transformer blocks")
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help="MLP ratio in the Transformer blocks")
    parser.add_argument('--image_sz', type=int, default=224, help="Image size for resizing input")
    parser.add_argument('--name', type=str, default="exp", help="Unique name for this experiment")
    parser.add_argument('--match', type=bool, default=False)
    parser.add_argument('--device', type=str, default='0', help="GPU devices to use, e.g., '0', '0,1', or 'cpu'")

    global args
    args = parser.parse_args()

    if args.device == 'cpu':
        device = torch.device('cpu')
        device_ids = None
    else:
        device_ids = [int(d) for d in args.device.split(',')]
        device = torch.device(f'cuda:{device_ids[0]}')

    print(f"Using device(s): {args.device}")
    
    
    save_dir = os.path.join('attention', args.name)
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset và lấy ảnh đầu vào
    dataset = load_data_and_transform()
    class_names = dataset.classes  # Danh sách tên các lớp
    
    # Lấy ảnh và ground truth label ngẫu nhiên
    # idx = random.randint(0, len(dataset) - 1)
    # image, label = dataset[idx]  
    # print(f"Selected image index: {idx}, True label: {class_names[label]}")

    # Tạo mô hình và load checkpoint
    model = create_model_instance(args)
    
    # Tìm một ảnh có nhãn dự đoán giống hoặc khác nhãn thực tế, chỉnh sửa match=True/False để thay đổi
    image, label, pred_label = find_image(dataset, model, class_names, match=args.match)
    

    # Visualize attention từ layer và head cụ thể
    #visualize_attention(image, label, model, class_names, layer_idx=0, save_dir=save_dir, head_idx=None)

    # Visualize tất cả layers và heads
    visualize_all_attentions(image, label, model, class_names, save_dir)


# Các phương thức hỗ trợ
#----------------------------------------------
# Hàm load dữ liệu và transform
def load_data_and_transform():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    # Load CIFAR-100 dataset
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    return dataset
#----------------------------------------------
def find_image(dataset, model, class_names, match=True):
    """
    Tìm một ảnh có nhãn dự đoán giống hoặc khác nhãn thực tế.

    Args:
        dataset: Tập dữ liệu.
        model: Mô hình Vision Transformer.
        class_names: Danh sách tên các lớp.
        match: True nếu muốn nhãn dự đoán giống nhãn thực, False nếu muốn khác.

    Returns:
        image (torch.Tensor): Ảnh phù hợp.
        label (int): Nhãn thực tế của ảnh.
        pred_label (int): Nhãn dự đoán.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    for idx, (image, label) in enumerate(dataset):
        image = image.to(device).unsqueeze(0)  # Thêm batch dimension
        with torch.no_grad():
            logits, _ = model.forward_features(image, return_attention=True)
            cls_logits = model.head(logits[:, 0, :])  # Sử dụng CLS token
            pred_label = cls_logits.argmax(dim=-1).item()

        if (pred_label == label and match) or (pred_label != label and not match):
            print(f"Found image at index {idx}: True label: {class_names[label]}, Predicted: {class_names[pred_label]}")
            return image.squeeze(0), label, pred_label

    print("No matching image found.")
    return None, None, None



#----------------------------------------------
# Hàm tạo mô hình
def create_model_instance(args):
    checkpoint_path = './checkpoints/v2_pz8-img64-h8-d8-hd512-2/checkpoint_epoch_100.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(checkpoint.keys())
        print(checkpoint['embed_dim'], checkpoint['patch_size'], checkpoint['image_sz'], checkpoint['depth'])

        # Nếu checkpoint chứa các tham số như embed_dim, patch_size, v.v.
        if 'embed_dim' in checkpoint:
            model_instance = VisionTransformer(
                img_size=checkpoint.get('image_sz', args.image_sz),
                patch_size=checkpoint.get('patch_size', args.patch_size),
                embed_dim=checkpoint.get('embed_dim', args.embed_dim),
                depth=checkpoint.get('depth', args.depth),
                num_heads=checkpoint.get('num_heads', args.num_heads),
                mlp_ratio=checkpoint.get('mlp_ratio', args.mlp_ratio),
                num_classes=100,  # CIFAR-100 
                qkv_bias=True
            )
        else:
            # Nếu checkpoint chỉ chứa model_state_dict
            model_instance = VisionTransformer(
                img_size=args.image_sz,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                num_classes=100,
                qkv_bias=True
            )
        
        # Load state_dict nếu có
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Bỏ qua phần head nếu checkpoint không tương thích
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
            model_instance.load_state_dict(state_dict, strict=False)
        else:
            state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('head')}
            model_instance.load_state_dict(state_dict, strict=False)
    else:
        # Trường hợp không có checkpoint
        model_instance = VisionTransformer(
            img_size=args.image_sz,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            num_classes=100,
            qkv_bias=True
        )
    
    # Thêm một head mới cho CIFAR-100
    model_instance.head = nn.Linear(model_instance.head.in_features, 100)
    # Khởi tạo lại head
    nn.init.xavier_uniform_(model_instance.head.weight)
    nn.init.constant_(model_instance.head.bias, 0)
    
    print(f"Head output size: {model_instance.head.out_features}")  # In số lớp để kiểm tra
    # Head output size: 100
    return model_instance


#----------------------------------------------
def visualize_all_attentions(image, label, model, class_names, save_dir):
    """
    Trực quan hóa attention weights từ tất cả các layers và heads của mô hình Vision Transformer.

    Args:
        image (torch.Tensor): Ảnh đầu vào đã qua preprocessing (C, H, W).
        label (int): Ground truth label của ảnh.
        model (nn.Module): Mô hình Vision Transformer.
        class_names (List[str]): Danh sách tên các lớp.
        save_dir (str): Thư mục để lưu kết quả plot.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        # logits, attentions = model.forward_features(image.unsqueeze(0), return_attention=True)
        # pred_label = logits.argmax(dim=-1).item()  # Lấy predicted label


        # Forward pass qua mô hình để lấy predictions và attention weights
        logits, attentions = model.forward_features(image.unsqueeze(0), return_attention=True)
        print(f"Logits shape: {logits.shape}") 

        # (batch_size, num_patches + 1, embed_dim)
        # Logits shape: torch.Size([1, 65, 512])

        #Chỉ lấy vector CLS token cho dự đoán
        cls_logits = logits[:, 0, :]  # Lấy vector đầu tiên (CLS token)
        print(f"CLS logits shape: {cls_logits.shape}")  # Kích thước phải là (1, 100)
        
        # Sử dụng model.head để ánh xạ sang số lớp (100)
        cls_logits = model.head(cls_logits)              # Ánh xạ CLS token sang logits
        print(f"Logits after head: {cls_logits.shape}")  # Kích thước phải là (1, 100)
        pred_label = cls_logits.argmax(dim=-1).item()    # Lấy predicted label

        # Kiểm tra chỉ số có hợp lệ
        if pred_label >= len(class_names):
            raise ValueError(f"Predicted label {pred_label} vượt quá số lượng lớp ({len(class_names)})")

        # Hiển thị thông tin true và predicted label
        true_class = class_names[label]
        pred_class = class_names[pred_label]

        # #----------------------------------------------#
        # Tạo lưới plot
        num_layers = len(attentions)
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(18, 6))

        # Plot ảnh gốc
        axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[0].axis('off')
        axes[0].set_title(f"Original Image\nTrue: {true_class}\nPred: {pred_class}")

        # Plot trung bình attention của mỗi layer
        for i, attention_layer in enumerate(attentions):
            attention_layer = attention_layer.squeeze(0).cpu().numpy()  # (num_heads, seq_len, seq_len)
            avg_attention = attention_layer.mean(axis=0)  # Trung bình tất cả các heads
            cls_attention = avg_attention[0, 1:]  # Attention từ class token đến các patch
            cls_attention = cls_attention.reshape(model.patch_embed.grid_size)  # Reshape thành lưới patch

            # Plot attention map
            axes[i + 1].imshow(cls_attention, cmap='viridis')
            axes[i + 1].axis('off')
            axes[i + 1].set_title(f"Layer {i + 1}\nAvg Attention")

        # Lưu hình ảnh tổng hợp
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"attention_avg_{true_class}_visualization.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved to {save_path}")
        # #----------------------------------------------#

        # Lặp qua tất cả các layers và heads
        for layer_idx, attention_layer in enumerate(attentions):
            attention_layer = attention_layer.squeeze(0).cpu().numpy()  # (num_heads, seq_len, seq_len)
            num_heads = attention_layer.shape[0]

            for head_idx in range(num_heads):
                attention = attention_layer[head_idx]  # Attention của head cụ thể
                cls_attention = attention[0, 1:]  # Attention từ class token đến các patch
                cls_attention = cls_attention.reshape(model.patch_embed.grid_size)  # Reshape thành lưới patch

                # Vẽ attention map
                plt.figure(figsize=(8, 8))
                plt.imshow(cls_attention, cmap='viridis')
                plt.axis('off')
                plt.title(f'Layer {layer_idx}, Head {head_idx}')
                plt.xlabel(f'True: {true_class}\nPred: {pred_class}', fontsize=12, color='white')
                plt.colorbar()

                # Lưu plot vào thư mục `save_dir`
                os.makedirs(save_dir, exist_ok=True)
                filename = f"attention_{true_class}_layer_{layer_idx}_head_{head_idx}.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()  # Đóng plot sau khi lưu để tiết kiệm bộ nhớ

        print(f"All attention maps saved to {save_dir}")

def visualize_attention(image, label, model, class_names, layer_idx, save_dir, head_idx=None):
    """
    Trực quan hóa attention weights từ mô hình Vision Transformer với label true và predicted.

    Args:
        image (torch.Tensor): Ảnh đầu vào đã qua preprocessing (C, H, W).
        label (int): Ground truth label của ảnh.
        model (nn.Module): Mô hình Vision Transformer.
        class_names (List[str]): Danh sách tên các lớp.
        layer_idx (int): Chỉ số của layer để lấy attention.
        save_dir (str): Thư mục để lưu kết quả plot.
        head_idx (int, optional): Chỉ số của head. Nếu None, trung bình tất cả các heads.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        # Forward pass qua mô hình để lấy predictions và attention weights
        logits, attentions = model.forward_features(image.unsqueeze(0), return_attention=True)
        print(f"Logits shape: {logits.shape}") 
        # (batch_size, num_patches + 1, embed_dim)

        #Chỉ lấy vector CLS token cho dự đoán
        cls_logits = logits[:, 0, :]  # Lấy vector đầu tiên (CLS token)
        print(f"CLS logits shape: {cls_logits.shape}")  # Kích thước phải là (1, 100)
        

        # Sử dụng model.head để ánh xạ sang số lớp (100)
        cls_logits = model.head(cls_logits)  # Ánh xạ CLS token sang logits
        print(f"Logits after head: {cls_logits.shape}")  # Kích thước phải là (1, 100)
        pred_label = cls_logits.argmax(dim=-1).item()  # Lấy predicted label


        # Kiểm tra chỉ số có hợp lệ
        if pred_label >= len(class_names):
            raise ValueError(f"Predicted label {pred_label} vượt quá số lượng lớp ({len(class_names)})")

        # Hiển thị thông tin true và predicted label
        true_class = class_names[label]
        pred_class = class_names[pred_label]

        # Lấy attention từ layer được chọn
        attention = attentions[layer_idx].squeeze(0).cpu().numpy()  
        # (num_heads, seq_len, seq_len)

        # Nếu chỉ định head_idx, lấy attention của head đó
        if head_idx is not None:
            attention = attention[head_idx]
        else:
            # Trung bình tất cả các heads
            attention = attention.mean(axis=0)

        # Trích xuất attention của class token
        cls_attention = attention[0, 1:]  # Attention từ class token đến các patch
        cls_attention = cls_attention.reshape(model.patch_embed.grid_size)  # Reshape thành lưới patch

        # Vẽ attention map
        plt.figure(figsize=(8, 8))
        plt.imshow(cls_attention, cmap='viridis')
        plt.axis('off')
        plt.title(f'Layer {layer_idx}, Head {head_idx}' if head_idx else f'Layer {layer_idx}')

        # Hiển thị thông tin true và predicted label
        true_class = class_names[label]
        pred_class = class_names[pred_label]
        plt.xlabel(f'True: {true_class}\nPred: {pred_class}', fontsize=12, color='white')
        plt.colorbar()

        # Lưu plot vào thư mục `save_dir`
        os.makedirs(save_dir, exist_ok=True)
        filename = f"attention_layer_{layer_idx}_head_{head_idx if head_idx is not None else 'all'}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Đóng plot sau khi lưu để tiết kiệm bộ nhớ

    print(f"Attention map saved to {save_path}")


if __name__ == "__main__":
    main()