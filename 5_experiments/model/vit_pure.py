# vit_org.py
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class PatchEmbed(nn.Module):
    """Chia ảnh thành các patch và chuyển đổi chúng thành vector embeddings."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, N, embed_dim) where N is the number of patches.
        """
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x

class Attention(nn.Module):
    """Multi-head Self-Attention module."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads."

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, dim)
        Returns:
            Tensor of shape (B, N, dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each has shape (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """Feed-Forward Network."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, dropout: float = 0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, dim)
        Returns:
            Tensor of shape (B, N, dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., dropout: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, dim)
        Returns:
            Tensor of shape (B, N, dim)
        """
        x = x + self.attn(self.norm1(x))  # Residual connection
        x = x + self.mlp(self.norm2(x))   # Residual connection
        return x

class VisionTransformerPure(nn.Module):
    """Vision Transformer as per the original paper."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        dropout: float = 0.,
        representation_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder Layers
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if isinstance(self.head, nn.Linear) and self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        # Initialize transformer blocks
        for blk in self.blocks:
            nn.init.xavier_uniform_(blk.attn.qkv.weight)
            nn.init.xavier_uniform_(blk.attn.proj.weight)
            nn.init.xavier_uniform_(blk.mlp.fc1.weight)
            nn.init.xavier_uniform_(blk.mlp.fc2.weight)
            if blk.attn.qkv.bias is not None:
                nn.init.zeros_(blk.attn.qkv.bias)
            if blk.attn.proj.bias is not None:
                nn.init.zeros_(blk.attn.proj.bias)
            if blk.mlp.fc1.bias is not None:
                nn.init.zeros_(blk.mlp.fc1.bias)
            if blk.mlp.fc2.bias is not None:
                nn.init.zeros_(blk.mlp.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, num_classes)
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Concatenate CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Normalize
        x = self.norm(x)

        # Classification head using CLS token
        cls_output = x[:, 0]  # (B, embed_dim)
        x = self.head(cls_output)  # (B, num_classes)

        return x

if __name__ == "__main__":
    # Kiểm tra mô hình với một input ngẫu nhiên
    model = VisionTransformerPure(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        dropout=0.1,
    )
    x = torch.randn(2, 3, 224, 224)  # Batch size 2
    y = model(x)
    print(y.shape)  # Should output: torch.Size([2, 1000])
