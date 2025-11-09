"""
DAS Vision Transformer Model
============================
Pure model definitions for DAS event classification
"""

import torch
import torch.nn as nn


class DASViT(nn.Module):
    """
    Vision Transformer for DAS event classification

    Requirements:
    - Handle variable-length spectrograms
    - Support 9 event classes
    - Include positional embeddings for spatial-temporal context
    """

    def __init__(
        self,
        img_size=(256, 256),  # Spectrogram dimensions
        patch_size=16,
        in_channels=1,
        num_classes=9,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Patch embedding layer
        # Calculate number of patches
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        print(f"ViT Configuration: img_size={img_size}, patches={self.num_patches}, grid={self.grid_size}")

        # Patch embedding using convolutional layer
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # 2. Position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # 3. Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for transformer"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Parameter):
            # Initialize positional embeddings and cls token
            nn.init.trunc_normal_(module, std=0.02)

    def forward(self, x):
        """
        Forward pass for DAS ViT
        Input: (B, 1, 256, 256) - full spectrogram images
        """
        B, C, H, W = x.shape

        # 1. Extract patches using convolutional embedding
        x = self.patch_embed(x)  # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # 2. Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 3. Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Use class token for classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Class token output
        logits = self.head(cls_output)

        return logits

    def get_trainable_parameters(self):
        """Return count of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        # Pre-norm architecture for better training stability
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # QKV projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation"""

    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DASCNN(nn.Module):
    """
    CNN baseline model for comparative analysis with ViT
    Modified for input_shape=(1, 256, 256)
    """

    def __init__(self, in_channels=1, num_classes=9):
        super().__init__()
        # Feature extraction backbone for 256x256 input
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Second conv block
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        """Return count of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


if __name__ == "__main__":
    # Test the model
    model = DASViT(
        img_size=(256, 256),
        patch_size=16,
        in_channels=1,
        num_classes=9
    )

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params, trainable_params = model.get_trainable_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")