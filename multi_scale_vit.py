"""
Multi-Scale Vision Transformer for DAS Event Classification
Implements multi-scale patch sizes (8x8, 16x16, 32x32) with feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePatchEmbedding(nn.Module):
    """Multi-scale patch embedding with different patch sizes"""

    def __init__(self, img_size=256, in_channels=1, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim

        # Three different patch sizes for multi-scale processing
        self.patch_sizes = [8, 16, 32]
        self.embedders = nn.ModuleDict({
            'scale_8': nn.Conv2d(in_channels, embed_dim, kernel_size=8, stride=8),
            'scale_16': nn.Conv2d(in_channels, embed_dim, kernel_size=16, stride=16),
            'scale_32': nn.Conv2d(in_channels, embed_dim, kernel_size=32, stride=32)
        })

        # Calculate number of patches for each scale
        self.num_patches = {
            'scale_8': (img_size // 8) ** 2,  # 32x32 = 1024 patches
            'scale_16': (img_size // 16) ** 2,  # 16x16 = 256 patches
            'scale_32': (img_size // 32) ** 2  # 8x8 = 64 patches
        }

        print(f"Multi-Scale Configuration:")
        print(f"  - Scale 8x8: {self.num_patches['scale_8']} patches")
        print(f"  - Scale 16x16: {self.num_patches['scale_16']} patches")
        print(f"  - Scale 32x32: {self.num_patches['scale_32']} patches")

    def forward(self, x):
        """
        Extract patches at multiple scales
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Dictionary of patch embeddings for each scale
        """
        scale_embeddings = {}

        for scale_name, embedder in self.embedders.items():
            # Extract patches using convolution
            patches = embedder(x)  # (B, embed_dim, grid_h, grid_w)
            B, C, H, W = patches.shape

            # Flatten to sequence format
            patches_seq = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
            scale_embeddings[scale_name] = patches_seq

        return scale_embeddings


class ScaleSpecificTransformer(nn.Module):
    """Transformer encoder for a specific scale"""

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Transformer blocks for this specific scale
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Apply transformer blocks to scale-specific features"""
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class MultiScaleViT(nn.Module):
    """
    Multi-Scale Vision Transformer for DAS event classification
    Features:
    - Multi-scale patch extraction (8x8, 16x16, 32x32)
    - Scale-specific transformer encoders
    - Cross-scale feature fusion
    - Multi-scale classification head
    """

    def __init__(
            self,
            img_size=256,
            in_channels=1,
            num_classes=9,
            embed_dim=768,
            depth_per_scale=4,  # Depth for each scale-specific transformer
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Multi-scale patch embedding
        self.patch_embed = MultiScalePatchEmbedding(img_size, in_channels, embed_dim)

        # 2. Scale-specific transformers (different depths possible for each scale)
        self.scale_transformers = nn.ModuleDict({
            'scale_8': ScaleSpecificTransformer(embed_dim, depth_per_scale, num_heads, mlp_ratio, dropout),
            'scale_16': ScaleSpecificTransformer(embed_dim, depth_per_scale, num_heads, mlp_ratio, dropout),
            'scale_32': ScaleSpecificTransformer(embed_dim, depth_per_scale, num_heads, mlp_ratio, dropout)
        })

        # 3. Position embeddings for each scale (learnable)
        self.pos_embeddings = nn.ParameterDict({
            'scale_8': nn.Parameter(torch.zeros(1, self.patch_embed.num_patches['scale_8'], embed_dim)),
            'scale_16': nn.Parameter(torch.zeros(1, self.patch_embed.num_patches['scale_16'], embed_dim)),
            'scale_32': nn.Parameter(torch.zeros(1, self.patch_embed.num_patches['scale_32'], embed_dim))
        })

        # 4. Class tokens for each scale (learnable)
        self.cls_tokens = nn.ParameterDict({
            'scale_8': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'scale_16': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'scale_32': nn.Parameter(torch.zeros(1, 1, embed_dim))
        })

        # 5. Cross-scale attention for feature fusion
        self.cross_scale_attention = MultiHeadAttention(embed_dim * 3, num_heads, dropout)

        # 6. Multi-scale classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

        print(f"✅ Multi-Scale ViT initialized with {sum(p.numel() for p in self.parameters()):,} parameters")

    def _init_weights(self, module):
        """Initialize weights for transformer components"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            nn.init.trunc_normal_(module, std=0.02)

    def process_scale(self, scale_name, patches):
        """
        Process a single scale through its dedicated transformer
        Args:
            scale_name: Name of the scale ('scale_8', 'scale_16', 'scale_32')
            patches: Patch embeddings for this scale
        Returns:
            Scale-specific class token representation
        """
        B, N, C = patches.shape

        # Add class token for this scale
        cls_token = self.cls_tokens[scale_name].expand(B, -1, -1)
        patches = torch.cat([cls_token, patches], dim=1)

        # Add scale-specific positional embeddings
        pos_embed = self.pos_embeddings[scale_name]
        patches = patches + pos_embed

        # Apply scale-specific transformer
        features = self.scale_transformers[scale_name](patches)

        # Return class token output (global representation for this scale)
        scale_representation = features[:, 0]  # (B, embed_dim)

        return scale_representation

    def forward(self, x):
        """
        Forward pass for multi-scale ViT
        Args:
            x: Input tensor of shape (B, 1, 256, 256)
        Returns:
            Classification logits of shape (B, num_classes)
        """
        B = x.shape[0]

        # 1. Extract multi-scale patches
        scale_embeddings = self.patch_embed(x)

        # 2. Process each scale through its own transformer
        scale_features = {}
        for scale_name, patches in scale_embeddings.items():
            scale_features[scale_name] = self.process_scale(scale_name, patches)

        # 3. Fuse features from different scales
        # Concatenate scale representations along feature dimension
        fused_features = torch.cat([
            scale_features['scale_8'],  # Fine-grained details (high frequency)
            scale_features['scale_16'],  # Medium-level features (balanced)
            scale_features['scale_32']  # Coarse global features (low frequency)
        ], dim=1)  # (B, embed_dim * 3)

        # 4. Apply cross-scale attention for better fusion
        fused_features = fused_features.unsqueeze(1)  # (B, 1, embed_dim * 3)
        fused_features = self.cross_scale_attention(fused_features)
        fused_features = fused_features.squeeze(1)  # (B, embed_dim * 3)

        # 5. Final classification
        logits = self.head(fused_features)

        return logits

    def get_trainable_parameters(self):
        """Return count of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# Reuse transformer components (compatible with existing architecture)
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


if __name__ == "__main__":
    # Test the multi-scale ViT
    model = MultiScaleViT(
        img_size=256,
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