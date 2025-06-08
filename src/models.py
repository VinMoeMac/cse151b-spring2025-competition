import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    model_kwargs = {k: v for k, v in cfg.model.items() if k not in ["type", "kernel_size"]}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    
    if cfg.model.type == "simple_cnn":
        model = BetterCNN(**model_kwargs)
    elif cfg.model.type == "unet":
        model = ClimateUNet(**model_kwargs)
    elif cfg.model.type == "convnext":
        model = ClimateConvNeXt(**model_kwargs)
    elif cfg.model.type == "vision_transformer":
        model = ClimateViT(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# === Improved Building Blocks ===

class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial awareness while maintaining translation invariance"""
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        pe = torch.zeros(channels, height, width)
        
        # Create position indices
        y_pos = torch.arange(height).unsqueeze(1).float()
        x_pos = torch.arange(width).unsqueeze(0).float()
        
        # Normalize positions to [-1, 1]
        y_pos = (y_pos / (height - 1)) * 2 - 1
        x_pos = (x_pos / (width - 1)) * 2 - 1
        
        # Create sinusoidal encodings
        div_term = torch.exp(torch.arange(0, channels, 2).float() * 
                           -(math.log(10000.0) / channels))
        
        # Alternate between x and y encodings across channels
        for i in range(0, channels, 4):
            if i < channels:
                pe[i] = torch.sin(y_pos * div_term[i//4])
            if i + 1 < channels:
                pe[i + 1] = torch.cos(y_pos * div_term[i//4])
            if i + 2 < channels:
                pe[i + 2] = torch.sin(x_pos * div_term[i//4])
            if i + 3 < channels:
                pe[i + 3] = torch.cos(x_pos * div_term[i//4])
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SpatialAttentionBlock(nn.Module):
    """Spatial attention for climate patterns"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn = self.spatial_attn(x)
        return x * attn


class DilatedConvBlock(nn.Module):
    """Multi-scale dilated convolutions for different climate scales"""
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(dilations), 3, 
                     padding=d, dilation=d) for d in dilations
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(outputs, dim=1)
        return self.relu(self.bn(out))


# === U-Net Architecture for Climate Data ===

class ClimateUNet(nn.Module):
    """U-Net with climate-specific modifications"""
    def __init__(self, n_input_channels, n_output_channels, base_dim=64, depth=4, dropout_rate=0.1):
        super().__init__()
        self.depth = depth
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        dims = [base_dim * (2 ** i) for i in range(depth)]
        in_dim = n_input_channels
        
        for i, dim in enumerate(dims):
            self.encoders.append(nn.Sequential(
                DilatedConvBlock(in_dim, dim),
                SpatialAttentionBlock(dim),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            ))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            in_dim = dim
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth - 1):
            up_dim = dims[depth - 1 - i]
            down_dim = dims[depth - 2 - i]
            
            self.upsamples.append(nn.ConvTranspose2d(up_dim, down_dim, 2, stride=2))
            self.decoders.append(nn.Sequential(
                DilatedConvBlock(down_dim * 2, down_dim),
                SpatialAttentionBlock(down_dim),
                nn.Conv2d(down_dim, down_dim, 3, padding=1),
                nn.BatchNorm2d(down_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            ))
        
        # Final output
        self.final = nn.Conv2d(dims[0], n_output_channels, 1)
        
    def forward(self, x):
        # Encoder path
        encoder_outputs = []
        current = x
        
        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            encoder_outputs.append(current)
            if i < self.depth - 1:
                current = self.pools[i](current)
        
        # Decoder path
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            current = upsample(current)
            # Concatenate with skip connection
            skip = encoder_outputs[self.depth - 2 - i]
            current = torch.cat([current, skip], dim=1)
            current = decoder(current)
        
        return self.final(current)


# === ConvNeXt-Style Architecture ===

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for climate data"""
    def __init__(self, dim, drop_rate=0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # Depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise/1x1 conv
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        return input + x


class ClimateConvNeXt(nn.Module):
    """ConvNeXt architecture for climate emulation"""
    def __init__(self, n_input_channels, n_output_channels, base_dim=64, depths=[2, 2, 6, 2], dropout_rate=0.1):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, base_dim, 4, stride=2, padding=1),
            nn.LayerNorm(base_dim, eps=1e-6)
        )
        
        # Stages
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]
        self.stages = nn.ModuleList()
        
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential()
            
            # Downsampling layer (except for first stage)
            if i > 0:
                stage.add_module("downsample", nn.Sequential(
                    nn.LayerNorm(dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dim, 2, stride=2)
                ))
            
            # ConvNeXt blocks
            for j in range(depth):
                stage.add_module(f"block_{j}", ConvNeXtBlock(dim, dropout_rate))
            
            self.stages.append(stage)
        
        # Decoder for upsampling back to original resolution
        self.decoder = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[i-1], 2, stride=2),
                nn.LayerNorm(dims[i-1], eps=1e-6),
                nn.GELU()
            ))
        
        # Final upsampling and output
        self.final_upsample = nn.ConvTranspose2d(base_dim, base_dim, 4, stride=2, padding=1)
        self.final = nn.Conv2d(base_dim, n_output_channels, 1)
        
    def forward(self, x):
        x = self.stem(x)
        
        # Forward through stages
        features = [x]
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        # Decode back to original resolution
        for i, decoder in enumerate(self.decoder):
            x = decoder(x)
            # Optional: add skip connections
            # x = x + features[-(i+3)]
        
        x = self.final_upsample(x)
        return self.final(x)


# === Vision Transformer for Climate Data ===

class PatchEmbed(nn.Module):
    """Split climate data into patches"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class ClimateViT(nn.Module):
    """Vision Transformer adapted for climate data"""
    def __init__(self, n_input_channels, n_output_channels, img_size=64, patch_size=8, 
                 embed_dim=512, depth=6, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, n_input_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Decoder to reconstruct spatial output
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, patch_size * patch_size * n_output_channels)
        )
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert to patches
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Decode back to spatial format
        x = self.decoder(x)  # (B, n_patches, patch_size^2 * out_channels)
        
        # Reshape to spatial output
        n_patches_h = n_patches_w = H // self.patch_size
        x = x.view(B, n_patches_h, n_patches_w, 
                  self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, out_channels, n_patches_h, patch_size, n_patches_w, patch_size)
        x = x.contiguous().view(B, -1, H, W)
        
        return x


# === Original Models (Enhanced) ===

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class BetterCNN(nn.Module):
    """Enhanced version of the original CNN with better translational invariance"""
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.ModuleList()
        current_dim = init_dim
        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim, kernel_size))
            if i < depth - 1:
                current_dim *= 2

        # Add spatial attention
        self.spatial_attn = SpatialAttentionBlock(current_dim)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.spatial_attn(x)
        x = self.dropout(x)
        x = self.final(x)
        return x
