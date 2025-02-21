import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .CST_encoder import CST_attention
from .layers import LocalPerceptionUint, InvertedResidualFeedForward, Temp_attention, Spec_attention

class CMT_Layers(nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.temp_embed_dim = temp_embed_dim
        
        # Local Processing Unit
        self.local_processing = LocalPerceptionUint(dim=params['nb_cnn2d_filt'])
        
        # Layer Norm - 차원을 동적으로 처리
        self.norm1 = nn.LayerNorm(params['nb_cnn2d_filt'])
        self.norm2 = nn.LayerNorm(params['nb_cnn2d_filt'])
        
        # Dropout
        self.dropout = nn.Dropout(params.get('dropout_rate', 0.1))
        
        # Feed Forward
        self.feed_forward = InvertedResidualFeedForward(
            dim=params['nb_cnn2d_filt'],
            dim_ratio=params.get('cmt_expansion_factor', 4.)
        )

    def forward(self, x):
        B, C, N, T = x.shape  # (batch, channels, sequence, time)
        
        # Reshape for LayerNorm
        x_trans = x.permute(0, 2, 3, 1)  # (B, N, T, C)
        
        # Local Processing
        x_local = self.local_processing(x)
        x = x + self.dropout(x_local)
        
        # Apply LayerNorm
        x = x_trans.reshape(-1, C)  # (B*N*T, C)
        x = self.norm1(x)
        x = x.view(B, N, T, C).permute(0, 3, 1, 2)  # Back to (B, C, N, T)
        
        # Feed Forward
        x_ff = self.feed_forward(x)
        x = x + self.dropout(x_ff)
        
        # Apply LayerNorm
        x = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*N*T, C)
        x = self.norm2(x)
        x = x.view(B, N, T, C).permute(0, 3, 1, 2)  # Back to (B, C, N, T)
        
        return x

class CMT_block(nn.Module):
    def __init__(self, params, temp_embed_dim):
        super().__init__()
        self.temp_embed_dim = temp_embed_dim
        self.num_layers = params.get('nb_self_attn_layers', 4)
        
        # CMT Layers
        self.block_list = nn.ModuleList([
            CMT_Layers(temp_embed_dim, params) 
            for _ in range(self.num_layers)
        ])
        
        print("CST attention with CMT block initialized.")

    def forward(self, x):
        # print("\nCMT_block forward pass:")
        # print(f"Input shape: {x.shape}")
        
        # Process through CMT layers
        for i, block in enumerate(self.block_list):
            x = block(x)
            # print(f"After block {i} shape: {x.shape}")
        
        return x