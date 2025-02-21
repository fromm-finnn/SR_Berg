import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .CST_details.encoder import Encoder
from .CST_details.CST_encoder import CST_encoder
from .CST_details.CMT_Block import CMT_block

class CST_former(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ch_attn_dca = params['ChAtten_DCA']
        self.ch_attn_unfold = params['ChAtten_ULE']
        self.cmt_block = params['CMT_block']
        
        # NGCC-PHAT의 출력 shape에 맞춤
        self.num_pairs = (params['nb_channels'] * (params['nb_channels'] - 1)) // 2  # 120 마이크 쌍
        self.feature_dim = params['ngcc_channels']  # 32 채널
        
        print(f"Expected NGCC-PHAT output shape: (B, {self.num_pairs}, {self.feature_dim})")
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim, params['nb_cnn2d_filt']),
            nn.LayerNorm(params['nb_cnn2d_filt']),
            nn.GELU()
        )
        
        # Feature dimension 계산
        self.conv_block_freq_dim = params['nb_cnn2d_filt']
        self.temp_embed_dim = self.conv_block_freq_dim
        
        # Attention Layer
        if not self.cmt_block:
            self.attention_stage = CST_encoder(self.temp_embed_dim, params)
        else:
            self.attention_stage = CMT_block(params, self.temp_embed_dim)
            
        # Temporal Pooling
        if self.t_pooling_loc == 'end':
            self.t_pooling = nn.AdaptiveAvgPool1d(1)
            
        # Position head
        self.position_head = nn.Sequential(
            nn.Linear(params['nb_cnn2d_filt'], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # [x, y, z]
        )
        
        # Rotation head (quaternion)
        self.rotation_head = nn.Sequential(
            nn.Linear(params['nb_cnn2d_filt'], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # [qw, qx, qy, qz]
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        입력: x (B, num_pairs, feature_dim)
        출력: dov (B, 7) - [x, y, z, qw, qx, qy, qz]
        """
        B, N, feat_dim = x.shape
        assert N == self.num_pairs, f"Expected {self.num_pairs} pairs, got {N}"
        assert feat_dim == self.feature_dim, f"Expected {self.feature_dim} features, got {feat_dim}"
        
        # Feature fusion
        x = self.feature_fusion(x)  # (B, num_pairs, nb_cnn2d_filt)
        
        # Reshape for attention
        if self.ch_attn_dca:
            x = x.permute(0, 2, 1).contiguous()  # (B, nb_cnn2d_filt, num_pairs)
            x = x.unsqueeze(-1)  # (B, nb_cnn2d_filt, num_pairs, 1)
        
        # Attention stages
        x = self.attention_stage(x)
        
        # Temporal pooling and reshape
        if self.t_pooling_loc == 'end':
            x = x.squeeze(-1)  # (B, nb_cnn2d_filt, num_pairs)
            x = x.permute(0, 2, 1)  # (B, num_pairs, nb_cnn2d_filt)
            x = torch.mean(x, dim=1)  # (B, nb_cnn2d_filt)
        
        # Position and Rotation prediction
        position = self.position_head(x)  # (B, 3)
        rotation = self.rotation_head(x)  # (B, 4)
        
        # Normalize quaternion
        rotation = F.normalize(rotation, p=2, dim=1)
        
        # Concatenate position and rotation
        dov = torch.cat([position, rotation], dim=1)  # (B, 7)
        
        return dov