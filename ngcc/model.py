import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from ngcc.dnn_models import SincNet

def get_pad(size, kernel_size, stride=1, dilation=1):
    out_size = (size + stride - 1) // stride
    pad = max(0, (out_size - 1) * stride + (kernel_size-1) * dilation + 1 - size)
    pad_1 = pad // 2
    pad_2 = pad - pad_1
    return (pad_1, pad_2)

class GCC(nn.Module):
    def __init__(self, max_tau=None, dim=2, filt='phat', epsilon=1e-8):
        super().__init__()
        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, y):
        B, C, time_dim = x.shape
        n = time_dim + time_dim
        
        x = x.float().contiguous()
        y = y.float().contiguous()
        
        X = torch.fft.rfft(x, n=n, dim=-1)
        Y = torch.fft.rfft(y, n=n, dim=-1)
        
        Gxy = X * torch.conj(Y)
        
        if self.filt == 'phat':
            abs_Gxy = torch.abs(Gxy) + self.epsilon
            Gxy = Gxy / abs_Gxy.clamp(min=self.epsilon)
        
        cc = torch.fft.irfft(Gxy, n=n, dim=-1)
        
        max_shift = self.max_tau if self.max_tau else time_dim//2
        result = torch.cat((cc[..., -max_shift:], cc[..., :max_shift+1]), dim=-1)
        
        return result

class NGCCPHAT(nn.Module):
    def __init__(self, max_tau=24, n_mel_bins=64, sig_len=1600,
                 num_channels=32, fs=16000, normalize_input=True):
        super().__init__()
        
        self.max_tau = max_tau
        self.normalize_input = normalize_input
        self.n_mel_bins = n_mel_bins
        self.num_mics = 16
        self.num_pairs = (self.num_mics * (self.num_mics - 1)) // 2
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            win_length=400,
            hop_length=160,
            n_mels=64,
            power=2.0,
            normalized=True,
            center=True,
            pad_mode='reflect',
            norm='slaney',
        )
            
        sincnet_params = {
            'input_dim': sig_len,
            'fs': fs,
            'cnn_N_filt': [num_channels] * 4,
            'cnn_len_filt': [sig_len-1, 11, 9, 7],
            'cnn_max_pool_len': [1, 1, 1, 1],
            'cnn_use_laynorm_inp': False,
            'cnn_use_batchnorm_inp': False,
            'cnn_use_laynorm': [False] * 4,
            'cnn_use_batchnorm': [True] * 4,
            'cnn_act': ['leaky_relu'] * 3 + ['linear'],
            'cnn_drop': [0.0] * 4,
            'use_sinc': True,
        }
        
        self.backbone = SincNet(sincnet_params)
        self.gcc = GCC(max_tau=self.max_tau)
        
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(num_channels + n_mel_bins, num_channels, 1),
            nn.BatchNorm1d(num_channels),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def process_mel_features(self, audio, tau):
        B, M, L = audio.shape
        mel_features = []
        
        for m in range(M):
            curr_audio = audio[:, m]
            std = torch.std(curr_audio, dim=-1, keepdim=True)
            std = torch.clamp(std, min=1e-5)
            curr_audio = curr_audio / std
            
            mel = self.mel_transform(curr_audio)
            mel = torch.log1p(mel.clamp(min=1e-5))
            
            mel_mean = mel.mean(dim=(1,2), keepdim=True)
            mel_std = mel.std(dim=(1,2), keepdim=True)
            mel_std = torch.clamp(mel_std, min=1e-5)
            mel = (mel - mel_mean) / mel_std
            
            mel_features.append(mel)
        
        mel_features = torch.stack(mel_features, dim=1)
        
        B, M, n_mels, time_dim = mel_features.shape
        mel_features = mel_features.reshape(B*M, n_mels, time_dim)
        
        mel_features = F.interpolate(
            mel_features,
            size=tau,
            mode='linear',
            align_corners=False
        )
        
        mel_features = mel_features.reshape(B, M, n_mels, tau)
        mel_features = mel_features.repeat_interleave(M-1, dim=1)[:, :self.num_pairs]
        mel_features = mel_features.reshape(B*self.num_pairs, n_mels, tau)
        
        mel_features = torch.clamp(mel_features, min=-5.0, max=5.0)
        
        return mel_features

    @torch.amp.autocast('cuda')
    def forward(self, audio):
        B, M, time_dim = audio.shape
        
        if self.normalize_input:
            std = torch.clamp(audio.std(dim=-1, keepdim=True), min=1e-8)
            audio = audio / std
        
        x = audio.reshape(-1, 1, time_dim)
        x = self.backbone(x)
        _, C, L = x.shape
        x = x.reshape(B, M, C, L)
        
        gcc_features = []
        for m1 in range(M):
            if m1 < M - 1:
                x1 = x[:, m1:m1+1]
                x2 = x[:, m1+1:]
                x1 = x1.expand(-1, M-m1-1, -1, -1)
                
                gcc = self.gcc(x1.reshape(-1, C, L), x2.reshape(-1, C, L))
                gcc = gcc.reshape(B, M-m1-1, C, -1)
                gcc_features.extend(gcc.unbind(dim=1))

        gcc_features = torch.stack(gcc_features, dim=1)
        B, N, C, tau = gcc_features.shape
        gcc_features = gcc_features.reshape(B*N, C, tau)
        
        mel_features = self.process_mel_features(audio, tau)
        
        concat_features = torch.cat([gcc_features, mel_features], dim=1)
        fused_features = self.fusion_layer(concat_features)
        
        output_features = self.mlp(fused_features)
        output_features = output_features.mean(dim=2)
        output_features = output_features.reshape(B, N, -1)
        
        return output_features

    @staticmethod
    def profile_forward(model, input_shape, device='cuda'):
        input_tensor = torch.randn(*input_shape, device=device)
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            model(input_tensor)
        
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10))
        return prof