import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch_same_pad import get_pad
from ngcc.dnn_models import SincNet
import torch.fft
import librosa
import torchaudio

"""
NGCC(Neural Generalized Cross-Correlation) 모델 구현

이 모듈은 SELD(음향 이벤트 위치 추정 및 감지) 작업을 위한 NGCC 모델을 구현합니다.
GCC-PHAT(Generalized Cross-Correlation with Phase Transform) 알고리즘의 신경망 기반 확장 버전으로,
다중 마이크 오디오 입력에서 공간적 특징을 추출하기 위해 사용됩니다.
"""

def get_pad(size, kernel_size, stride=1, dilation=1):
    """
    Conv1d에서 'SAME' 패딩을 위한 패딩 계산
    
    Args:
        size: 입력 크기
        kernel_size: 커널 크기
        stride: 스트라이드
        dilation: 확장(dilation)
        
    Returns:
        padding: 왼쪽 및 오른쪽 패딩 값
    """
    # 확장을 고려한 유효 커널 크기 계산
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    
    # 출력 크기 계산
    out_size = (size + stride - 1) // stride
    
    # 필요한 패딩 계산
    padding_needed = max(0, (out_size - 1) * stride + effective_kernel_size - size)
    
    # 왼쪽 및 오른쪽 패딩 계산
    padding_left = padding_needed // 2
    padding_right = padding_needed - padding_left
    
    return (padding_left, padding_right)

def next_greater_power_of_2(x):
    """
    주어진 수보다 크거나 같은 2의 거듭제곱 중 가장 작은 값 반환
    
    Args:
        x: 입력 숫자
        
    Returns:
        2의 거듭제곱 값
    """
    return 2 ** (x - 1).bit_length()

class GCC(nn.Module):
    """
    일반화된 상호 상관 방법(Generalized Cross-Correlation) 구현 클래스
    
    Knapp와 Carter의 논문 "The Generalized Correlation Method for Estimation of Time Delay",
    IEEE Trans. Acoust., Speech, Signal Processing, August, 1976을 기반으로 함
    """
    def __init__(self, max_tau=None, dim=2, filt='phat', epsilon=0.001, beta=None):
        """
        GCC 클래스 초기화
        
        Args:
            max_tau: 고려할 최대 지연 시간
            dim: 차원 수
            filt: 필터 유형 ('phat', 'roth', 'scot', 'ht', 'cc' 중 하나)
            epsilon: 수치 안정성을 위한 작은 상수
            beta: 필터 가중치 지수 (None이 아닌 경우 사용)
        """
        super().__init__()

        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, x, y):
        """
        두 신호 x와 y의 일반화된 상호 상관 계산
        
        Args:
            x: 첫 번째 신호
            y: 두 번째 신호
            
        Returns:
            cc: 상호 상관 결과
        """
        n = x.shape[-1] + y.shape[-1]

        # 일반화된 상호 상관 위상 변환
        X = torch.fft.rfft(x, n=n)  # x의 푸리에 변환
        Y = torch.fft.rfft(y, n=n)  # y의 푸리에 변환
        Gxy = X * torch.conj(Y)     # 크로스 스펙트럼 계산

        # 필터 유형에 따른 가중치 함수(phi) 선택
        if self.filt == 'phat':  # Phase Transform
            phi = 1 / (torch.abs(Gxy) + self.epsilon)

        elif self.filt == 'roth':  # Roth 필터
            phi = 1 / (X * torch.conj(X) + self.epsilon)

        elif self.filt == 'scot':  # SCOT(Smoothed Coherence Transform) 필터
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            phi = 1 / (torch.sqrt(Gxx * Gyy) + self.epsilon)

        elif self.filt == 'ht':  # HT(Hannan-Thomson) 필터
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            gamma = Gxy / torch.sqrt(Gxx * Gxy)
            phi = torch.abs(gamma)**2 / (torch.abs(Gxy)
                                         * (1 - gamma)**2 + self.epsilon)

        elif self.filt == 'cc':  # 일반 상호 상관(Cross-Correlation)
            phi = 1.0

        else:
            raise ValueError('지원되지 않는 필터 함수입니다')

        # 필터 가중치 지수(beta)가 제공된 경우 다중 필터 적용
        if self.beta is not None:
            cc = []
            for i in range(self.beta.shape[0]):
                cc.append(torch.fft.irfft(
                    Gxy * torch.pow(phi, self.beta[i]), n))

            cc = torch.cat(cc, dim=1)

        else:
            # 단일 필터 적용하여 역변환
            cc = torch.fft.irfft(Gxy * phi, n)

        # 최대 시간 지연 범위 설정
        max_shift = int(n / 2)
        if self.max_tau:
            max_shift = np.minimum(self.max_tau, int(max_shift))

        # 차원에 따른 상호 상관 결과 처리
        if self.dim == 2:
            cc = torch.cat((cc[:, -max_shift:], cc[:, :max_shift+1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat(
                (cc[:, :, -max_shift:], cc[:, :, :max_shift+1]), dim=-1)
        elif self.dim == 4:
            cc = torch.cat(
                (cc[:, :, :, -max_shift:], cc[:, :, :, :max_shift+1]), dim=-1)

        return cc


class NGCCPHAT(nn.Module):
    """
    SincNet 백본을 사용한 신경망 GCC-PHAT 구현
    
    다중 마이크 입력에서 공간적 특징을 추출하기 위한 심층 신경망 모델로,
    SELD 작업에 필요한 방향 정보를 포함한 특징을 생성합니다.
    """
    def __init__(self, max_tau=64, n_mel_bins=64, use_sinc=True,
                                        sig_len=960, num_channels=128, num_out_channels=8, fs=24000,
                                        normalize_input=True, normalize_output=False, pool_len=5, use_mel=True, use_mfcc=False,
                                        tracks=5, predict_tdoa=False, fixed_tdoa=False):
        """
        NGCCPHAT 모델 초기화
        
        Args:
            max_tau: 고려할 최대 지연 시간
            n_mel_bins: 멜 스케일 빈(bin)의 수
            use_sinc: SincNet 백본 사용 여부
            sig_len: 입력 신호 길이
            num_channels: GCC 상관 채널 수
            num_out_channels: 출력 채널 수
            fs: 샘플링 주파수
            normalize_input: 입력 정규화 여부
            normalize_output: 출력 정규화 여부
            pool_len: 풀링 길이
            use_mel: 멜 스펙트로그램 사용 여부
            use_mfcc: MFCC 사용 여부
            tracks: 트랙 수
            predict_tdoa: TDOA(Time Difference of Arrival) 예측 여부
            fixed_tdoa: 고정 TDOA 사용 여부
        """
        super().__init__()

        self.max_tau = max_tau
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.pool_len = pool_len
        self.n_mel_bins = n_mel_bins
        self.use_mel = use_mel
        self.use_mfcc = use_mfcc
        self.tracks = tracks
        self.predict_tdoa = predict_tdoa
        self.fixed_tdoa = fixed_tdoa 

        # SincNet 파라미터 설정
        sincnet_params = {'input_dim': sig_len,
                          'fs': fs,
                          'cnn_N_filt': [num_channels, num_channels, num_channels, num_channels],
                          'cnn_len_filt': [sig_len-1, 11, 9, 7],
                          'cnn_max_pool_len': [1, 1, 1, 1],
                          'cnn_use_laynorm_inp': False,
                          'cnn_use_batchnorm_inp': False,
                          'cnn_use_laynorm': [False, False, False, False],
                          'cnn_use_batchnorm': [True, True, True, True],
                          'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'linear'],
                          'cnn_drop': [0.0, 0.0, 0.0, 0.0],
                          'use_sinc': use_sinc,
                          } 

        # 백본 및 풀링 레이어 정의
        self.backbone = SincNet(sincnet_params)
        self.pool = torch.nn.AvgPool2d((pool_len, 1))  # 시간 축 평균 풀링
        self.mlp_kernels = [11, 9, 7]  # MLP 레이어 커널 크기
        self.channels = [num_channels, num_channels, num_channels, num_channels]  # 채널 수 설정
        self.final_kernel = 3  # 최종 컨볼루션 커널 크기

        # GCC 모듈 초기화
        self.gcc = GCC(max_tau=self.max_tau, dim=4, filt='phat')

        # MLP 레이어 구성 (GCC 특징 처리용)
        self.mlp = nn.ModuleList([nn.Sequential(
                nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size=k),
                nn.BatchNorm1d(self.channels[i+1]),
                nn.LeakyReLU(0.2)) for i, k in enumerate(self.mlp_kernels)])
        
        # 최종 컨볼루션 레이어
        self.final_conv = nn.Sequential(nn.Conv1d(num_channels, num_out_channels, kernel_size=self.final_kernel),
                                        nn.BatchNorm1d(num_out_channels),
                                        nn.LeakyReLU(0.2))

        # TDOA 예측 레이어 (필요한 경우)
        if self.predict_tdoa:
            self.tdoa_conv = nn.Conv1d(num_out_channels, tracks, kernel_size=self.final_kernel)

        # 스펙트럼 컨볼루션 레이어
        self.spec_conv = nn.Sequential(
                nn.Conv1d(num_channels, num_out_channels, kernel_size=self.final_kernel, stride=self.final_kernel),
                nn.BatchNorm1d(num_out_channels),
                nn.GELU()
        )
        
        # 상호 상관 투영 레이어
        self.cc_proj = nn.Sequential(
                nn.Linear(max_tau*2+1, self.n_mel_bins // 2),
                nn.LayerNorm(self.n_mel_bins // 2),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(self.n_mel_bins // 2, self.n_mel_bins)
        )

        # 멜 스펙트로그램 변환 설정
        if self.use_mel:
            self.nfft = next_greater_power_of_2(2 * sig_len)
            self.spec_transform = torchaudio.transforms.Spectrogram(n_fft=self.nfft, win_length=2*sig_len, hop_length=sig_len, normalized=True)
            self.mel_transform = torchaudio.transforms.MelScale(n_mels=self.n_mel_bins, sample_rate=fs, n_stft=self.nfft//2+1, norm='slaney')
            self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
            
            # MFCC 변환 설정 (사용하는 경우)
            if self.use_mfcc:
                melkwargs = {"n_fft": self.nfft, "win_length": 2*sig_len, "power": 1,
                                         "hop_length": sig_len, "n_mels": 80, "f_min": 20, "f_max": 7000}
                self.mfcc = torchaudio.transforms.MFCC(sample_rate=fs,
                                               n_mfcc=n_mel_bins, log_mels=True,
                                               melkwargs=melkwargs)
        # 멜 변환을 사용하지 않는 경우 투영 레이어 설정
        else:
            in_size = sig_len // self.final_kernel
            self.proj = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_size, in_size // 2),
                    nn.LayerNorm(in_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.5),
                    nn.Linear(in_size // 2, self.n_mel_bins)
            )

    def forward(self, audio):
        """
        NGCCPHAT 모델의 순전파
        
        다중 마이크 오디오 신호에서 공간적 특징을 추출하는 과정을 수행합니다.
        
        Args:
            audio: 입력 오디오 텐서, 형태: (batch_size, #mics, #time_windows, win_len)
            
        Returns:
            feat: 추출된 특징 텐서
            또는 (feat, cc_out) 튜플 (predict_tdoa가 True인 경우)
        """
        # 입력 정규화 (필요한 경우)
        if self.normalize_input:
            audio /= audio.std(dim=-1, keepdims=True)

        # TDOA가 고정된 경우 그라디언트 계산 비활성화
        with torch.set_grad_enabled(not self.fixed_tdoa):
            # 신호 필터링
            B, M, T, L = audio.shape  # (배치 크기, 마이크 수, 시간 윈도우 수, 윈도우 길이)
            x = audio.reshape(-1, 1, T*L)
            x = self.backbone(x)  # SincNet 백본 통과

            _, C, _ = x.shape
            L_spec = int(L // self.final_kernel)
            x_cc = x.reshape(B, M, C, T*L)  # (배치 크기, 마이크 수, 채널, 시간 윈도우 * 윈도우 길이)
            x_cc = x.reshape(B, M, C, T, L).permute(0, 1, 3, 2, 4)  # (배치 크기, 마이크 수, 시간 윈도우, 채널, 윈도우 길이)

        # 멜 스펙트로그램을 사용하지 않는 경우 스펙트럼 특징 계산
        if not self.use_mel:
            x_spec = self.spec_conv(x)
            _, C_spec, _ = x_spec.shape
            x_spec = x_spec.reshape(B, M, C_spec, T*L_spec)  # (배치 크기, 마이크 수, 채널, 시간 윈도우 * 윈도우 길이)
            x_spec = x_spec.reshape(B, M, C_spec, T, L_spec).permute(0, 1, 3, 2, 4)  # (배치 크기, 마이크 수, 시간 윈도우, 채널, 윈도우 길이)

        with torch.set_grad_enabled(not self.fixed_tdoa):
            cc = [] 
            # 쌍별 마이크 조합에 대한 GCC-PHAT 계산
            for m1 in range(0, M):
                for m2 in range(m1+1, M):
                
                    y1 = x_cc[:, m1, :, :, :]
                    y2 = x_cc[:, m2, :, :, :]
                    cc1 = self.gcc(y1, y2)  # (배치 크기, 시간 윈도우, 채널, 지연)
                    cc.append(cc1)

            cc = torch.stack(cc, dim=-1)  # (배치 크기, 시간 윈도우, 채널, 지연, 조합)
            cc = cc.permute(0, 4, 1, 2, 3)  # (배치 크기, 조합, 시간 윈도우, 채널, 지연)

            # GCC 특징 처리를 위한 MLP 레이어 통과
            B, N, _, C, tau = cc.shape
            cc = cc.reshape(-1, C, tau)
            for k, layer in enumerate(self.mlp):
                s = cc.shape[2]
                padding = get_pad(
                    size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1)
                cc = F.pad(cc, pad=padding, mode='constant')
                cc = layer(cc)

            # 최종 컨볼루션 레이어 통과
            s = cc.shape[2]
            padding = get_pad(
                size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
            cc = F.pad(cc, pad=padding, mode='constant')
            cc = self.final_conv(cc)

            # TDOA 예측 (필요한 경우)
            if self.predict_tdoa:
                s = cc.shape[2]
                padding = get_pad(
                    size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
                cc_out = F.pad(cc, pad=padding, mode='constant')
                cc_out = self.tdoa_conv(cc_out)

                _, C, tau = cc_out.shape
                cc_out = cc_out.reshape(B, N, T, self.tracks, tau)

                # 텐서 차원 재배열 (B, T, 13, Tr, ntdoa)
                cc_out = cc_out.permute(0, 2, 4, 3, 1)

        # 상호 상관 특징 형태 변환 및 투영
        _, C, tau = cc.shape
        cc = cc.reshape(B, N, T, C, tau)
        cc = cc.permute(0, 1, 3, 2, 4)  # (배치 크기, 조합, 채널, 시간 윈도우, 지연)
        cc = cc.reshape(B, N * C, T, tau)  # (배치 크기, 조합 * 채널, 시간 윈도우, 지연)
        cc = self.cc_proj(cc)  # 투영 레이어 통과

        # 출력 정규화 (필요한 경우)
        if self.normalize_output:
            cc /= cc.std(dim=-1, keepdims=True)

        # 멜 스펙트로그램 계산
        if self.use_mel:
            B, M, T, L = audio.shape
            audio_in = audio.reshape(B, M, T*L)  # (배치, 마이크, 시간)
            
            # MFCC 또는 멜 스펙트로그램 계산
            if self.use_mfcc:
                mel_spectra = self.mfcc(audio_in)[:, :, :, :T]
            else:
                mag_spectra = self.spec_transform(audio_in)[:, :, :, :T]  # (배치, 마이크, 주파수, 시간)
                mel_spectra = self.mel_transform(mag_spectra)  # (배치, 마이크, 멜 가중치, 시간)
                mel_spectra = self.to_db(mel_spectra)  # dB 변환
            
            mel_spectra = mel_spectra.permute(0, 1, 3, 2)  # (배치, 마이크, 시간, 멜 가중치)

            # 멜 스펙트로그램과 GCC 특징 결합
            feat = torch.cat((mel_spectra, cc), dim=1)
        else:
            # 멜을 사용하지 않는 경우 스펙트럼 특징 처리
            x_spec = x_spec.permute(0, 1, 3, 2, 4)  # (배치 크기, 마이크, 시간 윈도우, 채널, 지연)
            x_spec = x_spec.reshape(B, M * C_spec, T, L_spec)  # (배치 크기, 마이크 * 채널, 시간 윈도우, 지연)
            mel_spectra = self.proj(x_spec)  # 투영 레이어 통과

            # 특징 결합 (멜 가중치와 지연의 수가 같아야 함)
            feat = torch.cat((mel_spectra, cc), dim=1)  # (배치 크기, 마이크 * 채널 + 조합, 시간 윈도우, 멜 가중치)

        # 시간 풀링 적용
        feat = self.pool(feat)

        # 결과 반환 (TDOA 예측 여부에 따라)
        if self.predict_tdoa:
            return feat, cc_out
        else:
            return feat



