import numpy as np
import torch
import torch.nn as nn

from einops import rearrange
from .CST_details.encoder import Encoder
from .CST_details.CST_encoder import CST_encoder
from .CST_details.CMT_Block import CMT_block
from .CST_details.layers import FC_layer
from ngcc.model import NGCCPHAT

"""
CST_former (Channel-Spectral-Temporal Transformer) 모델 구현

이 모듈은 SELD(Sound Event Localization and Detection) 작업을 위한 
CST_former 모델을 구현합니다. 이 모델은 다음 세 가지 주요 특성을 결합합니다:
1. 채널 어텐션(Channel Attention): 다중 마이크 입력 간의 관계 모델링
2. 스펙트럴 어텐션(Spectral Attention): 주파수 도메인에서의 관계 모델링
3. 시간적 어텐션(Temporal Attention): 시간 도메인에서의 관계 모델링

NGCC(Neural Generalized Cross-Correlation) 특징을 입력으로 사용합니다.
"""


class CST_former(torch.nn.Module):
    """
    CST_former : Channel-Spectral-Temporal Transformer for SELD task
    SELD(음향 이벤트 위치 추정 및 감지) 작업을 위한 채널-스펙트럴-시간 트랜스포머
    """
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        """
        CST_former 모델 초기화
        
        Args:
            in_feat_shape: 입력 특징 형태
            out_shape: 출력 형태
            params: 모델 파라미터 딕셔너리
            in_vid_feat_shape: 비디오 특징 형태 (사용되는 경우)
        """
        super().__init__()
        self.nb_classes = params['unique_classes']  # 고유 클래스 수
        self.t_pooling_loc = params["t_pooling_loc"]  # 시간 풀링 위치
        self.ch_attn_dca = params['ChAtten_DCA']  # 채널 어텐션 DCA 사용 여부
        self.ch_attn_unfold = params['ChAtten_ULE']  # 채널 어텐션 ULE 사용 여부
        self.cmt_block = params['CMT_block']  # CMT 블록 사용 여부
        self.encoder = Encoder(in_feat_shape, params)  # 인코더 초기화
        self.mel_bins = params['nb_mel_bins']  # 멜 빈 수
        self.fs = params['fs']  # 샘플링 주파수
        self.sig_len = int(self.fs * params['hop_len_s'])  # 480 samples, 신호 길이
        self.predict_tdoa = params['predict_tdoa']  # TDOA 예측 여부
        self.use_ngcc = params['use_ngcc']  # NGCC 사용 여부

        if params['use_ngcc']:
            self.ngcc_channels = params['ngcc_channels']  # NGCC 채널 수
            self.ngcc_out_channels = params['ngcc_out_channels']  # NGCC 출력 채널 수

        self.input_nb_ch = params['nb_channels']  # 입력 채널 수
            #if params['use_mel']:
            #    self.input_nb_ch = int(self.ngcc_out_channels * params['n_mics'] * (params['n_mics'] - 1) / 2 +  params['n_mics'])
            #else:
            #    self.input_nb_ch = int(self.ngcc_out_channels * params['n_mics'] * ( 1 + (params['n_mics'] - 1) / 2))
        #elif params['use_salsalite']:
        #    self.input_nb_ch = 7
        #else:
        #    self.input_nb_ch = 10

        # NGCC 모듈 초기화 (사용하는 경우)
        if params['use_ngcc']:
            self.ngcc = NGCCPHAT(max_tau=params['max_tau'], n_mel_bins=self.mel_bins , use_sinc=True,
                                        sig_len=self.sig_len , num_channels=self.ngcc_channels, num_out_channels=self.ngcc_out_channels, fs=self.fs,
                                        normalize_input=False, normalize_output=False, pool_len=1, use_mel=params['use_mel'], use_mfcc=params['use_mfcc'],
                                        predict_tdoa=params['predict_tdoa'], tracks=params['tracks'], fixed_tdoa=params['fixed_tdoa'])

        # 사용할 주파수 빈 수 결정
        if params['use_salsalite']:
            bins = 382
        else:
            bins = params['nb_mel_bins']
        
        # 컨볼루션 블록 주파수 차원 계산
        self.conv_block_freq_dim = int(np.floor(bins / np.prod(params['f_pool_size'])))
        
        # 시간적 임베딩 차원 계산 (채널 어텐션 사용 여부에 따라 달라짐)
        self.temp_embed_dim = self.conv_block_freq_dim * params['nb_cnn2d_filt'] * self.input_nb_ch if self.ch_attn_dca \
            else self.conv_block_freq_dim * params['nb_cnn2d_filt']

        ## 어텐션 레이어 ===========================================================================================##
        # CMT 블록 사용 여부에 따라 어텐션 단계 결정
        if not self.cmt_block:
            self.attention_stage = CST_encoder(self.temp_embed_dim, params)
        else:
            self.attention_stage = CMT_block(params, self.temp_embed_dim)

        # 시간 풀링 위치가 끝인 경우 풀링 레이어 설정
        if self.t_pooling_loc == 'end':
            if not params["f_pool_size"] == [1,1,1]:
                self.t_pooling = nn.MaxPool2d((5,1))
            else:
                self.t_pooling = nn.MaxPool2d((5,4))

        ## 완전 연결 레이어 ========================================================================================##
        self.fc_layer = FC_layer(out_shape, self.temp_embed_dim, params)

        # 가중치 초기화 적용
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        모델 가중치 초기화 메서드
        
        Args:
            m: 초기화할 모듈
        """
        if isinstance(m, nn.Linear):
            # JAX ViT 공식 구현을 따라 xavier_uniform 초기화 사용
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, video=None):
        """
        모델의 순전파(forward) 메서드
        
        Args:
            x: 입력 텐서, 형태: (batch_size, mic_channels, time_steps, mel_bins)
            video: 비디오 입력 (사용되는 경우)
            
        Returns:
            doa: 방향 각도(Direction of Arrival) 출력
            또는 doa와 tdoa(Time Difference of Arrival)의 튜플 (predict_tdoa가 True인 경우)
        """
        B, M, T, F = x.size()  # 배치 크기, 마이크 채널 수, 시간 단계, 멜 빈 수

        # NGCC 사용 시 특징 추출
        if self.use_ngcc:
            if self.predict_tdoa:
                x, tdoa = self.ngcc(x)
            else:
                x = self.ngcc(x)

        # 채널 어텐션 DCA 사용 시 텐서 재배열
        if self.ch_attn_dca:
            x = rearrange(x, 'b m t f -> (b m) 1 t f', b=B, m=M, t=T, f=F).contiguous()
        
        # 인코더와 어텐션 단계 통과
        x = self.encoder(x)  # 출력: [(b m) c t f] (ch_attn_dca 사용 시) 또는 [b c t f]
        x = self.attention_stage(x)

        # 시간 풀링 적용 (위치가 'end'인 경우)
        if self.t_pooling_loc == 'end':
            x = self.t_pooling(x)

        # 완전 연결 레이어로 방향 각도(DOA) 예측
        doa = self.fc_layer(x)
        
        # TDOA 예측 여부에 따라 다른 출력 반환
        if self.predict_tdoa:
            return doa, tdoa[:, ::self.pool_len]  # 올바른 해상도를 얻기 위해 tdoa 풀링
        else:
            return doa 
