import os
import numpy as np
import torch
from torch.utils.data import Dataset
import quaternion
import torchaudio

class DataGenerator(Dataset):
    def __init__(self, params, split=1, shuffle=True, per_file=False, is_eval=False):
        super().__init__()
        self._batch_size = params['batch_size']
        self._shuffle = shuffle
        self._is_eval = is_eval
        
        # 샘플링 레이트 설정
        self.original_sr = params['original_sr']  # 48000
        self.target_sr = params['fs']            # 16000
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.original_sr,
            new_freq=self.target_sr
        )
        
        # SounDR 데이터 경로
        self.data_dir = params['dataset_dir']
        self.input_path = os.path.join(self.data_dir, "input.npy")
        self.output_path = os.path.join(self.data_dir, "output.npy")
        self.starts_path = os.path.join(self.data_dir, "starts.npy")
        
        # 데이터 로드
        self.input_data = np.load(self.input_path, mmap_mode='r')
        self.output_data = np.load(self.output_path)
        self.starts = np.load(self.starts_path, allow_pickle=True)
        
        # VAD 마스킹된 유효 인덱스 저장
        vad_mask = self.output_data[:, -1] > 0
        self.valid_indices = np.where(vad_mask)[0]
        
        # 데이터 크기 확인
        self._nb_total_samples = len(self.valid_indices)
        self._nb_total_batches = self._nb_total_samples // self._batch_size
        
        # 특성 차원
        self._nb_channels = params['nb_channels']  # 16
        self._sig_len = params['sig_len']         # 1600 (다운샘플링 후)
        
        # 정규화를 위한 epsilon
        self.eps = 1e-8
        
        print(
            '\n=== Data Generator 초기화 ===\n'
            f'모드: {"평가" if self._is_eval else "학습"}\n'
            f'총 샘플 수: {self._nb_total_samples}\n'
            f'배치 크기: {self._batch_size}\n'
            f'셔플: {self._shuffle}\n'
            f'총 배치 수: {self._nb_total_batches}\n'
            f'입력 shape: {self.input_data.shape}\n'
            f'출력 shape: {self.output_data.shape}\n'
            f'원본 SR: {self.original_sr}Hz\n'
            f'목표 SR: {self.target_sr}Hz\n'
            '===========================\n'
        )

    def __len__(self):
        return self._nb_total_samples

    def __getitem__(self, idx):
        """단일 샘플 반환"""
        real_idx = self.valid_indices[idx]
        
        # 오디오 데이터 로드 및 다운샘플링
        X = torch.from_numpy(self.input_data[real_idx].copy()).float()
        X_downsampled = torch.zeros(X.shape[0], self._sig_len, device=X.device)
        
        # 배치 단위로 다운샘플링
        X_flat = X.reshape(-1)  # 모든 채널을 일렬로
        X_down_flat = self.resampler(X_flat)
        X_downsampled = X_down_flat.reshape(X.shape[0], -1)
        
        # 채널별 정규화
        std = torch.std(X_downsampled, dim=1, keepdim=True)
        std = torch.clamp(std, min=self.eps)
        X_downsampled = X_downsampled / std
        
        # 범위 체크 및 클리핑
        X_downsampled = torch.clamp(X_downsampled, min=-10.0, max=10.0)
        
        # NaN 체크
        if torch.isnan(X_downsampled).any():
            print(f"Warning: NaN detected in sample {real_idx}")
            # NaN이 있는 경우 이전 또는 다음 유효한 샘플로 대체
            alt_idx = real_idx + 1 if real_idx < len(self.input_data) - 1 else real_idx - 1
            X = torch.from_numpy(self.input_data[alt_idx].copy()).float()
            X_downsampled = self.process_audio(X)
        
        # DOV 레이블
        y = torch.from_numpy(self.output_data[real_idx, :7]).float()
        
        return X_downsampled, y

    def _quaternion_to_euler(self, quaternions):
        """
        Quaternion을 Euler angles (pitch, yaw)로 변환
        quaternions: [qw, qx, qy, qz] 순서
        """
        # 쿼터니언 순서 변경 (qx,qy,qz,qw -> qw,qx,qy,qz)
        q = np.roll(quaternions, 1, axis=1)
        
        # Pitch (x축 회전)
        sinp = 2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])
        pitch = np.where(np.abs(sinp) >= 1,
                        np.sign(sinp) * np.pi / 2,  # 90도 제한
                        np.arcsin(sinp))

        # Yaw (y축 회전)
        siny_cosp = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
        cosy_cosp = 1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.stack([pitch, yaw], axis=1)

    def process_audio(self, X):
        """오디오 처리 헬퍼 함수"""
        X_flat = X.reshape(-1)
        X_down_flat = self.resampler(X_flat)
        X_downsampled = X_down_flat.reshape(X.shape[0], -1)
        
        std = torch.std(X_downsampled, dim=1, keepdim=True)
        std = torch.clamp(std, min=self.eps)
        X_downsampled = X_downsampled / std
        X_downsampled = torch.clamp(X_downsampled, min=-10.0, max=10.0)
        
        return X_downsampled

    def get_data_sizes(self):
        """데이터 크기 반환"""
        feat_shape = (self._batch_size, self._nb_channels, self._sig_len)
        label_shape = (self._batch_size, 7)  # [x, y, z, qw, qx, qy, qz]
        return feat_shape, label_shape

    def get_nb_classes(self):
        return 7  # DOV 출력 차원 [x, y, z, qw, qx, qy, qz]

    def nb_frames_1s(self):
        return self._sig_len

    def get_hop_len_sec(self):
        return 0.1  # 100ms segments

    def get_filelist(self):
        return [self.input_path]

    def get_frame_per_file(self):
        return self._sig_len

    def get_nb_frames(self):
        return self._sig_len

    def get_data_gen_mode(self):
        return self._is_eval