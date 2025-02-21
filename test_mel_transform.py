import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap
import argparse

def plot_spectrogram(mel_spec, title="Mel Spectrogram"):
    if len(mel_spec.shape) == 1:
        mel_spec = mel_spec.unsqueeze(0)  # 2D로 변환
    
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time frame')
    plt.ylabel('Mel bin')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def test_mel_transform(data_path, batch_idx=0, mic_idx=0):
    # 1. 메모리 맵으로 데이터 로드
    try:
        data = open_memmap(data_path, mode='r')
        print(f"Loaded data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        # 지정된 배치와 마이크의 데이터만 복사
        audio = torch.from_numpy(data[batch_idx, mic_idx].copy()).float()
        print(f"Sample data range: [{data[batch_idx, mic_idx].min():.3f}, {data[batch_idx, mic_idx].max():.3f}]")
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Generating random test data instead...")
        audio = torch.randn(4800)
    
    print(f"\nTest audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    # 2. 멜스펙트로그램 변환기 초기화
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=2048,
        win_length=400,
        hop_length=160,
        n_mels=64,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode='reflect'
    )
    
    # 3. 단계별 변환 테스트
    print("\nTesting mel transformation steps:")
    
    # 기본 멜스펙트로그램
    mel = mel_transform(audio.unsqueeze(0))  # Add batch dimension
    print(f"Raw mel shape: {mel.shape}")
    print(f"Raw mel range: [{mel.min():.3f}, {mel.max():.3f}]")
    plot_spectrogram(torch.log10(mel.clamp(min=1e-10))[0], "Raw Mel Spectrogram")
    
    # 진폭 스펙트로그램
    mel_amp = torch.sqrt(mel.clamp(min=1e-10))
    print(f"\nAmplitude mel range: [{mel_amp.min():.3f}, {mel_amp.max():.3f}]")
    plot_spectrogram(mel_amp[0], "Amplitude Mel Spectrogram")
    
    # 정규화 테스트
    # 1) Min-Max 정규화
    mel_max = mel_amp.amax(dim=(1,2), keepdim=True)
    mel_min = mel_amp.amin(dim=(1,2), keepdim=True)
    mel_norm1 = (mel_amp - mel_min) / (mel_max - mel_min + 1e-8)
    print(f"\nMin-Max normalized range: [{mel_norm1.min():.3f}, {mel_norm1.max():.3f}]")
    plot_spectrogram(mel_norm1[0], "Min-Max Normalized Mel")
    
    # 2) Z-score 정규화
    mel_mean = mel_amp.mean(dim=(1,2), keepdim=True)
    mel_std = mel_amp.std(dim=(1,2), keepdim=True)
    mel_norm2 = (mel_amp - mel_mean) / (mel_std + 1e-8)
    print(f"Z-score normalized range: [{mel_norm2.min():.3f}, {mel_norm2.max():.3f}]")
    plot_spectrogram(mel_norm2[0], "Z-score Normalized Mel")
    
    # 4. 시간 차원 보간 테스트
    target_length = 49  # NGCC-PHAT에서 사용하는 길이
    mel_interp = torch.nn.functional.interpolate(
        mel_norm1,  # Already has batch dimension
        size=target_length,
        mode='linear',
        align_corners=False
    )
    print(f"\nInterpolated shape: {mel_interp.shape}")
    print(f"Interpolated range: [{mel_interp.min():.3f}, {mel_interp.max():.3f}]")
    plot_spectrogram(mel_interp[0], "Interpolated Mel")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Mel Spectrogram Transformation')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input.npy file')
    parser.add_argument('--batch_idx', type=int, default=0, help='Batch index to test')
    parser.add_argument('--mic_idx', type=int, default=0, help='Microphone index to test')
    
    args = parser.parse_args()
    test_mel_transform(args.data_path, args.batch_idx, args.mic_idx)