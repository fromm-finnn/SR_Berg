import torch
from parameters import get_params
from cls_data_generator import DataGenerator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def test_dataloader():
    # 파라미터 로드
    params = get_params()
    
    print("1. 데이터셋 초기화 중...")
    # 데이터셋 생성
    dataset = DataGenerator(params)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print("\n2. 데이터 샘플 로드 중...")
    # 첫 번째 배치 가져오기
    audio, labels = next(iter(dataloader))
    
    print("\n3. 데이터 형태 확인:")
    print(f"Audio shape: {audio.shape}")  # Expected: (2, 16, 4800)
    print(f"Labels shape: {labels.shape}")  # Expected: (2, 2)
    print(f"Labels (pitch, yaw):\n{labels}")
    
    print("\n4. 오디오 데이터 통계:")
    print(f"Audio mean: {audio.mean():.4f}")
    print(f"Audio std: {audio.std():.4f}")
    print(f"Audio min: {audio.min():.4f}")
    print(f"Audio max: {audio.max():.4f}")
    
    print("\n5. 시각화...")
    # 첫 번째 샘플의 첫 번째 채널 파형 그리기
    plt.figure(figsize=(15, 5))
    plt.plot(audio[0, 0].numpy())
    plt.title("First channel waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig("waveform.png")
    plt.close()
    
    # 모든 채널의 상관관계 확인
    correlation = np.corrcoef(audio[0].numpy())
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation, cmap='coolwarm')
    plt.colorbar()
    plt.title("Channel correlation matrix")
    plt.savefig("correlation.png")
    plt.close()
    
    print("\n6. VAD 마스크 확인:")
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

if __name__ == "__main__":
    test_dataloader()