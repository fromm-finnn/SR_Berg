import torch
from parameters import get_params
from ngcc.model import NGCCPHAT
from cst_former.CST_former_model import CST_former

def test_model():
    # 파라미터 로드
    params = get_params()
    
    print("1. NGCC-PHAT 모델 테스트")
    # NGCC-PHAT 모델 초기화
    ngcc_model = NGCCPHAT(
        max_tau=params['max_tau'],
        n_mel_bins=params['nb_mel_bins'],
        sig_len=params['sig_len'],
        num_channels=params['ngcc_channels'],
        num_out_channels=params['ngcc_out_channels'],
        fs=params['fs']
    ).cuda()
    
    # 테스트 입력
    test_input = torch.randn(2, 16, 4800).cuda()
    print(f"Input shape: {test_input.shape}")
    
    # NGCC-PHAT forward pass
    with torch.no_grad():
        ngcc_output = ngcc_model(test_input)
    print(f"NGCC-PHAT output shape: {ngcc_output.shape}")
    
    print("\n2. CST-former 모델 테스트")
    # CST-former 모델 초기화
    model = CST_former(
        in_feat_shape=(2, 120, 2064),  # NGCC-PHAT 출력 shape에 맞춤
        out_shape=(2, 2),
        params=params
    ).cuda()
    
    # CST-former forward pass
    with torch.no_grad():
        output = model(ngcc_output)  # NGCC-PHAT의 출력을 입력으로 사용
    print(f"CST-former output shape: {output.shape}")
    
    print("\n3. 모델 파라미터 수")
    ngcc_params = sum(p.numel() for p in ngcc_model.parameters())
    cst_params = sum(p.numel() for p in model.parameters())
    print(f"NGCC-PHAT parameters: {ngcc_params:,}")
    print(f"CST-former parameters: {cst_params:,}")
    print(f"Total parameters: {ngcc_params + cst_params:,}")
    
    print("\n4. 메모리 사용량")
    print(f"NGCC-PHAT memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    torch.cuda.empty_cache()
    print(f"CST-former memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_model()