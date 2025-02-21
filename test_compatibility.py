import torch
from parameters import get_params
from cls_data_generator import DataGenerator
from seldnet_model import get_model

def test_compatibility():
    # 1. STARSS 데이터셋 테스트
    print("Testing STARSS dataset compatibility...")
    params = get_params()
    params['dataset'] = 'starss'
    params['use_ngcc'] = True
    
    # DataGenerator 초기화
    data_gen = DataGenerator(params)
    feat_shape = data_gen.get_feat_shape()
    out_shape = data_gen.get_label_shape()
    
    # 모델 생성
    model, loss_fn = get_model(params, feat_shape, out_shape)
    
    # 더미 데이터로 테스트
    batch_size = 4
    dummy_input = torch.randn(batch_size, *feat_shape)
    try:
        output = model(dummy_input)
        print(f"STARSS Output shape: {output.shape}")
        print("STARSS test passed!")
    except Exception as e:
        print(f"STARSS test failed: {str(e)}")

    # 2. SounDR 데이터셋 테스트
    print("\nTesting SounDR dataset compatibility...")
    params['dataset'] = 'soundr'
    
    # DataGenerator 초기화
    data_gen = DataGenerator(params)
    feat_shape = data_gen.get_feat_shape()
    out_shape = (None, 5)  # x,y,z + pitch,yaw
    
    # 모델 생성
    model, loss_fn = get_model(params, feat_shape, out_shape)
    
    # 더미 데이터로 테스트
    dummy_input = torch.randn(batch_size, *feat_shape)
    try:
        output = model(dummy_input)
        print(f"SounDR Output shape: {output.shape}")
        print("SounDR test passed!")
    except Exception as e:
        print(f"SounDR test failed: {str(e)}")

if __name__ == "__main__":
    test_compatibility()