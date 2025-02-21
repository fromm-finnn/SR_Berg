def get_params(argv='1'):
    print("SET: {}".format(argv))
    
    params = dict(
        # INPUT/OUTPUT PATHS
        dataset_dir='../soundr-dl/data',
        model_dir='models',
        
        # DATASET PARAMETERS
        fs=16000,
        hop_len_s=0.1,
        sig_len=1600,
        nb_channels=16,
        
        # SAMPLING PARAMETERS
        original_sr=48000,
        target_sr=16000,
        
        # MODEL PARAMETERS
        model='cstformer',
        use_ngcc=True,
        predict_dov=True,
        dov_output_size=7,  # [x,y,z, qw,qx,qy,qz]
        
        # ENCODER PARAMETERS
        encoder='CNN',
        nb_cnn2d_filt=64,
        kernel_size=[3, 3, 3],
        nb_blocks=4,
        nb_conv_blocks=2,
        pool_size=[[2, 2], [2, 2], [1, 1]],
        dropout_rate=0.2,
        
        # NGCC-PHAT PARAMETERS
        ngcc_channels=32,
        ngcc_out_channels=16,
        max_tau=24,
        nb_mel_bins=64,  # 추가
        n_mel_bins=64,   # 추가 (nb_mel_bins와 동일)
        use_mel=True,
        use_mfcc=False,
        
        # MEL-SPECTROGRAM PARAMETERS
        n_fft=2048,      # 추가
        win_length=400,  # 추가
        hop_length=160,  # 추가
        
        # FEATURE FUSION PARAMETERS
        fusion_dim=128,  # 특징 융합 차원
        use_feature_fusion=True,  # 특징 융합 사용 여부
        
        # CST-FORMER PARAMETERS
        FreqAtten=True,
        ChAtten_DCA=True,
        ChAtten_ULE=True,
        CMT_block=True,
        f_pool_size=[1,1,1],
        t_pool_size=[1,1,1],
        t_pooling_loc='end',
        nb_fnn_layers=1,
        fnn_size=256,
        
        # ATTENTION PARAMETERS
        nb_self_attn_layers=4,
        nb_heads=8,
        attn_dropout=0.1,
        ff_multiplier=4,
        ff_dropout=0.1,
        
        # CMT BLOCK PARAMETERS
        cmt_dim=512,
        cmt_depth=2,
        cmt_kernel_size=3,
        cmt_expansion_factor=4,
        CMT_split=True,
        CMT_atten=True,
        CMT_shift=True,
        CMT_compatibility=True,
        CMT_init_scale=1.0,
        CMT_drop_path=0.1,
        CMT_head_dim=32,
        CMT_stem_dim=64,
        
        # SPECTRAL ATTENTION PARAMETERS
        linear_layer=True,
        spec_atten_dim=256,
        spec_key_dim=32,
        spec_num_heads=8,
        spec_attn_dropout=0.1,
        spec_proj_dropout=0.1,
        spec_qkv_bias=False,
        spec_out_bias=False,
        
        # TRAINING PARAMETERS
        batch_size=32,
        eval_batch_size=32,
        nb_epochs=100,
        lr=1e-4,
        final_lr=1e-5,
        warmup=5,
        weight_decay=0.05,
        
        # EVALUATION PARAMETERS
        evaluate_distance=True,  # DOV에서는 거리 평가 필요
        segment_based_metrics=True,
        
        # DOV SPECIFIC PARAMETERS
        position_threshold=0.35,  # meters
        rotation_threshold=25.0,  # degrees
    )
    
    return params