# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights

        # INPUT PATH
        dataset_dir='./data_2024/',  # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATHS
        feat_label_dir='./data_2024/seld_feat_label/',  # Directory to dump extracted features and labels

        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='mic',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        raw_chunks=False,
        saved_chunks=False,
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        model = 'seldnet',
        modality='audio',  # 'audio' or 'audio_visual'
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=8,              # Batch size 
        eval_batch_size=8,
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,
        nb_rnn_layers=2,
        rnn_size=128,
        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=50,  # Train for maximum epochs
        eval_freq=25, # evaluate every x epochs
        lr=1e-4,
        final_lr=1e-5, # final learning rate in cosine scheduler
        weight_decay=0.05,
        predict_tdoa=False,
        warmup=5, #number of warmup epochs
        relative_dist = True, # scales MSE loss with 1/d
        no_dist = False, # removes distance from loss, can be used if we don't want to perform distance estimation

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics

        #CST-former params
        encoder = 'conv',           # ['conv', 'ResNet', 'SENet']
        LinearLayer = False,        # Linear Layer right after attention layers (usually not used/employed in baseline model)
        FreqAtten = False,          # Use of Divided Spectro-Temporal Attention (DST Attention)
        ChAtten_DCA = False,        # Use of Divided Channel-S-T Attention (CST Attention)
        ChAtten_ULE = False,        # Use of Divided C-S-T attention with Unfold (Unfolded CST attention)
        CMT_block = False,          # Use of LPU & IRFNN
        CMT_split = False,          # Apply LPU & IRFNN on S, T attention layers independently
        use_ngcc = False,
        use_mfcc = False,
        
    )

    params['feature_label_resolution'] = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * params['feature_label_resolution']
    params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]  # CNN time pooling
    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached
    params['model_dir'] = params['model_dir'] + '_' + params['modality']
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality']

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['n_mics'] = 4

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '9': # TDOA pre-training
        print("RAW AUDIO CHUNKS w/ NGCC model + multi ACCDOA, TDOA-pretraining\n")
        # 특징 추출(batch_feature_extraction.py)을 위한 설정
        # TDOA(Time Difference of Arrival) 사전 학습 모델을 위한 설정
        params['label_sequence_length'] = 1 # TDOA 학습을 위해 한 프레임만 사용
        params['feature_sequence_length'] = params['label_sequence_length'] * params['feature_label_resolution']
        params['raw_chunks'] = True  # 원본 오디오 청크 사용 
        params['pretrained_model_weights'] = 'models/9_tdoa-3tracks-16channels.h5'  # 사전 학습된 모델 가중치 파일
        params['quick_test'] = False
        params['dataset'] = 'mic'  # 마이크로폰 데이터셋 사용
        params['use_salsalite'] = False  # SALSA-Lite 특징 대신 GCC 특징 사용
        params['multi_accdoa'] = True  # 다중 ACCDOA(Angular and Cartesian Distance of Arrival) 방식 사용
        params['n_mics'] = 4  # 마이크 개수
        params['model'] = 'ngccmodel'  # NGCC(Normalized Generalized Cross-Correlation) 모델 사용
        params['ngcc_channels'] = 32  # NGCC 입력 채널 수
        params['ngcc_out_channels'] = 16  # NGCC 출력 채널 수
        params['saved_chunks'] = True  # 저장된 오디오 청크 사용
        params['use_mel'] = False  # 멜 스펙트로그램 비사용
        params['nb_epochs'] = 1  # 학습 에포크 수
        params['predict_tdoa'] = True  # TDOA 예측 활성화
        params['lambda'] = 1.0 # TDOA만 학습하도록 람다값을 1.0으로 설정 (0.0이면 SELD만 학습)
        params['max_tau'] = 6  # 최대 시간 지연값
        params['tracks'] = 3  # 동시에 추적할 소리 개수
        params['fixed_tdoa'] = False  # 고정된 TDOA 사용하지 않음
        params['batch_size'] = 32  # 배치 크기
        params['lr'] = 1e-4  # 학습률
        params['warmup'] = 0  # 워밍업 에포크 없음

    elif argv == '10': # fine-tuning from tdoa-pretrained model
        print("RAW AUDIO CHUNKS w/ NGCC model + multi ACCDOA, pre-trained TDOA features\n")
        params['finetune_mode'] = True
        params['raw_chunks'] = True
        params['pretrained_model_weights'] = 'models/9_tdoa-3tracks-16channels.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['n_mics'] = 4
        params['model'] = 'ngccmodel'
        params['ngcc_channels'] = 32
        params['ngcc_out_channels'] = 16
        params['saved_chunks'] = True
        params['use_mel'] = True

        params['predict_tdoa'] = False
        params['lambda'] = 0.0 # set to 1.0 to only train tdoa, and 0.0 to only train SELD
        params['max_tau'] = 6
        params['tracks'] = 3
        params['fixed_tdoa'] = True


    elif argv == '32':
        print("[CST-former: Divided Channel Attention] FOA + Multi-ACCDOA + CST_DCA + CMT (S dim : 16)\n")
        params['model'] = 'cstformer'
        params['quick_test'] = False
        params['multi_accdoa'] = True
        params['t_pooling_loc'] = 'front'

        params['FreqAtten'] = True
        params['ChAtten_DCA'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [2, 2, 1]
        params['t_pool_size'] = [params['feature_label_resolution'], 1, 1]
        params['fnn_size'] = 256

    elif argv == '33':
        print("[CST-former: Unfolded Local Embedding] GCC + Multi-ACCDOA + CST Unfold + CMT (S dim : 16)\n")
        params['model'] = 'cstformer'
        params['quick_test'] = False
        params['multi_accdoa'] = True
        params['t_pooling_loc'] = 'front'

        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [1,2,2] 
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 256
        params['nb_channels'] = 10

    elif argv == '34':
        print("[CST-former: Unfolded Local Embedding] SALSA-LITE + Multi-ACCDOA + CST Unfold + CMT (S dim : 16)\n")
        params['model'] = 'cstformer'
        params['use_salsalite'] = True
        params['quick_test'] = False
        params['multi_accdoa'] = True
        params['t_pooling_loc'] = 'front'

        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True

        params["f_pool_size"] = [1,4,6]
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 256

    elif argv == '333': #CST former with NGCC-PHAT
        print("CST-Former Large w/ NGCC model + multi ACCDOA")
        # CST-Former 모델과 NGCC 특징을 사용한 학습 설정
        # README에서 설명한 주요 실행 ID (train_seldnet.py 333 my_experiment 명령어)
        params = get_params()
        params['model'] = 'cstformer'  # CST-Former 모델 사용 (Channel-Spectral-Temporal Transformer)
        params['multi_accdoa'] = True  # 다중 ACCDOA 방식 활성화
        params['t_pooling_loc'] = 'front'  # 시간 풀링 위치 설정

        # CST-Former 모델 구성 설정
        params['FreqAtten'] = True  # 주파수 어텐션 활성화
        params['ChAtten_ULE'] = True  # Unfolded Local Embedding 채널 어텐션 활성화
        params['CMT_block'] = True  # CMT(Channel Mixing Transformer) 블록 활성화

        # 풀링 크기 및 레이어 설정
        params["f_pool_size"] = [1, 2, 2] # 주파수 풀링 사이즈 (Large 버전은 [1, 1, 1] 사용)
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]  # 시간 풀링 사이즈
        params['nb_fnn_layers'] = 1  # 피드포워드 신경망 레이어 수
        params['fnn_size'] = 256  # 피드포워드 신경망 크기

        # 모델 및 데이터 설정
        params['finetune_mode'] = True  # 미세 조정 모드 활성화
        params['raw_chunks'] = True  # 원본 오디오 청크 사용
        params['pretrained_model_weights'] = 'models/333_cst-3t16c.h5'  # 사전 학습된 모델 가중치 (레이턴시 측정용)
        params['dataset'] = 'mic'  # 마이크로폰 데이터셋 사용
        params['n_mics'] = 4  # 마이크 개수
        params['ngcc_channels'] = 32  # NGCC 입력 채널 수
        params['ngcc_out_channels'] = 16  # NGCC 출력 채널 수
        params['saved_chunks'] = True  # 저장된 오디오 청크 사용
        params['use_mel'] = True  # 멜 스펙트로그램 사용
        params['use_ngcc'] = True  # NGCC 특징 사용

        # TDOA 관련 설정
        params['predict_tdoa'] = False  # TDOA 예측 비활성화
        params['lambda'] = 0.0 # SELD만 학습하도록 람다값을 0.0으로 설정
        params['max_tau'] = 6  # 최대 시간 지연값
        params['tracks'] = 3  # 동시에 추적할 소리 개수
        params['fixed_tdoa'] = True  # 고정된 TDOA 사용

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    if params['dataset'] == 'mic':
        if params['use_ngcc']:
            if params['use_mel']:
                params['nb_channels'] = int(params['ngcc_out_channels'] * params['n_mics'] * (params['n_mics'] - 1) / 2 +  params['n_mics'])
            else:
                params['nb_channels'] = int(params['ngcc_out_channels'] * params['n_mics'] * ( 1 + (params['n_mics'] - 1) / 2))
        elif params['use_salsalite']:
            params['nb_channels'] = 7
    else:
        params['nb_channels'] = 7


    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2024' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif 'sim' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
