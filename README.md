## 0. Environment Setup

### 0.1 Conda 환경 생성
새로운 Conda 환경을 생성하고 활성화합니다:
```bash
# Python 3.8-3.10 버전이 호환됩니다. 가장 안정적인 버전은 3.9입니다.
conda create -n seld_env python=3.9
conda activate seld_env
```

### 0.2 필요한 패키지 설치
requirements.txt 파일을 사용하여 필요한 모든 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

requirements.txt에는 다음과 같은 주요 패키지가 포함되어 있습니다:
- numpy, scipy, matplotlib: 기본 과학 계산 라이브러리
- torch, torchaudio, torchvision: PyTorch 딥러닝 프레임워크
- librosa, soundfile: 오디오 처리 라이브러리
- speechbrain: 음성 처리 라이브러리
- einops: 텐서 변환 유틸리티 (CST-Former 모델에 필요)
- opencv-python: 컴퓨터 비전 라이브러리

### 0.3 CUDA 설정 
CUDA 11.8 또는 12.1이 권장됩니다.

## 1. Datasets download & Directory setting

데이터는 다음 링크에서 다운 받으실 수 있습니다.: [**Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23)**](https://doi.org/10.5281/zenodo.7709052) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7709052.svg)](https://doi.org/10.5281/zenodo.7709052)

데이터를 압축 해제한 후 (mic_dev, metadata_dev) "data_2024" 라는 이름으로 아래 폴더 구조와 동일하게 배치하세요:

```
SR_BERG
├── cst_former
├── logs
└── data_2024
    ├── metadata_dev
    ├── mic_dev
...
```


## 2. Feature Extraction

**NGCC + MS** 특징을 추출합니다:
```bash
python batch_feature_extraction.py 9
```
이 명령어는 원본 오디오에서 NGCC(Normalized Generalized Cross-Correlation) 특징을 추출하여 'data_2024/seld_feat_label' 폴더에 저장합니다.

## 3. SELD Training

**CST-Former 모델 학습하기 (NGCC+MS 특징 사용)**
```bash
python3 train_seldnet.py 333 my_experiment
```

이 프로젝트는 SELDnet 기반이 아닌 **CST-Former(Channel-Spectral-Temporal Transformer)** 모델을 사용합니다. 다음 매개변수를 조정할 수 있습니다:
- `parameters.py`에서 epoch 수와 learning rate 조정
- `parameters.py`의 CST-Former 관련 설정 조정 (argv=='333' 부분)

기본 설정:
- nb_epoch = 50
- learning rate = 1e-5
- CST-Former 구성: FreqAtten=True, ChAtten_ULE=True, CMT_block=True

**실험 결과 재현하기**
1. `parameters.py`에서 `nb_epochs`를 1로, `lr`을 0으로 변경
2. `parameters.py`에서 `params['pretrained_model_weights']`를 `models_audio/333_new_dist_dev_split0_multiaccdoa_mic_gcc_model.h5`로 변경 (argv==333 조건에서)

그리고 다음 명령어를 실행:
```bash
python3 train_seldnet.py 333 check_best
```

| Model | F<sub>20°</sub> | DOAE<sub>CD</sub> | RDE<sub>CD</sub> | weights |
| ----| --- | --- | --- | --- |
| new_dist | 30.2% | 22.64&deg; | 0.34 | `models_audio/333_new_dist_dev_split0_multiaccdoa_mic_gcc_model.h5` |

## 4. 레이턴시 측정

CST-Former 모델의 레이턴시만 측정하려면 다음 명령어를 실행하세요:
```bash
python3 train_seldnet.py 333 latency_check
```

이 명령어는 CST-Former 모델의 추론 속도만 측정하고 즉시 종료됩니다 (테스트 단계 없음).

| Model | latency (GPU) | weights |
| ---- | --- | --- |
| CST-Former Small w/ NGCC | 11ms (A100) | `models/333_cst-3t16c.h5` |
| CST-Former Large w/ NGCC | 11ms (A100) | `models/333_cst-3t16c-large.h5` |

## 5. 프로젝트 파일 구조 및 주요 파일 설명

마이크 데이터만을 이용한 DOA(Direction of Arrival) 측정을 위한 파일과 구조는 다음과 같습니다.

```
test_berg/
├── batch_feature_extraction.py  # 특징 추출 스크립트
├── train_seldnet.py             # 모델 학습 및 테스트 메인 스크립트
├── parameters.py                # 프로젝트 설정 및 하이퍼파라미터
├── cls_feature_class.py         # 특징 추출 클래스
├── cls_data_generator.py        # 데이터 생성 및 로딩 클래스
├── SELD_evaluation_metrics.py   # 평가 메트릭 계산
├── cls_compute_seld_results.py  # 결과 계산 클래스
├── cst_former/                  # CST-Former 모델 구현
│   ├── CST_former_model.py      # CST-Former 모델 클래스
│   └── CST_details/             # CST-Former 세부 구현 (인코더, 어텐션 등)
├── ngcc/                        # NGCC 모델 구현
│   ├── model.py                 # NGCC 모델 클래스
│   └── dnn_models.py            # SincNet 등 DNN 모델 구현
└── data_2024/                   # 데이터셋 폴더
    ├── metadata_dev/            # 메타데이터
    ├── mic_dev/                 # 마이크 오디오 파일
    └── seld_feat_label/         # 추출된 특징 및 레이블
```

### 5.1 주요 명령어에 사용되는 핵심 파일

4개의 주요 명령어에서 사용되는 핵심 파일들은 다음과 같습니다:

#### 1. 특징 추출 (`python batch_feature_extraction.py 9`)
- **batch_feature_extraction.py**: 특징 추출을 위한 메인 스크립트
- **parameters.py**: 작업 ID 9에 대한 파라미터 설정 (NGCC 특징 관련)
- **cls_feature_class.py**: 오디오 파일에서 특징을 추출하는 클래스
- **ngcc/model.py**: NGCC 특징 추출을 위한 모델 구현

#### 2. 모델 학습 (`python3 train_seldnet.py 333 my_experiment`)
- **train_seldnet.py**: 모델 학습 메인 스크립트
- **parameters.py**: 작업 ID 333에 대한 파라미터 설정 (CST-Former 모델 관련)
- **cls_data_generator.py**: 학습 데이터 생성 및 로딩
- **cst_former/CST_former_model.py**: CST-Former 모델 구현
- **ngcc/model.py**: 학습 중 특징 추출에 사용

#### 3. 사전 학습된 모델 평가 (`python3 train_seldnet.py 333 check_best`)
- **train_seldnet.py**: 모델 평가 스크립트
- **parameters.py**: 작업 ID 333에 대한 파라미터 설정 (사전 학습된 모델 가중치 로드)
- **cls_compute_seld_results.py**: 결과 계산 및 평가
- **SELD_evaluation_metrics.py**: 평가 메트릭 계산

#### 4. 레이턴시 측정 (`python3 train_seldnet.py 333 latency_check`)
- **train_seldnet.py**: 레이턴시 측정 코드 포함
- **parameters.py**: 작업 ID 333에 대한 파라미터 설정
- **cst_former/CST_former_model.py**: 레이턴시 측정 대상 모델

### 5.2 주요 파일 설명

#### 1. 핵심 스크립트 및 설정 파일
- **batch_feature_extraction.py**: 오디오 데이터에서 NGCC 및 멜 스펙트로그램 특징을 추출하기 위한 스크립트
- **train_seldnet.py**: 모델 훈련, 테스트, 평가, 레이턴시 측정을 위한 메인 스크립트
- **parameters.py**: 다양한 작업 ID에 따른 모든 하이퍼파라미터 및 설정 정의

#### 2. 모델 구현 파일
- **cst_former/CST_former_model.py**: Channel-Spectral-Temporal Transformer 모델의 주요 클래스 구현
- **ngcc/model.py**: Neural Generalized Cross-Correlation 구현 및 특징 추출 파이프라인
- **ngcc/dnn_models.py**: SincNet 등 기본 DNN 모델 구현

#### 3. 데이터 처리 및 평가 파일
- **cls_feature_class.py**: 오디오 파일 처리 및 특징 추출 클래스
- **cls_data_generator.py**: 데이터셋 로딩 및 배치 생성
- **SELD_evaluation_metrics.py**: 음향 이벤트 위치 추정 평가 메트릭
- **cls_compute_seld_results.py**: 모델 출력 후처리 및 결과 계산

## 추가 정보
- 자세한 구현 세부 사항은 `train_seldnet.py`, `parameters.py`, `cst_former/CST_former_model.py` 파일을 참조하세요.
- CST-Former 모델은 `cst_former` 디렉토리에 구현되어 있습니다.
- NGCC 특징 추출 관련 코드는 `ngcc` 디렉토리에 있습니다.
