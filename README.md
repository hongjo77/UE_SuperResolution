# UE_SuperResolution

파이썬 머신러닝 프로젝트  
언리얼 엔진5를 이용한 이미지 Super Resolution

## Files

- **Model.ipynb**: 메인 코드
- **LR/**: 저해상도 이미지 폴더
- **HR/**: 고해상도 이미지 폴더
- **model/**: Model pth 저장소
- **config.json**: train.py args 저장
- **data_loaders.pth**: 데이터셋 경로 및 정보 저장
- **dataset_module.py**: 데이터셋 정의
- **model_class_model_cnn6.py**: 모델 정의
- **train.py**: 학습 코드

## Dataset

- **데이터셋 출처**: [Generate UE Stylized Data](https://gitlab.informatik.uni-wuerzburg.de/Brandner/generate_ue_stylized_data)  
  본 프로젝트에서 사용된 데이터셋은 위 링크의 언리얼 프로젝트 맵에서 추출됩니다.

- **데이터셋 구성**:
  - **LR/**: 저해상도 이미지
  - **HR/**: 고해상도 이미지
