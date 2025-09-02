# eOCRfinetune: 부동산 계약서 OCR을 위한 EasyOCR 파인튜닝

이 프로젝트는 등기부등본, 임대차계약서 등 한국 부동산 관련 문서에 특화된 단어를 더 정확하게 인식하기 위해 EasyOCR 모델을 파인튜닝한 것입니다. 특정 도메인의 어휘에 대한 OCR 정확도를 높이는 것을 목표로 합니다.

파인튜닝에 사용된 베이스 모델은 `facebook/m2m100_1.2B` 입니다.

## 저장소 구조

```
eOCRfinetune/
├── data/                  # 학습 및 검증용 데이터셋
│   ├── train/
│   └── validation/
├── saved_models/          # 파인튜닝된 모델이 저장되는 폴더
├── eOCR_finetune.py       # 메인 파인튜닝 스크립트
├── requirements.txt       # 실행에 필요한 패키지 목록
├── ... (기타 데이터 처리용 .py 스크립트들)
└── README.md              # 프로젝트 설명 파일
```

## 주요 기능

-   **특화된 어휘 인식**: 등기부등본, 임대차계약서 등 법률 및 부동산 관련 문서에 자주 등장하는 용어, 이름, 숫자 인식에 최적화되었습니다.
-   **정확도 향상**: 기존 EasyOCR 모델 대비 특정 문서에서의 글자 인식률이 향상되었습니다.
-   **간편한 사용법**: 파인튜닝된 모델을 EasyOCR 프레임워크 내에서 쉽게 불러와 사용할 수 있습니다.

## 설치 방법

1.  **저장소 복제 (Clone)**
    ```bash
git clone https://github.com/pokqok/eOCRfinetune.git
cd eOCRfinetune
    ```

2.  **필요 패키지 설치**
    가상 환경(virtual environment) 사용을 권장합니다.
    ```bash
pip install -r requirements.txt
    ```

## 파인튜닝된 모델 사용법

EasyOCR의 `Reader` 클래스를 사용하여 파인튜닝된 모델을 간단하게 불러올 수 있습니다. 모델 파일(.pth, .yaml)이 저장된 폴더 경로를 지정하기만 하면 됩니다.

```python
import easyocr

# 파인튜닝된 모델 파일들이 들어있는 폴더 경로
model_path = './saved_models/ko_g2_new'

# 모델 불러오기
# gpu=True 또는 False로 설정하여 GPU 사용 여부를 정할 수 있습니다.
reader = easyocr.Reader(['ko'], model_storage_directory=model_path, gpu=True)

# 이미지 파일로 OCR 실행
image_path = '경로/입력할/이미지파일.png'
result = reader.readtext(image_path)

# 결과 출력
for (bbox, text, prob) in result:
    print(f'인식된 글자: "{text}", 신뢰도: {prob:.4f}')
```

## 직접 파인튜닝하는 방법

1.  **데이터셋 준비**
    학습 및 검증에 사용할 이미지와 라벨 파일(`labels.csv`)을 각각 `data/train`과 `data/validation` 폴더에 위치시킵니다. 데이터 구조는 파인튜닝 스크립트가 요구하는 형식을 따라야 합니다.

2.  **학습 스크립트 실행**
    `eOCR_finetune.py` 스크립트를 필요한 인자(argument)와 함께 실행합니다.

    ```bash
python eocr_finetune.py \
    --train_data ./data/train \
    --valid_data ./data/validation \
    --Transformation None \
    --FeatureExtraction VGG \
    --Prediction CTC \
    --SequenceModeling BiLSTM \
    --sensitive \
    --character "0123456789가강개객건게겨계고공과관광구규권근금기길김나남대더도동등라로록리마매명목물미및바박배백버번베보부북사산삼상서선설성세소수시신아안양어업에여역연영예오옥용우원월유윤을음의이인임입자작장재전정제조종주중지직차채천청초최추충치타토파판평포하학한함합항해행향현호화확황효후" \
    --saved_model ./saved_models/ko_g2_new/korean_g2.pth
    ```

### 주요 인자 설명

-   `--train_data`: 학습 데이터셋 경로
-   `--valid_data`: 검증 데이터셋 경로
-   `--character`: 모델이 인식해야 할 모든 글자 집합
-   `--saved_model`: 학습이 완료된 모델이 저장될 경로와 파일명

## 기타 유틸리티 스크립트

메인 파인튜닝 스크립트(`eOCR_finetune.py`) 외에 포함된 여러 `.py` 파일들은 파인튜닝 과정에서 데이터를 준비하고 정리하기 위해 사용된 보조 스크립트들입니다.

-   `make_label.py`: 이미지 파일 이름에 기반하여 학습에 필요한 `labels.csv` 파일을 생성합니다.
-   `rename_file.py`: 데이터 파일들의 이름을 일괄적으로 변경합니다.
-   `move_file.py`, `random_file.py`: 데이터셋 분리나 정리를 위해 파일을 이동시키거나 무작위로 섞는 작업을 수행합니다.
-   `remove_word.py`, `split_word.py`: 라벨 데이터에서 특정 단어를 제거하거나 분리하는 등 텍스트 전처리 작업을 합니다.

필요에 따라 이 스크립트들을 사용하여 자신만의 데이터셋을 구축하고 전처리할 수 있습니다.

## 감사의 말

이 프로젝트는 강력하고 유연한 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 라이브러리를 기반으로 제작되었습니다.
