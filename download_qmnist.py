# download_qmnist.py

import os
import torch
import torchvision
from torchvision.datasets import QMNIST # MNIST 대신 QMNIST를 import
from torch.utils.data import ConcatDataset
from PIL import Image
from tqdm import tqdm
import pandas as pd
import sys

def get_script_path():
    try: return os.path.dirname(os.path.abspath(__file__))
    except NameError: return os.getcwd()

# --- 1. 경로 설정 ---
print("[1/4] 기본 경로를 설정합니다...")
try:
    BASE_DIR = get_script_path()
    SOURCE_DATA_ROOT = os.path.join(BASE_DIR, "다양한 형태의 한글 문자 OCR")
    
    # QMNIST 결과물이 저장될 위치
    QMNIST_IMAGE_DIR = os.path.join(SOURCE_DATA_ROOT, "qmnist_images")
    QMNIST_LABEL_PATH = os.path.join(SOURCE_DATA_ROOT, "qmnist_labels.csv")
    
    QMNIST_DOWNLOAD_TEMP_DIR = os.path.join(BASE_DIR, "qmnist_temp_data")

    os.makedirs(QMNIST_IMAGE_DIR, exist_ok=True)
    os.makedirs(QMNIST_DOWNLOAD_TEMP_DIR, exist_ok=True)
    print("✅ 경로 설정 완료.")
    print(f" > QMNIST 이미지가 저장될 경로: {QMNIST_IMAGE_DIR}")

except Exception as e:
    print(f"❌ 경로 설정 중 오류 발생: {e}"); sys.exit()

# --- 2. torchvision으로 QMNIST 데이터셋 다운로드 ---
print("\n[2/4] torchvision을 통해 QMNIST 데이터셋을 다운로드합니다...")
try:
    # QMNIST는 what 인자로 'train', 'test', 'test10k', 'test50k', 'nist' 등을 선택할 수 있습니다.
    # 'train'과 'test'를 합쳐 최대한 많은 데이터를 확보합니다.
    train_dataset = QMNIST(root=QMNIST_DOWNLOAD_TEMP_DIR, what='train', download=True)
    test_dataset = QMNIST(root=QMNIST_DOWNLOAD_TEMP_DIR, what='test', download=True)

    full_dataset = ConcatDataset([train_dataset, test_dataset])
    print(f"✅ 다운로드 완료! 총 {len(full_dataset):,}개의 QMNIST 데이터를 준비합니다.")
except Exception as e:
    print(f"❌ QMNIST 다운로드 중 오류 발생: {e}"); sys.exit()

# --- 3. 모든 데이터를 PNG 이미지와 라벨 파일로 저장 ---
print(f"\n[3/4] {len(full_dataset):,}개의 이미지를 PNG 파일로 저장합니다... (시간이 다소 소요될 수 있습니다)")
label_data = []
for i, (image_tensor, label) in enumerate(tqdm(full_dataset, desc="이미지 저장 중")):
    
    # QMNIST는 텐서가 아닌 PIL 이미지로 바로 제공될 수 있으므로 확인
    if not isinstance(image_tensor, Image.Image):
         image_pil = torchvision.transforms.ToPILImage()(image_tensor)
    else:
         image_pil = image_tensor
    
    # 라벨이 텐서 형태일 수 있으므로 Python 숫자로 변환
    label_item = label.item() if hasattr(label, 'item') else label
    
    filename = f"qmnist_{i:06d}_{label_item}.png"
    file_path = os.path.join(QMNIST_IMAGE_DIR, filename)
    image_pil.save(file_path)
    
    label_data.append([filename, label_item])

print("✅ 이미지 파일 저장 완료!")


# --- 4. 라벨 CSV 파일 생성 ---
print(f"\n[4/4] 라벨 파일 '{os.path.basename(QMNIST_LABEL_PATH)}'을 생성합니다...")
try:
    df = pd.DataFrame(label_data, columns=['filename', 'text'])
    df.to_csv(QMNIST_LABEL_PATH, index=False, encoding='utf-8')
    print(f"✅ 라벨 파일 생성 완료! 총 {len(df):,} 개")
except Exception as e:
    print(f"❌ 라벨 파일 생성 중 오류 발생: {e}")

print("\n" + "="*50)
print("🎉 QMNIST 데이터 준비가 모두 완료되었습니다! 🎉")
print("="*50)