# verify_final.py

import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

print("[1/3] 경로 설정 및 데이터 로드를 시작합니다...")
try:
    BASE_DIR = os.getcwd()
    ALL_DATA_DIR = os.path.join(BASE_DIR, "all_data_pruned")
    IMAGE_DIR = os.path.join(ALL_DATA_DIR, "images")
    LABEL_FILE = os.path.join(ALL_DATA_DIR, "labels.txt")
    FONT_PATH = 'c:/Windows/Fonts/malgun.ttf'

    if not os.path.exists(LABEL_FILE) or not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError("all_data/images 폴더 또는 all_data/labels.txt 파일이 없습니다.")

    # 1. 실제 폴더에 있는 이미지 파일 이름 목록 (Set으로 만들어 검색 속도 향상)
    image_files_on_disk = set(os.listdir(IMAGE_DIR))

    # 2. 라벨 파일에 있는 파일 이름 목록
    labels_df = pd.read_csv(LABEL_FILE, sep='\\t', header=None, engine='python', names=['filepath', 'text'])
    # 상대 경로에서 파일 이름만 추출하여 Set으로 만듦
    filenames_in_label = set(labels_df['filepath'].apply(lambda x: os.path.basename(x)))
    
    print("✅ 데이터 로드 완료.")

except Exception as e:
    print(f"❌ 데이터 로드 중 오류: {e}"); sys.exit()


print("\n[2/3] 데이터 정합성 교차 검증을 시작합니다...")
try:
    # 검증 1: 라벨은 있지만 실제 이미지가 없는 경우 (Orphaned Labels)
    orphaned_labels = filenames_in_label - image_files_on_disk
    
    # 검증 2: 이미지는 있지만 라벨이 없는 경우 (Orphaned Images)
    orphaned_images = image_files_on_disk - filenames_in_label
    
    print("\n" + "="*50)
    print("                 데이터 정합성 검증 결과")
    print("="*50)
    print(f"  - 실제 이미지 파일 총 개수        : {len(image_files_on_disk):,} 개")
    print(f"  - 라벨 파일의 총 라인 수          : {len(labels_df):,} 개")
    print("-" * 50)
    print(f"  - 라벨에 있지만 이미지가 없는 경우 : {len(orphaned_labels)} 건")
    print(f"  - 이미지에 있지만 라벨이 없는 경우 : {len(orphaned_images)} 건")
    print("="*50)

    if len(orphaned_labels) == 0 and len(orphaned_images) == 0:
        print("✅ 완벽합니다! 모든 이미지와 라벨이 1:1로 정확하게 일치합니다.")
    else:
        print("❗️ 데이터 불일치가 발견되었습니다. 전처리 과정을 다시 확인해야 할 수 있습니다.")
        sys.exit()

except Exception as e:
    print(f"❌ 교차 검증 중 오류: {e}"); sys.exit()


print("\n[3/3] 검증된 데이터에서 무작위 샘플 5개를 시각화합니다...")
try:
    # 짝이 맞는 전체 데이터프레임에서 5개 샘플 추출
    matching_df = labels_df[labels_df['filepath'].apply(lambda x: os.path.basename(x)).isin(image_files_on_disk)]
    sample_df = matching_df.sample(n=min(len(matching_df), 5), random_state=42)

    processed_images = []
    for _, row in sample_df.iterrows():
        img_path = os.path.join(ALL_DATA_DIR, row['filepath'])
        text_label = row['text']
        
        cropped_image = Image.open(img_path)
        
        # 이미지와 텍스트를 합치는 로직
        font = ImageFont.truetype(FONT_PATH, size=20)
        canvas_width = max(cropped_image.width, int(len(str(text_label)) * 15))
        canvas = Image.new('RGB', (canvas_width, cropped_image.height + 30), 'white')
        canvas.paste(cropped_image, ((canvas_width - cropped_image.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, cropped_image.height + 5), str(text_label), font=font, fill='black')
        processed_images.append(canvas)

    if processed_images:
        fig, axes = plt.subplots(len(processed_images), 1, figsize=(15, 3 * len(processed_images)))
        if len(processed_images) == 1: axes = [axes]
        fig.suptitle("최종 데이터 샘플 시각적 확인", fontsize=16)
        for i, img in enumerate(processed_images):
            axes[i].imshow(img)
            axes[i].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("시각화할 샘플이 없습니다.")
        
except Exception as e:
    print(f"❌ 시각화 중 오류: {e}")

print("\n🎉 모든 확인 과정이 끝났습니다.")