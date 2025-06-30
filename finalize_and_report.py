# finalize_and_report.py

import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

def main():
    print("[1/4] 경로를 설정하고 데이터를 로드합니다...")
    try:
        BASE_DIR = os.getcwd()
        ALL_DATA_DIR = os.path.join(BASE_DIR, "all_data")
        IMAGE_DIR = os.path.join(ALL_DATA_DIR, "images")
        LABEL_FILE = os.path.join(ALL_DATA_DIR, "labels.txt")
        FONT_PATH = 'c:/Windows/Fonts/malgun.ttf'

        if not os.path.exists(LABEL_FILE) or not os.path.exists(IMAGE_DIR):
            raise FileNotFoundError("all_data 폴더에 images 또는 labels.txt가 없습니다.")

        # 1. 실제 디스크에 있는 이미지 파일 목록
        image_files_on_disk = set(os.listdir(IMAGE_DIR))
        # 2. 라벨 파일에 기록된 파일명 목록
        labels_df = pd.read_csv(LABEL_FILE, sep='\\t', header=None, engine='python', names=['filepath', 'text'])
        labels_df['filename'] = labels_df['filepath'].apply(lambda x: os.path.basename(x))
        filenames_in_label = set(labels_df['filename'])
        
        print("✅ 데이터 로드 완료.")
        print(f"  - 정리 전 이미지 파일 수: {len(image_files_on_disk):,} 개")
        print(f"  - 정리 전 라벨 수: {len(labels_df):,} 개")

    except Exception as e:
        print(f"❌ 데이터 로드 중 오류: {e}"); sys.exit()


    print("\n[2/4] 데이터 정합성 교차 검증 및 불일치 데이터 삭제를 시작합니다...")
    try:
        # 검증 1: 이미지는 있지만 라벨이 없는 경우 (고아 이미지)
        orphaned_images = image_files_on_disk - filenames_in_label
        if orphaned_images:
            print(f"  - {len(orphaned_images)}개의 라벨 없는 이미지를 찾아 삭제합니다...")
            for filename in tqdm(orphaned_images, desc="고아 이미지 삭제 중"):
                os.remove(os.path.join(IMAGE_DIR, filename))
        
        # 검증 2: 라벨은 있지만 실제 이미지가 없는 경우 (고아 라벨)
        # 실제 디스크에 있는 이미지 파일 목록을 기준으로 DataFrame을 필터링합니다.
        cleaned_df = labels_df[labels_df['filename'].isin(image_files_on_disk)].copy()
        
        if len(cleaned_df) < len(labels_df):
            print(f"  - {len(labels_df) - len(cleaned_df)}개의 이미지 없는 라벨을 찾아 제거합니다...")
        
        # 정리된 데이터프레임을 새로운 라벨 파일로 덮어씁니다.
        cleaned_df[['filepath', 'text']].to_csv(LABEL_FILE, sep='\t', header=False, index=False, encoding='utf-8')
        print("✅ 데이터 정리 완료!")

    except Exception as e:
        print(f"❌ 데이터 정리 중 오류: {e}"); sys.exit()


    print("\n[3/4] 최종 데이터셋 구성 비율을 분석합니다...")
    try:
        final_df = cleaned_df
        counts = {
            'form (인쇄체)': final_df['filename'].str.startswith('printed_').sum(),
            '글자 (필기체)': final_df['filename'].str.startswith('hw_char_').sum(),
            '단어 (필기체)': final_df['filename'].str.startswith('hw_word_').sum(),
            'qmnist (숫자)': final_df['filename'].str.startswith('qmnist_').sum()
        }
        total_count = len(final_df)

        print("\n" + "="*60)
        print("                 🎉 최종 데이터셋 구성 결과 🎉")
        print("="*60)
        print(f"  - 총 데이터 개수: {total_count:,} 개")
        print("-" * 60)
        for name, count in counts.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            print(f"  - {name:<15}: {count:>,} 개 ({percentage:.1f}%)")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 구성 분석 중 오류: {e}")


    print("\n[4/4] 최종 데이터에서 무작위 샘플 5개를 시각화합니다...")
    try:
        sample_df = final_df.sample(n=min(len(final_df), 5))
        processed_images = []
        for _, row in sample_df.iterrows():
            img_path = os.path.join(IMAGE_DIR, row['filename'])
            text_label = row['text']
            
            if not os.path.exists(img_path): continue
            
            cropped_image = Image.open(img_path)
            font = ImageFont.truetype(FONT_PATH, size=20)
            canvas = Image.new('RGB', (cropped_image.width + 20, cropped_image.height + 30), 'white')
            canvas.paste(cropped_image, (10, 5))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, cropped_image.height + 10), str(text_label), font=font, fill='black')
            processed_images.append(canvas)

        if processed_images:
            fig, axes = plt.subplots(1, len(processed_images), figsize=(20, 5))
            if len(processed_images) == 1: axes = [axes]
            for i, img in enumerate(processed_images):
                axes[i].imshow(img)
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"❌ 시각화 중 오류: {e}")

if __name__ == '__main__':
    main()