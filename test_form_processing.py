# test_form_creation.py (자동 정리 기능 제외 버전)

import os
import glob
import json
import random
from PIL import Image
import shutil
import sys

# --- 1. 경로 설정 및 테스트 환경 준비 ---
print("[1/3] 테스트 환경을 준비합니다...")
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    SOURCE_DATA_DIR = os.path.join(BASE_DIR, "다양한 형태의 한글 문자 OCR")
    ALL_DATA_DIR = os.path.join(BASE_DIR, "all_data")
    CROP_IMAGE_DIR = os.path.join(ALL_DATA_DIR, "images")
    LABEL_FILE_PATH = os.path.join(ALL_DATA_DIR, "labels.txt")

    os.makedirs(CROP_IMAGE_DIR, exist_ok=True)
    
    # 기존 라벨 파일이 있다면 덮어쓰기 경고
    if os.path.exists(LABEL_FILE_PATH):
        print(f"⚠️ 경고: 기존 '{LABEL_FILE_PATH}' 파일이 존재합니다. 이번 테스트로 덮어쓰여집니다.")
        # 안전을 위해 기존 파일을 삭제하지는 않습니다.

    if not os.path.exists(SOURCE_DATA_DIR):
        raise FileNotFoundError(f"소스 데이터 폴더를 찾을 수 없습니다: {SOURCE_DATA_DIR}")
        
    print("✅ 테스트 환경 준비 완료.")

except Exception as e:
    print(f"❌ 환경 준비 중 오류 발생: {e}")
    sys.exit()


# --- 2. 'form' 유형의 JSON 파일 1개 무작위 선택 ---
print("\n[2/3] 'form' 유형의 테스트용 JSON 파일을 찾습니다...")
try:
    form_json_paths = glob.glob(os.path.join(SOURCE_DATA_DIR, '**', 'form', '**', '*.json'), recursive=True)
    if not form_json_paths:
        raise FileNotFoundError("'form' 유형의 JSON 파일을 찾을 수 없습니다.")
    
    json_path = random.choice(form_json_paths)
    print(f"✅ 테스트 대상 선택 완료: {os.path.relpath(json_path, BASE_DIR)}")
except Exception as e:
    print(f"❌ 테스트 파일 선택 중 오류 발생: {e}")
    sys.exit()

# --- 3. 샘플 생성 및 파일 저장 ---
print("\n[3/3] 샘플 데이터 생성 및 파일 저장을 시작합니다...")
try:
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)

    image_filename = data['image'].get('file_name')
    word_annotations = data['text'].get('word', [])
    if not image_filename or not word_annotations:
        raise ValueError("JSON 파일에 이미지 또는 주석 정보가 없습니다.")

    image_folder_path = os.path.dirname(json_path).replace('[라벨]', '[원천]')
    src_image_path = os.path.join(image_folder_path, image_filename)
    if not os.path.exists(src_image_path):
        raise FileNotFoundError(f"짝이 되는 이미지 파일을 찾을 수 없습니다: {src_image_path}")

    large_image = Image.open(src_image_path)
    
    sample_annotations = random.sample(word_annotations, min(len(word_annotations), 5))
    
    labels_to_write = []
    for anno in sample_annotations:
        text_label = anno.get('value', '').strip()
        bbox = anno.get('wordbox')
        if not text_label or not bbox or len(bbox) != 4: continue
        
        crop_box = tuple(bbox)
        cropped_image = large_image.crop(crop_box)
        
        new_filename = f"test_printed_{os.path.splitext(os.path.basename(json_path))[0]}_{random.randint(1000,9999)}.png"
        dest_image_path = os.path.join(CROP_IMAGE_DIR, new_filename)
        cropped_image.save(dest_image_path)
        
        relative_path = os.path.join('images', new_filename).replace('\\', '/')
        labels_to_write.append(f"{relative_path}\t{text_label}")

    if labels_to_write:
        # 'w'(쓰기) 모드로 열어 항상 새로운 테스트 라벨 파일을 만듭니다.
        with open(LABEL_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(labels_to_write))
        print(f"✅ {len(labels_to_write)}개의 샘플 데이터를 생성했습니다.")
        print(f" > 이미지 저장 폴더: {CROP_IMAGE_DIR}")
        print(f" > 라벨 파일: {LABEL_FILE_PATH}")
    else:
        print("처리할 유효한 샘플이 없습니다.")

except Exception as e:
    print(f"❌ 처리 중 오류 발생: {e}")


# --- 4. [주석 처리됨] 테스트 환경 정리 ---
# finally:
#     print("\n[4/4] 테스트 환경을 정리합니다...")
#     # 생성했던 테스트 이미지 파일 삭제
#     for f in created_files:
#         if os.path.exists(f):
#             os.remove(f)
#     print(f" > {len(created_files)}개의 테스트 이미지 파일을 삭제했습니다.")
    
#     # 테스트용 라벨 파일 삭제
#     if os.path.exists(LABEL_FILE_PATH):
#         os.remove(LABEL_FILE_PATH)
#         print(" > 테스트용 labels.txt 파일을 삭제했습니다.")

#     # 백업했던 원본 라벨 파일 복원
#     if os.path.exists(LABEL_FILE_PATH + ".bak"):
#         os.rename(LABEL_FILE_PATH + ".bak", LABEL_FILE_PATH)
#         print(" > 기존 labels.txt 파일을 복원했습니다.")
    
#     # 만약 images 폴더가 비었다면 삭제
#     if os.path.exists(CROP_IMAGE_DIR) and not os.listdir(CROP_IMAGE_DIR):
#         os.rmdir(CROP_IMAGE_DIR)

print("\n🎉 테스트 파일 생성이 완료되었습니다. all_data 폴더에서 결과물을 확인해주세요.")