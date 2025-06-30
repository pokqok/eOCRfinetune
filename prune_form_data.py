# prune_form_data.py (파일명 교차 검증 포함 최종 버전)

import os
import shutil
import pandas as pd
from tqdm import tqdm
import sys

# --- 1. 설정 및 경로 정의 ---
print("[1/5] 경로를 설정합니다...")
try:
    TARGET_COUNT = 80000
    BASE_DIR = os.getcwd()
    ALL_DATA_DIR = os.path.join(BASE_DIR, "all_data")
    ORIGINAL_IMAGE_DIR = os.path.join(ALL_DATA_DIR, "images")
    ORIGINAL_LABEL_FILE = os.path.join(ALL_DATA_DIR, "labels.txt")
    PRUNED_DATA_DIR = os.path.join(BASE_DIR, "all_data_pruned")
    PRUNED_IMAGE_DIR = os.path.join(PRUNED_DATA_DIR, "images")
    PRUNED_LABEL_FILE = os.path.join(PRUNED_DATA_DIR, "labels.txt")

    os.makedirs(PRUNED_IMAGE_DIR, exist_ok=True)
except Exception as e:
    print(f"❌ 경로 설정 중 오류: {e}"); sys.exit()

# --- 2. 사전 확인 ---
print("\n[2/5] 정리 전 현재 상태를 확인합니다...")
try:
    if not os.path.exists(ORIGINAL_LABEL_FILE):
        raise FileNotFoundError(f"오류: 원본 라벨 파일({ORIGINAL_LABEL_FILE})이 없습니다.")
    
    original_df = pd.read_csv(ORIGINAL_LABEL_FILE, sep='\\t', header=None, engine='python', names=['filename', 'text'])
    print(f"✅ 현재 라벨 파일에 {len(original_df):,}개의 항목이 있습니다.")
except Exception as e:
    print(f"❌ 원본 라벨 파일 읽기 오류: {e}"); sys.exit()
    
# --- 3. 샘플링 및 파일 이동 ---
print(f"\n[3/5] {len(original_df):,}개 중 {TARGET_COUNT:,}개를 무작위로 샘플링하여 이동합니다...")
try:
    num_to_sample = min(len(original_df), TARGET_COUNT)
    sampled_df = original_df.sample(n=num_to_sample, random_state=42)

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="이미지 이동 중"):
        src_path = os.path.join(ALL_DATA_DIR, row['filename'])
        dest_path = os.path.join(PRUNED_IMAGE_DIR, os.path.basename(row['filename']))
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
    print("✅ 샘플 이미지 이동 완료.")
except Exception as e:
    print(f"❌ 샘플링 및 이동 중 오류: {e}"); sys.exit()

# --- 4. 새 라벨 파일 생성 및 폴더 교체 ---
print("\n[4/5] 새 라벨 파일을 생성하고 폴더를 교체합니다...")
try:
    sampled_df.to_csv(PRUNED_LABEL_FILE, sep='\t', header=False, index=False, encoding='utf-8')
    print("  - 새 라벨 파일 저장 완료.")
    print(f"  - 기존 '{ALL_DATA_DIR}' 폴더를 삭제합니다. (시간이 걸릴 수 있습니다)")
    shutil.rmtree(ALL_DATA_DIR)
    print("  - 기존 폴더 삭제 완료.")
    print(f"  - '{PRUNED_DATA_DIR}' 폴더의 이름을 '{ALL_DATA_DIR}'로 변경합니다.")
    os.rename(PRUNED_DATA_DIR, ALL_DATA_DIR)
    print("✅ 폴더 교체 완료!")
except Exception as e:
    print(f"❌ 라벨 생성 및 폴더 교체 중 오류: {e}"); sys.exit()

# --- 5. [수정됨] 최종 검증 (파일명 일치 여부 포함) ---
print("\n[5/5] 최종 검증 (파일명 일치 여부 포함)을 시작합니다...")
try:
    # 5.1 실제 폴더에 있는 이미지 파일 이름 목록을 모두 불러옵니다.
    final_image_dir = os.path.join(BASE_DIR, "all_data", "images")
    image_files_on_disk = set(os.listdir(final_image_dir))
    final_image_count = len(image_files_on_disk)

    # 5.2 라벨 파일을 읽고, 각 라벨에 해당하는 이미지가 실제로 존재하는지 확인합니다.
    final_label_file = os.path.join(BASE_DIR, "all_data", "labels.txt")
    with open(final_label_file, 'r', encoding='utf-8') as f:
        label_entries = f.read().splitlines()
    final_label_count = len(label_entries)

    missing_files_count = 0
    for entry in tqdm(label_entries, desc="라벨-이미지 매칭 검증"):
        # 라벨 파일의 첫 번째 열(상대 경로)에서 파일 이름만 추출
        filename_in_label = os.path.basename(entry.split('\t')[0])
        # 실제 파일 목록(Set)에 해당 이름이 있는지 확인 (Set을 사용하면 매우 빠름)
        if filename_in_label not in image_files_on_disk:
            missing_files_count += 1

    print("\n" + "="*60)
    print("             최종 정리 및 정합성 검증 완료")
    print("="*60)
    print(f"  - `images` 폴더의 최종 파일 수 : {final_image_count:,} 개")
    print(f"  - `labels.txt`의 최종 라인 수  : {final_label_count:,} 개")
    print(f"  - [교차 검증] 라벨에 있지만 실제 파일이 없는 경우: {missing_files_count} 건")
    
    if final_image_count == final_label_count == TARGET_COUNT and missing_files_count == 0:
        print("\n  - ✅ 완벽합니다! 이미지와 라벨의 개수가 일치하며, 모든 라벨의 이미지 파일이 존재합니다.")
    else:
        print("\n  - ❌ 실패: 최종 데이터에 불일치가 발견되었습니다. 확인이 필요합니다.")
    print("="*60)

except Exception as e:
    print(f"❌ 최종 검증 중 오류: {e}")