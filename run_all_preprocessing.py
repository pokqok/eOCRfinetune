# run_all_preprocessing.py (변수 범위 오류 수정 최종본)

import os
import glob
import json
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import sys
import random

# --- [사용자 설정] 최종 데이터 레시피 ---
DATA_RECIPE = {
    '글자': 80000,
    '단어': 120000,
    'qmnist': 120000
}

def verify_step(step_name, image_dir, label_file, image_prefix):
    """중간 검증 함수"""
    print("\n" + "-"*20 + f" [{step_name}] 중간 검증 " + "-"*20)
    try:
        image_count = 0
        if os.path.exists(image_dir):
            image_count = len([f for f in os.listdir(image_dir) if f.startswith(image_prefix)])
        print(f"  - 생성된 '{image_prefix}...' 이미지 파일 수: {image_count:,} 개")

        label_count = 0
        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if os.path.basename(line.strip().split('\t')[0]).startswith(image_prefix):
                        label_count += 1
        print(f"  - 추가된 '{image_prefix}...' 라벨 수: {label_count:,} 개")
        
        if image_count > 0 and label_count > 0:
            print("  - ✅ 검증 결과: 성공적으로 처리된 것으로 보입니다.")
        else:
            print("  - ⚠️  경고: 해당 유형의 파일이 생성되지 않았거나 라벨이 기록되지 않았습니다.")
    except Exception as e:
        print(f"  - ❌ 검증 중 오류 발생: {e}")
    print("-"*(42 + len(step_name)))


def process_data(data_type, target_count, paths):
    """지정된 유형의 데이터만 전처리하는 범용 함수"""
    
    print(f"\n{'='*15} {data_type} 데이터 처리 시작 (목표: {target_count:,}개) {'='*15}")
    
    source_items = []
    # 소스 목록 생성
    if data_type == 'qmnist':
        mnist_label_file = os.path.join(paths['AIHUB_SOURCE_DIR'], "qmnist_labels.csv")
        if os.path.exists(mnist_label_file):
            source_items = pd.read_csv(mnist_label_file).to_dict('records')
    else: # AI Hub 데이터
        source_items = glob.glob(os.path.join(paths['AIHUB_SOURCE_DIR'], '**', f'*{data_type}*/**/', '*.json'), recursive=True)
    
    if not source_items:
        print(f"'{data_type}' 유형의 데이터를 찾을 수 없어 건너뜁니다.")
        return

    # 처리 안 된 항목 필터링 및 샘플링
    unprocessed_items = []
    for item in source_items:
        item_id = f"qmnist_{item['filename']}" if data_type == 'qmnist' else os.path.relpath(item, paths['AIHUB_SOURCE_DIR'])
        if item_id not in paths['completed_items']:
            unprocessed_items.append(item)
    
    items_to_process = random.sample(unprocessed_items, min(len(unprocessed_items), target_count))
    print(f"  - 처리 대상 {len(items_to_process):,}개를 선택하여 진행합니다.")

    # 메인 루프
    for item in tqdm(items_to_process, desc=f"{data_type} 처리 중"):
        try:
            item_id = "" # item_id 초기화
            if data_type == 'qmnist':
                filename, text = item['filename'], item['text']
                item_id = f"qmnist_{filename}"
                src_path = os.path.join(paths['AIHUB_SOURCE_DIR'], "qmnist_images", filename)
                new_name = f"qmnist_{filename}"
                dest_path = os.path.join(paths['CROP_IMAGE_DIR'], new_name)
                if not os.path.exists(src_path): continue
                
                shutil.move(src_path, dest_path)
                relative_path = os.path.join('images', os.path.basename(dest_path)).replace('\\', '/')
                with open(paths['LABEL_FILE_PATH'], 'a', encoding='utf-8') as f: f.write(f"{relative_path}\t{str(text)}\n")

            else: # AI Hub
                json_path = item
                item_id = os.path.relpath(json_path, paths['AIHUB_SOURCE_DIR'])
                with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
                
                image_filename = data.get('image', {}).get('file_name')
                if not image_filename: continue
                image_folder_path = os.path.dirname(json_path).replace('[라벨]', '[원천]')
                src_path = os.path.join(image_folder_path, image_filename)
                if not os.path.exists(src_path): continue
                
                annotations = data.get('text', {}).get('word', [])
                if 'letter' in data['text']: annotations = [data['text']['letter']]
                if not annotations: continue
                
                label = "".join([c.get('value', '') for c in annotations]) if data_type == '단어' else annotations[0].get('value', '')
                if not label.strip(): continue
                
                prefix = "hw_word_" if data_type == '단어' else "hw_char_"
                new_name = f"{prefix}{os.path.splitext(os.path.basename(json_path))[0]}.png"
                dest_path = os.path.join(paths['CROP_IMAGE_DIR'], new_name)
                
                shutil.move(src_path, dest_path)
                
                relative_path = os.path.join('images', new_name).replace('\\', '/')
                with open(paths['LABEL_FILE_PATH'], 'a', encoding='utf-8') as f: f.write(f"{relative_path}\t{label.strip()}\n")
            
            with open(paths['LOG_FILE_PATH'], 'a', encoding='utf-8') as f: f.write(item_id + '\n')
            paths['completed_items'].add(item_id)
        except Exception as e:
            print(f"\n오류: {item} 처리 중 문제: {e}", file=sys.stderr)
            
    # 단계별 검증
    prefix_map = {'글자': 'hw_char_', '단어': 'hw_word_', 'qmnist': 'qmnist_'}
    if data_type in prefix_map:
        verify_step(data_type, paths['CROP_IMAGE_DIR'], paths['LABEL_FILE_PATH'], prefix_map[data_type])


def main():
    # --- 경로 설정 ---
    print("[1/3] 기본 경로를 설정합니다...")
    try:
        paths = {}
        paths['BASE_DIR'] = os.getcwd()
        paths['AIHUB_SOURCE_DIR'] = os.path.join(paths['BASE_DIR'], "다양한 형태의 한글 문자 OCR")
        paths['ALL_DATA_DIR'] = os.path.join(paths['BASE_DIR'], "all_data")
        paths['CROP_IMAGE_DIR'] = os.path.join(paths['ALL_DATA_DIR'], "images")
        paths['LABEL_FILE_PATH'] = os.path.join(paths['ALL_DATA_DIR'], "labels.txt")
        paths['LOG_FILE_PATH'] = os.path.join(paths['BASE_DIR'], "final_preprocessing_log.txt")
        
        os.makedirs(paths['CROP_IMAGE_DIR'], exist_ok=True)
        if not os.path.exists(paths['AIHUB_SOURCE_DIR']):
            raise FileNotFoundError(f"소스 폴더 없음: {paths['AIHUB_SOURCE_DIR']}")
        print("✅ 경로 설정 완료.")
    except Exception as e:
        print(f"❌ 경로 설정 중 오류: {e}"); sys.exit()

    # --- 로그 준비 ---
    print("\n[2/3] 작업 로그를 준비합니다...")
    try:
        with open(paths['LOG_FILE_PATH'], 'r', encoding='utf-8') as f:
            paths['completed_items'] = set(f.read().splitlines())
    except FileNotFoundError:
        paths['completed_items'] = set()
    print(f"✅ 총 {len(paths['completed_items'])}개 항목이 이전에 처리되었습니다.")

    # --- 3. 데이터 유형별로 전처리 실행 ---
    print("\n[3/3] 데이터 전처리를 시작합니다...")
    # 'form' 데이터는 이미 처리되었으므로 제외
    for data_type, target_count in DATA_RECIPE.items():
        process_data(data_type, target_count, paths)

    print("\n" + "="*50)
    print("🎉 모든 데이터 준비 작업이 완료되었습니다! 🎉")
    print(f"최종 데이터 폴더: {paths['ALL_DATA_DIR']}")
    print("="*50)

if __name__ == '__main__':
    main()