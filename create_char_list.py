# 1_create_char_list.py

import os
import pandas as pd
from tqdm import tqdm

print("[1/2] 경로를 설정하고 라벨 파일을 읽습니다...")
try:
    BASE_DIR = os.getcwd()
    FINAL_DATA_DIR = os.path.join(BASE_DIR, "final_dataset")
    
    train_label_path = os.path.join(FINAL_DATA_DIR, "train_data", "labels.txt")
    valid_label_path = os.path.join(FINAL_DATA_DIR, "valid_data", "labels.txt")
    
    df_train = pd.read_csv(train_label_path, sep='\\t', header=None, engine='python', names=['filename', 'text'])
    df_valid = pd.read_csv(valid_label_path, sep='\\t', header=None, engine='python', names=['filename', 'text'])
    
    # 두 데이터프레임을 합쳐 전체 텍스트를 준비
    all_text = pd.concat([df_train['text'], df_valid['text']], ignore_index=True)
    print("✅ 라벨 파일 로드 완료.")

except Exception as e:
    print(f"❌ 파일 읽기 중 오류: {e}")

print("\n[2/2] 고유 문자 목록을 추출하고 chars.txt 파일로 저장합니다...")
try:
    # 모든 텍스트를 하나의 문자열로 합친 후, set을 이용해 고유 문자 추출
    all_char_set = set("".join(all_text.astype(str)))
    
    # 정렬된 리스트로 변환
    char_list = sorted(list(all_char_set))
    
    # 파일로 저장
    char_file_path = os.path.join(BASE_DIR, "chars.txt")
    with open(char_file_path, 'w', encoding='utf-8') as f:
        f.write("".join(char_list))

    print("\n" + "="*50)
    print("🎉 `chars.txt` 생성이 완료되었습니다! 🎉")
    print(f" > 저장된 경로: {char_file_path}")
    print(f" > 총 고유 문자 수: {len(char_list):,} 개")
    print("="*50)

except Exception as e:
    print(f"❌ 문자 목록 생성 중 오류: {e}")