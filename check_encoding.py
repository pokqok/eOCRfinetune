# check_encoding.py
import os
import glob
import json

# --- 설정 ---
BASE_DIR = os.getcwd()
UNZIPPED_DIR = os.path.join(BASE_DIR, "unzipped")
# 검사하고 싶은 원본 JSON 파일 이름
TARGET_JSON_FILENAME = "03310103037.json" 

print(f"'{TARGET_JSON_FILENAME}' 파일의 인코딩을 확인합니다...")

# --- 파일 찾기 ---
target_paths = glob.glob(os.path.join(UNZIPPED_DIR, '**', TARGET_JSON_FILENAME), recursive=True)

if not target_paths:
    print(f"❌ 오류: '{TARGET_JSON_FILENAME}' 파일을 찾을 수 없습니다.")
else:
    json_path = target_paths[0]
    print(f"✅ 파일을 찾았습니다: {json_path}")
    
    # --- 여러 인코딩으로 읽기 시도 ---
    encodings_to_try = ['utf-8', 'euc-kr', 'cp949']
    
    for enc in encodings_to_try:
        print("\n" + "="*30)
        print(f"--- '{enc}' 인코딩으로 읽기를 시도합니다 ---")
        try:
            with open(json_path, 'r', encoding=enc) as f:
                data = json.load(f)
            
            # '각종보험용' 이라는 텍스트가 포함된 주석을 찾습니다.
            found = False
            for anno in data.get('text', {}).get('word', []):
                value = anno.get('value', '')
                if '각종보험용' in value:
                    print("✅ 해당 주석을 찾았습니다!")
                    print(f"   - 원본 텍스트: {value}")
                    found = True
                    break
            if not found:
                 print("   - 파일은 열었지만, 해당 텍스트를 포함한 주석은 찾지 못했습니다.")

        except Exception as e:
            print(f"   - ❌ '{enc}'로 읽기 실패: {e}")