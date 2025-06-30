# check_structure.py

import os
import sys

# --- 사용자 설정 ---
# 구조를 확인하고 싶은 최상위 폴더 경로를 여기에 입력하세요.
# 윈도우 경로를 파이썬에서 사용할 때는 슬래시(/)를 사용하거나 역슬래시를 두 번(\\) 써야 합니다.
TARGET_DIR = "E:/download/"

# 결과가 저장될 파일 이름
OUTPUT_FILE = "structure.txt"
# --------------------


def generate_tree_structure():
    """지정된 폴더의 구조를 텍스트 파일로 저장하는 함수"""
    
    # 결과 파일은 이 스크립트가 있는 위치에 저장됩니다.
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd() # .py로 실행하지 않을 경우 대비
        
    output_path = os.path.join(base_dir, OUTPUT_FILE)

    print(f"'{TARGET_DIR}'의 폴더 구조를 읽어서 '{output_path}'에 저장합니다...")

    if not os.path.exists(TARGET_DIR):
        print(f"❌ 오류: 지정된 폴더를 찾을 수 없습니다 - '{TARGET_DIR}'")
        return

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for root, dirs, files in os.walk(TARGET_DIR):
                # __pycache__ 같은 불필요한 폴더는 건너뜁니다.
                if '__pycache__' in dirs:
                    dirs.remove('__pycache__')

                level = root.replace(TARGET_DIR, '').count(os.sep)
                indent = '│   ' * (level)
                
                # 현재 폴더 이름을 파일에 씁니다.
                f.write(f'{indent}└── {os.path.basename(root)}/\n')
                
                # 해당 폴더 안의 파일들을 샘플로 기록합니다.
                sub_indent = '│   ' * (level + 1)
                
                # 파일 목록 샘플링 (최대 5개)
                files_to_show = files[:5]
                for i, file_name in enumerate(files_to_show):
                    f.write(f'{sub_indent}📄 {file_name}\n')
                
                if len(files) > 5:
                    f.write(f'{sub_indent}(... and {len(files) - 5} more files)\n')

        print(f"✅ 작업 완료! '{output_path}' 파일을 확인해주세요.")

    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")


if __name__ == '__main__':
    generate_tree_structure()