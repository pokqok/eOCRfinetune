# check_dataset.py

import os
from pathlib import Path
import glob

def check_directory_structure(base_dir):
    print("\n=== 데이터셋 디렉토리 검사 시작 ===")
    
    # 기본 디렉토리 확인
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 기본 디렉토리가 존재하지 않습니다: {base_dir}")
        return False
    
    print(f"✅ 기본 디렉토리 확인됨: {base_dir}")
    
    # 학습/검증 데이터 디렉토리 확인
    train_dir = base_path / "final_dataset" / "train_data"
    valid_dir = base_path / "final_dataset" / "valid_data"
    
    for data_dir in [train_dir, valid_dir]:
        print(f"\n--- {data_dir.name} 검사 중 ---")
        
        if not data_dir.exists():
            print(f"❌ 디렉토리가 존재하지 않습니다: {data_dir}")
            continue
            
        print(f"✅ 디렉토리 존재함: {data_dir}")
        
        # 이미지 파일 찾기
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(data_dir.rglob(f"*{ext}")))
        
        if not image_files:
            print("❌ 이미지 파일을 찾을 수 없습니다!")
        else:
            print(f"✅ 발견된 이미지 파일 수: {len(image_files)}")
            
            # 처음 5개 이미지 파일 출력
            print("\n처음 5개 이미지 파일:")
            for img_path in image_files[:5]:
                label_path = img_path.with_suffix('.txt')
                print(f"\n이미지: {img_path.name}")
                
                # 라벨 파일 확인
                if label_path.exists():
                    try:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            label = f.read().strip()
                        print(f"✅ 라벨 파일 있음 (내용: {label})")
                    except Exception as e:
                        print(f"❌ 라벨 파일 읽기 오류: {e}")
                else:
                    print("❌ 라벨 파일 없음")
    
    # chars.txt 파일 확인
    chars_path = base_path / "chars.txt"
    if chars_path.exists():
        try:
            with open(chars_path, 'r', encoding='utf-8') as f:
                chars = f.read().strip()
            print(f"\n✅ chars.txt 파일 확인됨 (문자 수: {len(chars)})")
            print(f"포함된 문자: {chars[:100]}...")
        except Exception as e:
            print(f"\n❌ chars.txt 파일 읽기 오류: {e}")
    else:
        print("\n❌ chars.txt 파일이 없습니다!")
    
    print("\n=== 검사 완료 ===")

if __name__ == "__main__":
    check_directory_structure("E:/download")