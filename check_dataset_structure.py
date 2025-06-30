# check_dataset_structure.py

import os
from pathlib import Path
import glob

def check_directory_structure(base_dir):
    print("\n=== 데이터셋 디렉토리 구조 검사 시작 ===")
    
    base_path = Path(base_dir)
    
    for dataset_type in ['train_data', 'valid_data']:
        data_dir = base_path / "final_dataset" / dataset_type
        print(f"\n=== {dataset_type} 구조 ===")
        
        if not data_dir.exists():
            print(f"❌ {dataset_type} 디렉토리가 없습니다!")
            continue
            
        # 하위 디렉토리 탐색
        print("\n디렉토리 구조:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(str(data_dir), '').count(os.sep)
            indent = ' ' * 4 * level
            subdir = os.path.basename(root)
            print(f"{indent}[{subdir}]")
            
            # 파일 개수 출력
            image_count = len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
            label_count = len([f for f in files if f.endswith('.txt')])
            if image_count > 0 or label_count > 0:
                print(f"{indent}    - 이미지 파일: {image_count}개")
                print(f"{indent}    - 라벨 파일: {label_count}개")
        
        # images 폴더 확인
        images_dir = data_dir / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            if image_files:
                print(f"\n이미지 예시 (처음 3개):")
                for img_path in image_files[:3]:
                    print(f"- {img_path.name}")
        
        # labels 폴더 확인
        labels_dir = data_dir / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            if label_files:
                print(f"\n라벨 예시 (처음 3개):")
                for label_path in label_files[:3]:
                    try:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        print(f"- {label_path.name}: {content}")
                    except Exception as e:
                        print(f"- {label_path.name}: 읽기 오류 ({e})")
    
    print("\n=== 검사 완료 ===")

def create_label_files(base_dir):
    print("\n=== 라벨 파일 자동 생성 시작 ===")
    
    base_path = Path(base_dir)
    
    for dataset_type in ['train_data', 'valid_data']:
        data_dir = base_path / "final_dataset" / dataset_type
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        
        if not images_dir.exists():
            print(f"❌ {dataset_type}의 images 폴더가 없습니다!")
            continue
            
        # labels 디렉토리 생성
        labels_dir.mkdir(exist_ok=True)
        
        print(f"\n{dataset_type} 처리 중...")
        image_files = list(images_dir.glob("*.png"))
        
        for img_path in image_files:
            # 파일 이름에서 문자 추출 (예: hw_char_00130001012.png -> '가')
            # 여기서는 예시로 첫 번째 문자를 사용합니다
            char = "가"  # 실제 로직으로 교체 필요
            
            # 라벨 파일 생성
            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(char)
                
        print(f"✅ {len(image_files)}개의 라벨 파일 생성 완료")
    
    print("\n=== 라벨 파일 생성 완료 ===")

if __name__ == "__main__":
    base_dir = "E:/download"
    check_directory_structure(base_dir)
    
    # 사용자 확인 후 라벨 파일 생성
    response = input("\n라벨 파일을 자동으로 생성하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        create_label_files(base_dir)