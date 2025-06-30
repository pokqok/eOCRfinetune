# count_data.py (경로 수정 버전)

import os
import glob
import pandas as pd
import sys

def main():
    print("--- 각 데이터 유형별 사용 가능한 파일 개수를 확인합니다 ---")
    
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        # [수정됨] AI Hub 데이터가 있는 실제 폴더를 기준으로 경로 설정
        AIHUB_SOURCE_DIR = os.path.join(BASE_DIR, "다양한 형태의 한글 문자 OCR")
        # 최종 결과물이 저장된 폴더 (form 데이터 확인용)
        ALL_DATA_DIR = os.path.join(BASE_DIR, "all_data")

        source_counts = {}

        # 1. form 데이터 개수 확인 (이미 처리된 결과물 기준)
        form_label_file = os.path.join(ALL_DATA_DIR, "labels.txt")
        if os.path.exists(form_label_file):
            with open(form_label_file, 'r', encoding='utf-8') as f:
                # 'printed_' 로 시작하는 라인만 카운트하여 정확성 향상
                count = sum(1 for line in f if 'printed_' in line.split('\t')[0])
                source_counts['form (처리 완료된 글자/단어 수)'] = count
        else:
            source_counts['form (처리 완료된 글자/단어 수)'] = 0

        # 2. 글자 데이터 개수 확인 (수정된 경로에서 검색)
        source_counts['글자 (처리 대상 원본 파일 수)'] = len(glob.glob(os.path.join(AIHUB_SOURCE_DIR, '**', '*글자*/**/*.json'), recursive=True))

        # 3. 단어 데이터 개수 확인 (수정된 경로에서 검색)
        source_counts['단어 (처리 대상 원본 파일 수)'] = len(glob.glob(os.path.join(AIHUB_SOURCE_DIR, '**', '*단어*/**/*.json'), recursive=True))

        # 4. MNIST 데이터 개수 확인
        mnist_label_file = os.path.join(AIHUB_SOURCE_DIR, "mnist_labels.csv")
        if os.path.exists(mnist_label_file):
            source_counts['mnist (준비된 숫자 이미지 수)'] = len(pd.read_csv(mnist_label_file))
        else:
            source_counts['mnist (준비된 숫자 이미지 수)'] = 0

        print("\n" + "="*50)
        print("                 현재 데이터 개수 현황")
        print("="*50)
        for name, count in source_counts.items():
            # 보기 편하도록 간격 조정
            print(f"  - {name:<30}: {count:>,} 개")
        print("="*50)

    except Exception as e:
        print(f"❌ 개수 확인 중 오류가 발생했습니다: {e}")
        print("경로가 올바른지 확인해주세요.")

if __name__ == '__main__':
    main()