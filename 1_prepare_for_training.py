# 1_prepare_for_training.py

import os
import subprocess
import sys

# 1. EasyOCR 소스코드 다운로드 (git이 설치되어 있어야 함)
EASYOCR_DIR = os.path.join(os.getcwd(), "EasyOCR")
if not os.path.exists(EASYOCR_DIR):
    print("[1/2] EasyOCR 소스코드를 GitHub에서 다운로드합니다...")
    try:
        subprocess.run(["git", "clone", "https://github.com/JaidedAI/EasyOCR.git"], check=True)
        print("✅ 소스코드 다운로드 완료.")
    except Exception as e:
        print(f"❌ Git 클론 중 오류 발생: {e}")
        print("PC에 Git이 설치되어 있는지 확인해주세요. (https://git-scm.com/downloads)")
        sys.exit()
else:
    print("[1/2] EasyOCR 소스코드가 이미 존재합니다.")

# 2. 기본 모델 다운로드 및 경로 확인
print("\n[2/2] 파인튜닝에 사용할 기본 모델을 다운로드하고 경로를 확인합니다...")
try:
    # easyocr.Reader를 실행하면 필요한 모델이 자동으로 다운로드됩니다.
    import easyocr
    print("  - 기본 모델 다운로드를 위해 Reader를 초기화합니다. (시간이 걸릴 수 있습니다)")
    # gpu=True로 설정해야 GPU용 모델을 받습니다. PC에 NVIDIA GPU와 CUDA가 설치되어 있어야 합니다.
    reader = easyocr.Reader(['ko', 'en'], gpu=True) 
    
    # 모델이 저장된 표준 경로
    model_path = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model", "korean_g2.pth")
    
    if os.path.exists(model_path):
        print("\n" + "="*60)
        print("🎉 기본 모델 준비가 완료되었습니다!")
        print("다음 3단계에서 사용할 '사전 학습된 모델'의 경로는 아래와 같습니다.")
        print("이 경로를 복사해두세요.")
        print(f"\n{model_path}")
        print("="*60)
    else:
        print("\n❌ 오류: 기본 모델(korean_g2.pth)을 자동으로 다운로드하지 못했습니다.")

except Exception as e:
    print(f"\n❌ 기본 모델 준비 중 오류 발생: {e}")