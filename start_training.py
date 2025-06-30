# start_training.py

import os
import subprocess
import sys

def main():
    print("[1/3] 학습에 필요한 경로와 파일들을 확인합니다...")
    try:
        BASE_DIR = os.getcwd()
        EASYOCR_DIR = os.path.join(BASE_DIR, "EasyOCR")
        
        # --- 학습에 필요한 파일 3가지 ---
        # 1. 학습 실행 파일
        train_script_path = os.path.join(EASYOCR_DIR, "trainer", "train.py")
        # 2. 우리가 만든 설정 파일
        config_file_path = os.path.join(BASE_DIR, "custom_model.yaml")
        # 3. 파인튜닝의 기반이 될 원본 모델 파일
        pretrained_model_path = os.path.join(BASE_DIR, "korean_g2.pth")

        # 모든 파일이 존재하는지 확인
        for path in [train_script_path, config_file_path, pretrained_model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {path}")

        # 실행할 가상환경의 파이썬 경로
        python_executable = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
        if not os.path.exists(python_executable):
            # 시스템 기본 파이썬을 사용하도록 대체
            python_executable = "python"
            print("경고: .venv 가상환경을 찾지 못해 시스템 기본 파이썬으로 실행합니다.")

        print("✅ 모든 필수 파일과 경로를 확인했습니다.")

    except Exception as e:
        print(f"❌ 초기화 중 오류: {e}"); sys.exit()

    
    print("\n[2/3] 최종 학습 명령어를 생성합니다...")
    
    # [핵심] 명령어를 리스트 형태로 만듭니다. 이것이 가장 안정적인 방법입니다.
    command = [
        python_executable,
        train_script_path,
        "--config_file", config_file_path,
        "--pretrained_model", pretrained_model_path
    ]
    
    print("--- 실행될 최종 명령어 ---")
    # 사용자가 보기 편하도록 문자열로 변환하여 출력
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
    print("--------------------------")
    
    print("\n[3/3] 모델 파인튜닝을 시작합니다. (오류나 진행 상황은 이 아래에 표시됩니다)")
    try:
        # Popen을 사용하여 자식 프로세스로 학습을 실행하고, 모든 출력을 실시간으로 가져옵니다.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')

        # 학습 과정의 모든 출력을 실시간으로 화면에 보여줍니다.
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 프로세스가 끝난 후 최종 리턴 코드 확인
        rc = process.poll()
        if rc == 0:
            print("\n🎉 학습 프로세스가 성공적으로 완료되었습니다.")
        else:
            print(f"\n❗️ 학습 프로세스가 오류와 함께 종료되었습니다. (종료 코드: {rc})")

    except Exception as e:
        print(f"❌ 학습 실행 중 심각한 오류 발생: {e}")

if __name__ == '__main__':
    main()