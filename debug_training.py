# debug_training.py 수정

import os
import subprocess
import sys

def main():
    print("[1/3] 학습에 필요한 경로와 파일들을 확인합니다...")
    try:
        BASE_DIR = os.getcwd()
        EASYOCR_DIR = os.path.join(BASE_DIR, "EasyOCR")
        
        train_script_path = os.path.join(EASYOCR_DIR, "trainer", "train.py")
        config_file_path = os.path.join(BASE_DIR, "custom_model.yaml")
        pretrained_model_path = os.path.join(BASE_DIR, "korean_g2.pth")

        for path in [train_script_path, config_file_path, pretrained_model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {path}")

        python_executable = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
        if not os.path.exists(python_executable):
            python_executable = "python"
            print("경고: .venv 가상환경을 찾지 못해 시스템 기본 파이썬으로 실행합니다.")

        # PYTHONPATH 환경 변수에 EasyOCR 디렉토리 추가
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{EASYOCR_DIR}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = EASYOCR_DIR

        print("✅ 모든 필수 파일과 경로를 확인했습니다.")
    except Exception as e:
        print(f"❌ 초기화 중 오류: {e}"); sys.exit()

    print("\n[2/3] 최종 학습 명령어를 생성합니다...")
    command = [
        python_executable,
        train_script_path,
        "--config_file", config_file_path,
        "--pretrained_model", pretrained_model_path
    ]
    
    print("--- 실행될 최종 명령어 ---")
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
    print("--------------------------")
    
    print("\n[3/3] 모델 파인튜닝을 시작합니다. [디버그 모드]")
    try:
        result = subprocess.run(
            command, 
            env=env,  # 수정된 환경 변수 전달
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            errors='replace'
        )

        print("\n" + "="*50)
        print("               프로세스 실행 결과 분석")
        print("="*50)
        print(f"  - 종료 코드 (Return Code): {result.returncode} (0이면 성공, 0이 아니면 오류)")
        
        print("\n--- STDOUT (표준 출력) ---")
        if result.stdout:
            print(result.stdout.strip())
        else:
            print("(출력 내용 없음)")
            
        print("\n--- STDERR (오류 출력) ---")
        if result.stderr:
            print(result.stderr.strip())
        else:
            print("(오류 내용 없음)")
        print("="*50)

        if result.returncode == 0 and "iteration" in result.stdout:
             print("\n🎉 학습 프로세스가 성공적으로 시작된 것으로 보입니다!")
        elif result.returncode != 0:
             print("\n❗️ 드디어 찾았습니다! 학습 프로세스가 위의 오류와 함께 종료되었습니다.")
        else:
             print("\n❗️ 무반응 종료가 재현되었습니다. 오류는 없었지만 학습이 시작되지 않았습니다.")

    except Exception as e:
        print(f"❌ 학습 실행 중 심각한 오류 발생: {e}")

if __name__ == '__main__':
    main()