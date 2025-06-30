import os
import sys

# --- [최종 확인] 프로젝트의 기본 경로를 다시 한번 지정해주세요 ---
# 예: "E:/download" 또는 "E:\\download"
BASE_DIR = "E:/download"
# -----------------------------------------------------------

print(f"지정된 기본 경로: {BASE_DIR}")
print("이 경로를 기준으로 YAML 파일을 생성합니다...")

try:
    # 모든 경로 생성 시, 바로 슬래시(/)로 변환하여 오류 원천 차단
    FINAL_DATA_DIR = os.path.join(BASE_DIR, "final_dataset").replace('\\', '/')
    TRAIN_PATH = os.path.join(FINAL_DATA_DIR, 'train_data').replace('\\', '/')
    VALID_PATH = os.path.join(FINAL_DATA_DIR, 'valid_data').replace('\\', '/')
    CHAR_LIST_PATH = os.path.join(BASE_DIR, 'chars.txt').replace('\\', '/')
    SAVED_MODEL_PATH = os.path.join(BASE_DIR, 'saved_models').replace('\\', '/')
    USER_NETWORK_PATH = os.path.join(BASE_DIR, 'user_network').replace('\\', '/')
    FINAL_MODEL_PATH = os.path.join(SAVED_MODEL_PATH, "MyFinalModel.pth").replace('\\', '/')

    # YAML 내용 생성
    yaml_content = f"""
# train_data: 학습 데이터 경로
train_data: {TRAIN_PATH}

# valid_data: 검증 데이터 경로
valid_data: {VALID_PATH}

# data_filtering_off: True로 설정하면 일부 작은 이미지가 필터링되는 것을 방지
data_filtering_off: True

language_list:
  - 'ko'
  - 'en'

# character: 학습할 전체 문자 리스트가 담긴 txt 파일 경로
character_list: {CHAR_LIST_PATH}

# --- 모델 및 학습 파라미터 (8GB VRAM 최적화) ---
imgH: 64
batch_size: 16
num_workers: 0
valInterval: 1000
num_iter: 30000
FT: True

# saved_model: 최종 학습된 모델이 저장될 경로와 파일명
saved_model: {FINAL_MODEL_PATH}

# User-defined network an recognizer
user_network_params:
  recog_network: 'my_recognizer'
  user_network_directory: {USER_NETWORK_PATH}"""

    # 폴더 생성
    os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
    os.makedirs(USER_NETWORK_PATH, exist_ok=True)

    # 파일 쓰기
    yaml_file_path = os.path.join(BASE_DIR, "custom_model.yaml")
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content.strip())

    print("\n" + "="*50)
    print("🎉 YAML 파일 생성이 완료되었습니다! 🎉")
    print(f" > 저장된 경로: {yaml_file_path}")
    print("\n이제 다음 단계인 모델 학습을 진행할 수 있습니다.")
    print("="*50)

except Exception as e:
    import traceback
    print(f"❌ 오류가 발생했습니다: {e}")
    traceback.print_exc()