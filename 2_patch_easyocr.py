# patch_easyocr.py

import os
import shutil
from pathlib import Path

def patch_easyocr_files():
    print("[EasyOCR 패치 적용을 시작합니다...]")
    
    # 현재 작업 디렉토리 (E:\download) 확인
    base_dir = Path.cwd()
    easyocr_dir = base_dir / "EasyOCR"
    
    if not easyocr_dir.exists():
        print("❌ EasyOCR 폴더를 찾을 수 없습니다!")
        return False

    try:
        # 1. utils.py 수정
        utils_path = easyocr_dir / "trainer" / "utils.py"
        
        if not utils_path.exists():
            print("❌ utils.py 파일을 찾을 수 없습니다!")
            return False
        
        # 파일 내용 백업
        backup_path = utils_path.with_suffix('.py.backup')
        shutil.copy2(utils_path, backup_path)
        print(f"✅ 원본 파일 백업 완료: {backup_path}")

        # utils.py 수정된 내용
        utils_content = """
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CTCLabelConverter(object):
    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class AttnLabelConverter(object):
    def __init__(self, character):
        # character (str): set of the possible characters.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
"""
        # 새로운 내용으로 파일 작성
        with open(utils_path, 'w', encoding='utf-8') as f:
            f.write(utils_content.strip())
        
        print("✅ utils.py 파일 패치 완료")
        
        # 2. train.py 수정 - import 문 수정
        train_path = easyocr_dir / "trainer" / "train.py"
        with open(train_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # CTCLabelConverterForBaiduWarpctc 관련 import 제거
        content = content.replace(
            "from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager",
            "from utils import CTCLabelConverter, AttnLabelConverter, Averager"
        )
        
        # 수정된 내용 저장
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ train.py 파일 패치 완료")
        
        print("\n🎉 모든 패치가 성공적으로 적용되었습니다!")
        print("이제 학습 스크립트를 다시 실행해보세요.")
        
        return True

    except Exception as e:
        print(f"❌ 패치 적용 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    patch_easyocr_files()