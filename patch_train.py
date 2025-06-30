# patch_train.py

import os
from pathlib import Path

def patch_train_py():
    print("[train.py 패치를 시작합니다...]")
    
    # train.py 파일 경로 (E:\download\EasyOCR\trainer\train.py)
    train_path = Path("EasyOCR/trainer/train.py") 
    
    # patch_train.py가 E:\download 에 있다고 가정합니다.
    current_patch_script_dir = Path(__file__).resolve().parent 
    train_full_path = current_patch_script_dir / train_path
    
    if not train_full_path.exists():
        print(f"❌ train.py 파일을 찾을 수 없습니다! 예상 경로: {train_full_path}")
        return False
    
    # 백업 생성
    backup_path = train_full_path.with_suffix('.py.backup')
    if not backup_path.exists():
        print("원본 파일 백업 중...")
        with open(train_full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
    
    # 새로운 train.py 내용
    new_content = '''
import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml # yaml import 추가
import argparse # argparse import 추가
from pathlib import Path # Path import 추가

# 표준 출력 인코딩을 UTF-8로 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# === 변경된 부분 시작 ===
# sys.path에 EasyOCR 패키지를 포함하는 루트 디렉토리 (E:\download)를 추가하여
# Python이 'EasyOCR' 패키지를 찾을 수 있도록 합니다.
# train.py 파일의 경로: E:\download\EasyOCR\\trainer\\train.py
# 이 파일에서 E:\download를 얻기 위해 Path(__file__).resolve().parents[3] 사용 (trainer -> EasyOCR -> download)
project_root_for_sys_path = Path(__file__).resolve().parents[3] 

if str(project_root_for_sys_path) not in sys.path:
    sys.path.insert(0, str(project_root_for_sys_path))
print(f"[DEBUG] Project root for EasyOCR added to sys.path: {project_root_for_sys_path}")
print(f"[DEBUG] Current sys.path: {sys.path}") # 디버그 출력 추가

# EasyOCR 내부 모듈 임포트 - 'EasyOCR' 최상위 패키지 이름을 그대로 사용
# (easyocr -> EasyOCR로 변경)
from EasyOCR.trainer.utils.model_utils import get_model
from EasyOCR.utils.tools import get_size_of_output, CTCLabelConverter, AttnLabelConverter, Averager # Averager도 tools에 있음
from EasyOCR.trainer.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from EasyOCR.model import Model
from EasyOCR.test import validation
# === 변경된 부분 끝 ===

def print_debug(msg):
    print(f"[DEBUG] {msg}")

def load_yaml(yaml_path):
    print_debug(f"YAML 파일 로드 시도: {yaml_path}")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print_debug(f"로드된 설정: {config}")
            return config
    except Exception as e:
        print_debug(f"YAML 로드 실패: {e}")
        return None

def train(opt, show_number = 2, amp=False):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print_debug('Filtering the images containing characters which are not in opt.character')
        print_debug('Filtering the images whose label is longer than opt.batch_max_length')

    if isinstance(opt.select_data, str):
        opt.select_data = opt.select_data.split('-')
    if isinstance(opt.batch_ratio, str):
        opt.batch_ratio = opt.batch_ratio.split('-')

    train_dataset = Batch_Balanced_Dataset(opt)

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    if not os.path.exists(f'./saved_models/{opt.experiment_name}'):
        os.makedirs(f'./saved_models/{opt.experiment_name}')
    
    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")
    
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True, 
        num_workers=int(opt.workers), prefetch_factor=512,
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print_debug('-' * 80) 
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if not opt.character: 
        if not opt.character_list or not os.path.exists(opt.character_list):
            raise ValueError("character_list 경로가 필요하거나 파일이 존재하지 않습니다!")
        with open(opt.character_list, 'r', encoding='utf-8') as f:
            character_set_from_file = ''.join(f.read().splitlines())
        opt.character = character_set_from_file
    
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    model = Model(opt)
    print_debug('model input parameters') 
    print_debug(f'imgH: {opt.imgH}, imgW: {opt.imgW}, num_fiducial: {opt.num_fiducial if hasattr(opt, "num_fiducial") else "N/A"}, input_channel: {opt.input_channel}, output_channel: {opt.output_channel}, hidden_size: {opt.hidden_size}, num_class: {opt.num_class}, batch_max_length: {opt.batch_max_length}, Transformation: {opt.Transformation}, FeatureExtraction: {opt.FeatureExtraction}, SequenceModeling: {opt.SequenceModeling}, Prediction: {opt.Prediction}') 
    
    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v
        
        model = torch.nn.DataParallel(model).to(device) 
        print_debug(f'loading pretrained model from {opt.saved_model}') 

        model_dict = model.state_dict()
        if opt.FT:
            print_debug("FT 모드: Prediction 레이어 가중치 로드에서 제외.")
            pretrained_dict_filtered = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape and not k.startswith('Prediction.')}
        else:
            pretrained_dict_filtered = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        if not pretrained_dict_filtered:
            print_debug("⚠️ 사전 학습 모델에서 로드할 수 있는 유효한 키가 없습니다. 모든 키가 불일치하거나 비어있습니다.")
        else:
            model_dict.update(pretrained_dict_filtered)
            model.load_state_dict(model_dict)
            print_debug("사전 학습 모델 로드 성공 (일부 키는 건너뛸 수 있음)")

        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class) 
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device) 
    else:
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print_debug(f'Skip {name} as it is already initialized') 
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e: 
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)
    
    model.train() 
    print_debug("Model:") 
    print_debug(model) 
    count_parameters(model)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device) 
    loss_avg = Averager()

    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print_debug(f'Trainable params num : {sum(params_num)}') 

    if opt.optim=='adam':
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print_debug("Optimizer:") 
    print_debug(optimizer) 

    """ final options """
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    if not os.path.exists(f'./saved_models/{opt.experiment_name}'):
        os.makedirs(f'./saved_models/{opt.experiment_name}')

    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print_debug(opt_log) 
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print_debug(f'continue to train, start_iter: {start_iter}') 
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter
    
    scaler = GradScaler()
    t1= time.time()
        
    while(True):
        # train part
        optimizer.zero_grad(set_to_none=True)
        
        if amp:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1]) 
                    target = text[:, 1:] 
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)
            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1]) 
                target = text[:, 1:] 
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
            optimizer.step()
        loss_avg.add(cost)

        if (i % opt.valInterval == 0) and (i!=0):
            print_debug('training time: ', time.time()-t1) 
            t1=time.time()
            elapsed_time = time.time() - start_time
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                model.train()

                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print_debug(loss_model_log) 
                log.write(loss_model_log + '\n')

                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                start = random.randint(0,len(labels) - show_number )     
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print_debug(predicted_result_log) 
                log.write(predicted_result_log + '\n')
                print_debug('validation time: ', time.time()-t1) 
                t1=time.time()
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print_debug('end the training') 
            sys.exit()
        i += 1

def main(): 
    parser = argparse.ArgumentParser(description='Train EasyOCR model')
    
    parser.add_argument('--experiment_name', help='Where to store logs and models', default='new_model') 
    parser.add_argument('--train_data', required=True, help='path to training dataset') 
    parser.add_argument('--valid_data', required=True, help='path to validation dataset') 
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--optim', choices=['adam', 'adadelta'], default='adadelta', help='optimizer') 
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--rgb', action='store_true', help='Use RGB image')
    parser.add_argument('--character_list', type=str, default='', help='Path to character list file') 
    parser.add_argument('--Transformation', type=str, required=False, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=False, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=False, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=False, help='Prediction stage. CTC|Attn')
    parser.add_argument('--input_channel', type=int, default=1, help='number of input channels (grayscale)')
    parser.add_argument('--output_channel', type=int, default=512, help='number of output channels (character set size + CTC blank). Will be set by character_list.') 
    parser.add_argument('--imgH', type=int, default=32, help='height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='width of the input image')
    parser.add_argument('--hidden_size', type=int, default=256, help='size of the LSTM hidden states')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive case mode') 
    parser.add_argument('--PAD', action='store_true', help='for padding')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off')
    parser.add_argument('--batch_max_length', type=int, default=25, help='Maximum length of a label')
    parser.add_argument('--saveInterval', type=int, default=100000, help='Interval between each model save')
    parser.add_argument('--new_prediction', action='store_true', help='set new prediction layer') 
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points') 
    parser.add_argument('--contrast_adjust', action='store_true', help='Perform contrast adjustment') 
    parser.add_argument('--freeze_FeatureFxtraction', action='store_true', help='Freeze feature extraction layer') 
    parser.add_argument('--freeze_SequenceModeling', action='store_true', help='Freeze sequence modeling layer') 
    parser.add_argument('--character', type=str, default='', help='character label') 
    parser.add_argument('--select_data', type=str, default='MJ-ST', help='select training data') 
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5', help='batch ratio for each dataset') 
    
    opt = parser.parse_args()

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    if opt.config_file:
        print_debug(f"YAML 설정 파일 로드: {opt.config_file}")
        yaml_config = load_yaml(opt.config_file)
        if yaml_config:
            for k, v in yaml_config.items():
                if hasattr(opt, k): 
                    setattr(opt, k, v)
                    print_debug(f"YAML 설정 적용: {k} = {v}")
                else:
                    print_debug(f"⚠️ YAML에 정의되었지만 argparse에 없는 인자: {k} = {v}") 

    if not hasattr(opt, 'experiment_name') or opt.experiment_name is None or opt.experiment_name == '':
        opt.experiment_name = 'default_experiment'
    if not hasattr(opt, 'optim') or opt.optim is None:
        opt.optim = 'adadelta'
    if not hasattr(opt, 'num_fiducial') or opt.num_fiducial is None:
        opt.num_fiducial = 20
    if not hasattr(opt, 'contrast_adjust') or opt.contrast_adjust is None:
        opt.contrast_adjust = False
    if not hasattr(opt, 'freeze_FeatureFxtraction') or opt.freeze_FeatureFxtraction is None:
        opt.freeze_FeatureFxtraction = False
    if not hasattr(opt, 'freeze_SequenceModeling') or opt.freeze_SequenceModeling is None:
        opt.freeze_SequenceModeling = False
    if not hasattr(opt, 'new_prediction') or opt.new_prediction is None:
        opt.new_prediction = False
    
    if not opt.character and opt.character_list:
        with open(opt.character_list, 'r', encoding='utf-8') as f:
            opt.character = ''.join(f.read().splitlines())
    elif not opt.character:
        raise ValueError("character 또는 character_list 경로가 필요합니다!")

    print_debug(f"최종 설정: {vars(opt)}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    model = Model(opt) 
    
    train(opt) 

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print_debug(f"실행 중 오류 발생: {e}")
        import traceback
        print_debug(traceback.format_exc())
        sys.exit(1)
'''
    
    # 새 내용 저장
    try:
        with open(train_full_path, 'w', encoding='utf-8') as f:
            f.write(new_content.strip())
        print("✅ train.py 파일이 성공적으로 수정되었습니다!")
        return True
    except Exception as e:
        print(f"❌ 파일 수정 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    patch_train_py()