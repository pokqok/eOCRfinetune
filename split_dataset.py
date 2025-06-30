import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# --- 1. 허용 문자셋 정의 (여기에 복사하여 넣어주세요) ---
# 예시: ALLOWED_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALLOWED_CHARS = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㆍ가각간갇갈갉감갑값갓갔강갖같갚개객갤갭갯갱갸걀거걱건걷걸검겁것겉게겐겔겟겠겨격겪견결겸겹겼경곁계고곡곤곧골곰곱곳공곶과곽관괄괌광괘괜괭괴괼굉교굘구국군굳굴굵굶굽굿궁궂궈권궐궤귀귄귈규균귤그극근글긁금급긋긍기긴길김깁깃깊까깍깎깐깔깜깝깡깥깨깬꺼꺾껄껌껍껏껐껑께껴꼈꼬꼭꼴꼼꼽꽁꽂꽃꽈꽉꽝꽤꾀꾸꾼꿀꿇꿈꿔꿨꿰뀌뀐뀔끄끈끊끌끓끔끗끝끼낀낄낌나낙낚난날낡남납낫났낭낮낯낱낳내낸낼냄냅냇냈냉냐냥너넉넌널넓넘넛넣네넥넨넬넵넷녀녁년념녔녕녘녜녠노녹논놀놈놉농높놓놔놨뇌뇨뇰뇽누눅눈눌눔눕눙눠뉘뉜뉴늄느늑는늘늙늠능늦늪늬니닉닌닐님닙닛닝다닥닦단닫달닭닮닳담답닷당닻닿대댁댄댈댐댑댓더덕던덜덟덤덥덧덩덫덮데덱덴델뎀뎅뎌뎬도독돈돋돌돔돕돗동돛돼됐되된될됨됩두둑둔둘둠둡둥둬뒀뒤뒷듀듈듐드득든듣들듬듭듯등디딕딘딛딜딥딧딩딪따딱딴딸땀땃땄땅때땐땡떠떡떤떨떻떼또똑똔똘똥뚜뚝뚫뚱뛰뛴뛸뜀뜨뜩뜬뜯뜰뜸뜻띄띈띠띤라락란랄람랍랏랐랑랗래랙랜랠램랩랫랬랭랴략랸량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄로록론롤롬롭롯롱롸뢰룀료룡루룩룬룰룸룹룻룽뤄뤘뤼뤽륀륄류륙륜률륨륭르륵른를름릅릇릉릎리릭린릴림립릿링마막만많맏말맑맘맙맛망맞맡맣매맥맨맬맴맵맷맹맺먀머먹먼멀멈멋멍메멕멘멜멤멧멩며멱면멸몄명몇모목몫몬몰몸몹못몽뫼묀묘무묵묶문묻물뭄뭇뭉뭐뭔뭘뮌뮐뮤뮬므믈믐미믹민믿밀밈밋밍및밑바박밖반받발밝밟밤밥밧방밭배백밴밸뱀뱃뱅뱉뱌버벅번벌범법벗벙벚베벡벤벨벰벳벵벼벽변별볍병볕보복볶본볼봄봅봇봉봐봤뵈뵙뵤부북분불붉붐붓붕붙뷔뷘뷰브븐블비빅빈빌빔빕빗빙빚빛빠빡빤빨빵빼빽뺀뺏뺑뺨뻐뻔뻗뻘뻤뼈뽀뽑뾰뿌뿐뿔뿜쁘쁜쁨삐사삭산살삶삼삽삿샀상새색샌샐샘샛생샤샨샬샴샵샷샹섀서석섞선섣설섬섭섯섰성세섹센셀셈셉셋셍셔션셜셤셨셰셴셸소속손솔솜솟송솽쇄쇠쇼숀숄숍숏숑수숙순술숨숫숭숱숲숴쉐쉬쉴쉼쉽슈슐슘슛슝스슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌈쌌쌍쌓써썩썬썰썸썹썼썽쎄쏘쏙쏜쏟쏠쏴쐈쐐쑈쑤쑥쑨쑹쓰쓴쓸씀씌씨씩씬씹씻씽아악안앉않알앓암압앗았앙앞애액앤앨앰앱앳앴앵야약얀얄얇얌얏양얕얘어억언얹얻얼얽엄업없엇었엉엌엎에엑엔엘엠엡엣엥여역엮연열엷염엽엿였영옅옆예옌옐옙옛오옥온올옭옮옳옴옵옷옹옻와왁완왈왓왔왕왜외왼요욕욘욜용우욱운울움웁웃웅워웍원월웜웠웨웬웰웸웹위윅윈윌윔윗윙유육윤율윳융으은을음읍응의이익인일읽잃임입잇있잉잊잎자작잔잖잘잠잡잣장잦재잭잼잿쟁쟈쟝저적전절젊점접젓정젖제젝젠젤젭젯젱져젼졌조족존졸좀좁종좇좋좌좡죄죠주죽준줄줌줍중줘줬쥐쥔쥘쥬즈즉즌즐즘즙증지직진질짊짐집짓징짖짙짚짜짝짠짤짧짱째쨌쩌쩍쩐쩔쩡쪼쪽쫓쬐쭈쭉쭝쭤쯔쯤찌찍찐찔찢차착찬찮찰참찹찻창찾채책챈챌챔챗챙챠처척천철첨첩첫청체첸첼쳇쳐쳤초촉촌촐촘촛총촨촬최쵸추축춘출춤춥춧충춰췄췌취츄츠측츨츰츳층치칙친칠침칩칭카칸칼캄캅캇캉캐캔캘캠캡캣캥캬커컨컫컬컴컵컷컸컹케켁켄켈켐켓켜켰코콕콘콜콤콥콧콩콰콴콸쾌쾨쾰쿄쿠쿡쿤쿨쿰쿼퀀퀄퀘퀴퀸퀼큐큘크큰클큼키킥킨킬킴킵킷킹타탁탄탈탐탑탓탔탕태택탠탤탬탭탱탸터턱턴털텀텁텃텅테텍텐텔템텝텟텡톈토톡톤톨톰톱통퇴투툭툰툴툼퉁튀튕튜튠튬트특튼튿틀틈티틱틴틸팀팁팅파팍팎판팔팜팝팟팡팥패팩팬팰팹팻팽퍼퍽펀펄펌펑페펙펜펠펨펩펫펴편펼폄폈평폐포폭폰폴폼퐁푀표푸푹푼풀품풋풍퓌퓨퓰프픈플픔피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햇했행햐향허헉헌헐험헛헝헤헥헨헬헴헵혀혁현혈혐협혔형혜호혹혼홀홈홉홋홍화확환활황홰횃회획횟횡효후훅훈훌훑훔훗훙훤훨훼휘휜휠휩휴흉흐흑흔흘흙흠흡흥흩희흰히힉힌힐힘힙'
ALLOWED_CHAR_SET = set(list(ALLOWED_CHARS))

def check_label_contains_allowed_chars(label, allowed_char_set):
    """레이블의 모든 문자가 허용 문자셋에 있는지 확인합니다."""
    for char in label:
        if char not in allowed_char_set:
            return False
    return True

def setup_split_folder(df, dest_dir, source_base_root): # source_image_root 대신 source_base_root 사용
    """
    데이터프레임을 받아 최종 폴더 구조를 생성합니다.
    (이미지 복사 및 labels.txt 생성)
    """
    img_dest_dir = os.path.join(dest_dir, "images")
    labels_path = os.path.join(dest_dir, "labels.txt")
    
    # 기존 디렉토리가 있다면 삭제하고 새로 생성
    if os.path.exists(dest_dir): 
        shutil.rmtree(dest_dir)
    os.makedirs(img_dest_dir)
    
    # DataFrame의 'filename' 열은 'images/...'와 같은 상대 경로를 포함
    # 새 라벨 파일에는 이 상대 경로와 텍스트를 탭으로 구분하여 저장
    df[['filename', 'text']].to_csv(labels_path, sep='\t', header=False, index=False, encoding='utf-8')

    # 이미지 파일 복사
    # 'filename' (예: 'images/xxx.jpg')을 사용하여 원본 경로 구성
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"'{os.path.basename(dest_dir)}' 이미지 복사 중"):
        relative_img_path = row['filename'] # 예: images/printed_01010033039_0052.png
        src_img_path = os.path.join(source_base_root, relative_img_path) # 예: all_data/images/printed_01010033039_0052.png
        
        # 대상 폴더의 이미지 파일명은 labels.txt에 사용된 그대로의 파일명 (예: printed_01010033039_0052.png)
        # 즉, 'images/' 접두사 없이 최종 파일명만 dest_dir/images/ 에 복사
        dest_img_name = os.path.basename(relative_img_path) 
        
        if os.path.exists(src_img_path):
            shutil.copyfile(src_img_path, os.path.join(img_dest_dir, dest_img_name))
        else:
            print(f"경고: 원본 이미지를 찾을 수 없습니다 - {src_img_path}. 이 항목은 건너뜨.", file=sys.stderr)
            
    print(f"✅ '{os.path.basename(dest_dir)}' 생성 완료!")

def verify_split_folder_final(dir_path, expected_count):
    """
    분할된 폴더의 정합성을 양방향으로 완벽하게 검증하는 함수입니다.
    이미지 파일과 labels.txt의 파일 목록을 비교합니다.
    """
    print("\n" + "-"*20 + f" [{os.path.basename(dir_path)}] 최종 검증 " + "-"*20)
    try:
        image_dir = os.path.join(dir_path, "images")
        label_file = os.path.join(dir_path, "labels.txt")

        if not os.path.exists(image_dir):
            print(f" ❌ 오류: 이미지 디렉토리 '{image_dir}'를 찾을 수 없습니다.", file=sys.stderr)
            return False
        if not os.path.exists(label_file):
            print(f" ❌ 오류: 라벨 파일 '{label_file}'을 찾을 수 없습니다.", file=sys.stderr)
            return False

        # 1. 실제 디스크에 있는 이미지 파일 목록
        image_files_on_disk = set(os.listdir(image_dir))
        image_count = len(image_files_on_disk)

        # 2. 라벨 파일에 있는 파일 목록
        # labels.txt의 filepath는 'images/xxx.jpg' 형태일 수 있으므로 basename으로 실제 파일명만 추출해야 비교 가능
        labels_df = pd.read_csv(label_file, sep='\t', header=None, engine='python', names=['filepath', 'text'])
        filenames_in_label = set(labels_df['filepath'].apply(os.path.basename)) # basename 적용
        label_count = len(labels_df)

        # 3. 교차 검증: 라벨에 있지만 이미지가 없는 경우 / 이미지에 있지만 라벨이 없는 경우
        orphaned_labels = filenames_in_label - image_files_on_disk  # 라벨은 있는데 해당 이미지가 디스크에 없음
        orphaned_images = image_files_on_disk - filenames_in_label  # 이미지는 있는데 해당 라벨이 labels.txt에 없음

        print(f"  - 기대 데이터 수: {expected_count:,} 개")
        print(f"  - 실제 이미지 파일 수: {image_count:,} 개")
        print(f"  - 실제 라벨 수: {label_count:,} 개")
        print("-" * 25)
        print(f"  - [검증] 이미지 없는 라벨 수: {len(orphaned_labels)} 건")
        if len(orphaned_labels) > 0:
            print(f"    (예시: {list(orphaned_labels)[:5]})") # 첫 5개만 예시로 출력
        print(f"  - [검증] 라벨 없는 이미지 수: {len(orphaned_images)} 건")
        if len(orphaned_images) > 0:
            print(f"    (예시: {list(orphaned_images)[:5]})") # 첫 5개만 예시로 출력

        # 모든 조건이 일치해야 성공으로 간주
        if (expected_count == image_count and 
            image_count == label_count and 
            len(orphaned_labels) == 0 and 
            len(orphaned_images) == 0):
            print("  - ✅ 정합성 검증 성공! 완벽하게 일치합니다.")
            return True
        else:
            print("  - ❌ 정합성 검증 실패! 데이터에 불일치가 있습니다. 위 로그를 확인해주세요.")
            return False
            
    except Exception as e:
        print(f"  - ❌ 검증 중 오류 발생: {e}", file=sys.stderr)
        return False


def main():
    print("[1/4] 경로 설정 및 원본 데이터 로드를 시작합니다...")
    BASE_DIR = os.getcwd() # 현재 스크립트 실행 경로
    ALL_DATA_DIR = os.path.join(BASE_DIR, "all_data") # 원본 이미지와 labels.txt가 있는 최상위 폴더
    # SOURCE_IMAGE_DIR은 더 이상 직접적으로 사용되지 않습니다.
    # 대신 SOURCE_BASE_ROOT를 all_data로 설정하여 labels.txt의 상대 경로를 직접 사용합니다.
    SOURCE_BASE_ROOT = ALL_DATA_DIR 
    SOURCE_LABEL_FILE = os.path.join(ALL_DATA_DIR, "labels.txt") # 원본 labels.txt 파일 경로
    
    # 최종 분할된 데이터가 저장될 폴더 (LMDB 폴더가 아님)
    FINAL_DATA_DIR = os.path.join(BASE_DIR, "final_dataset") 
    TRAIN_DIR = os.path.join(FINAL_DATA_DIR, 'train_data') # 학습 데이터 폴더
    VALID_DIR = os.path.join(FINAL_DATA_DIR, 'valid_data') # 검증 데이터 폴더

    # 학습 및 검증 데이터셋 비율
    TRAIN_RATIO = 0.9 # 학습 90%, 검증 10%
    RANDOM_STATE = 42 # 재현성을 위한 시드

    # ALLOWED_CHARS가 설정되었는지 확인
    if ALLOWED_CHARS == "YOUR_ALLOWED_CHARS_STRING_HERE" or not ALLOWED_CHARS:
        print("에러: `ALLOWED_CHARS` 변수에 필터링할 문자열을 넣어주세요!", file=sys.stderr)
        sys.exit(1)
    print(f"사용할 허용 문자셋 ({len(ALLOWED_CHARS)}자): {ALLOWED_CHARS}")


    # labels.txt 파일 읽기 및 필터링
    print("\n[2/4] labels.txt 파일을 읽고 허용 문자셋 기준으로 데이터를 필터링합니다...")
    all_samples = [] # (이미지 파일명, 레이블 텍스트) 튜플 저장 (필터링 후)
    skipped_count = 0

    if not os.path.exists(SOURCE_LABEL_FILE):
        print(f"에러: 원본 레이블 파일이 존재하지 않습니다 - {SOURCE_LABEL_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(SOURCE_LABEL_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line: # 빈 줄 건너뛰기
                continue
            
            parts = line.split('\t', 1)
            if len(parts) == 2:
                img_relative_path, label_text = parts # labels.txt에 있는 그대로의 상대 경로
                full_img_path = os.path.join(SOURCE_BASE_ROOT, img_relative_path) # all_data/images/xxx.png 형태로 생성

                # 이미지 파일 존재 여부 확인 (필수)
                if not os.path.exists(full_img_path):
                    print(f"경고: 라벨은 있지만 이미지가 없는 파일 발견 (줄 {line_num+1}): {full_img_path}. 건너뜁니다.", file=sys.stderr)
                    skipped_count += 1
                    continue

                # --- 텍스트 필터링 적용 ---
                if check_label_contains_allowed_chars(label_text, ALLOWED_CHAR_SET):
                    # 필터링을 통과한 데이터만 저장. filename은 labels.txt에 기록될 상대 경로 (images/xxx.png)
                    all_samples.append({'filename': img_relative_path, 'text': label_text})
                else:
                    # 필터링으로 건너뛴 경우
                    # print(f"필터링됨 (허용되지 않는 문자 포함): '{label_text}' ({img_relative_path})", file=sys.stderr)
                    skipped_count += 1
            else:
                # 라벨 파일 형식이 잘못된 경우
                print(f"경고: 잘못된 형식의 라벨 파일 줄 (줄 {line_num+1}): {line}. 건너뜱니다.", file=sys.stderr)
                skipped_count += 1
    
    if not all_samples:
        print("에러: 필터링 후 유효한 샘플이 없습니다. 작업을 중단합니다.", file=sys.stderr)
        sys.exit(1)

    # 필터링된 데이터를 DataFrame으로 변환
    master_df = pd.DataFrame(all_samples)
    print(f"✅ 총 {len(master_df)}개의 유효한 샘플을 로드했습니다. ({skipped_count}개 샘플 필터링/건너뜀)")

    print(f"\n[3/4] 유효한 데이터를 학습용({TRAIN_RATIO*100:.0f}%)과 검증용({(1-TRAIN_RATIO)*100:.0f}%)으로 분할합니다...")
    # random_state를 고정하여 항상 동일한 결과가 나오도록 함
    train_df, valid_df = train_test_split(master_df, test_size=(1-TRAIN_RATIO), random_state=RANDOM_STATE)
    print(f"✅ 분할 완료! 학습용: {len(train_df):,}개, 검증용: {len(valid_df):,}개")
    
    print("\n[4/4] 최종 데이터셋 폴더를 생성하고 이미지를 복사합니다...")
    # 학습 데이터 폴더 생성 및 이미지/라벨 복사
    # setup_split_folder에 source_base_root (all_data)를 전달
    setup_split_folder(train_df, TRAIN_DIR, SOURCE_BASE_ROOT) 
    # 검증 데이터 폴더 생성 및 이미지/라벨 복사
    setup_split_folder(valid_df, VALID_DIR, SOURCE_BASE_ROOT)

    # --- 최종 검증 단계 ---
    print("\n" + "="*50)
    print("        분할된 데이터셋의 정합성을 최종 검증합니다")
    print("="*50)
    train_ok = verify_split_folder_final(TRAIN_DIR, len(train_df))
    valid_ok = verify_split_folder_final(VALID_DIR, len(valid_df))
    
    print("\n" + "="*50)
    print("                 최종 결과 요약")
    print("="*50)
    if train_ok and valid_ok:
        print("🎉 완벽합니다! 학습 및 검증 데이터셋이 오류 없이 생성되었습니다.")
        print(f"이제 생성된 `final_dataset` 폴더의 데이터를 기반으로 LMDB를 생성할 수 있습니다.")
        print(f"LMDB 생성 시 `--character` 인자로 다음 문자열을 사용하세요: \"{ALLOWED_CHARS}\"")
    else:
        print("❗️ 데이터 분할 결과에 문제가 있습니다. 위의 로그를 확인해주세요.")
    print("="*50)

if __name__ == '__main__':
    main()