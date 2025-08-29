import argparse
import os, random, re, json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import glob

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau 
import tqdm

from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Normalize
from torchvision.models import vgg16, VGG16_Weights
# import torchvision.transforms as T
from sklearn.metrics import f1_score, classification_report
import tqdm
import datetime
from tensorboardX import SummaryWriter
# from torchvision import models

###### 인자(옵션) 설정
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default="none",
                    choices=["none", "weights", "checkpoint"],
                    help="학습 재개 모드 선택 (none=새로학습, weights=가중치만 불러서 학습시작, checkpoint=체크포인트 이어서 학습)")
parser.add_argument("--epochs", type=int, default=None, help="총 목표 에폭(재시작 포함)")
parser.add_argument("--extra-epochs", type=int, default=None, help="체크포인트에서 추가로 돌릴 에폭")
args = parser.parse_args()

###### 하이퍼파라미터(hyper parameters) 및 경로설정 ########
# LOGDIR = f"tensorboard/vgg/vgg_20250825_1710"
DATA_DIR = "kfood_processed"  # train/val/test 폴더 루트
SAVE_DIR = "models2"
os.makedirs("models2", exist_ok=True)

SAVE_PATH = "models2/Vgg16_test_0825_1.pth" # 모델 저장 경로
SAVE_LAST_ALL_PATH = "models2/Vgg16_last_all_model_0825_1.pth"
BEST_PATH = "models2/best_Vgg16_test_0825_1.pth" # 최적 모델 저장 경로
CLASS_MAPPING_PATH = "class_mapping.json"   # 클래스 인덱스 매핑
CKPT_PATH = "models2/checkpoint_Vgg16_0825.pth" # 전체 상테 저장(재개용)

RESUME_WEIGHTS = "models2/best_Vgg16_test_0824_1.pth"  # 이어서 학습할 가중치
RESUME_CKPT = CKPT_PATH

SAVE_LAST_PATH="models2/Vgg16_last_model_0825_1.pth"

START_EPOCH = 0
EPOCHS = 20
if args.epochs is not None:
    EPOCHS = args.epochs
BATCH_SIZE = 32
# LR = 1e-3
SEED = 42
IMG_SIZE = 224

# Mixup / CutMix 관련 , 0825, Vgg는 잠깐 꺼야 함(데이터가 안정화 되기 전엔 신호가 희석되서 학습이 안 붙음.)
# MIXUP_PROB, CUTMIX_PROB = 0.25, 0.25
MIXUP_ALPHA, CUTMIX_ALPHA = 1.0, 1.0
EARLYSTOP_PATIENCE = 10    # EarlyStopping patience

# AFTER
MIXUP_PROB, CUTMIX_PROB = 0.0, 0.0
criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

# 파인튜닝 하이퍼파라이터(권장)
HEAD_LR = 1e-3
BACKBONE_LR = HEAD_LR / 10
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS_BASE = 3                        # 기본 워밍업 에폭
ETA_MIN = 1e-6                                # 코사인 최저 lr

SAVE_EVERY = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = f"tensorboard/Vgg16/Vgg16_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"


LOG_ROOT = "tensorboard"  # 상위 폴더 하나로
RUN_NAME = f"Vgg16/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR  = os.path.join(LOG_ROOT, RUN_NAME)
print("TensorBoard log dir:", os.path.abspath(LOG_DIR))

torch.cuda.set_device(1)          # cuda 1
print(DEVICE.type)

# 재현성(Seed) 고정
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

####### 데이터 전처리(transforms) ########
# Vgg16 모델의 기본 변환 가져오기
# weights = VGG16_Weights.IMAGENET1K_V1
# base_transform = weights.transforms()   # Resize/CenterCrop/ToTensor/Normalize(이미지넷 정규화)

# weights = VGG16_Weights.IMAGENET1K_V1
# mean, std = weights.meta["mean"], weights.meta["std"]

# weights = VGG16_Weights.IMAGENET1K_V1  # 모델엔 그대로 사용

# # torchvision 버전 상관없이 안전한 ImageNet 정규화 상수
# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)

# # weights.meta가 없거나 키가 없으면 위 상수로 fallback
# meta = getattr(weights, "meta", None)
# if isinstance(meta, dict) and ("mean" in meta) and ("std" in meta):
#     mean, std = meta["mean"], meta["std"]
# else:
#     mean, std = IMAGENET_MEAN, IMAGENET_STD



# IMG_SIZE = 224  # 위의 값과 일치

# def ensure_rgb(im: Image.Image) -> Image.Image:
#     # RGBA → 흰 배경 합성, 팔레트/그레이 → RGB 변환
#     if im.mode == "RGBA":
#         bg = Image.new("RGB", im.size, (255, 255, 255))
#         bg.paste(im, mask=im.split()[-1])
#         return bg
#     return im.convert("RGB") if im.mode != "RGB" else im

# to_rgb = transforms.Lambda(ensure_rgb)

# # 학습
# transforms_train = transforms.Compose([
#     to_rgb,
#     transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
#     # 초기엔 과한 증강은 잠시 끔 (붙으면 다시 켜세요)
#     # transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
#     # transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
#     # transforms.RandomGrayscale(p=0.1),
#     transforms.ToTensor(),             # ★ PIL → Tensor 강제
#     transforms.Normalize(mean, std),
# ])

# # 검증/테스트
# transforms_test = transforms.Compose([
#     to_rgb,
#     transforms.Resize(256),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.ToTensor(),             # ★ PIL → Tensor 강제
#     transforms.Normalize(mean, std),
# ])


# ===== 데이터 전처리: PIL → Tensor 보장 =====
####### 0825 데이터 전처리: base_transform 쓰지 않는 버전 ########
from PIL import ImageFile, Image
from torchvision.transforms import functional as TF
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 깨진 TIFF 허용

weights = VGG16_Weights.IMAGENET1K_V1  # 모델 weights는 그대로 사용
# torchvision 버전과 무관하게 안전한 ImageNet 상수
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_SIZE = 224  # 위 설정과 일치

def ensure_rgb(im: Image.Image) -> Image.Image:
    """RGBA → 흰 배경 합성, 팔레트/그레이 → RGB 변환"""
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB") if im.mode != "RGB" else im

class EnsureTensorAndNormalize:
    """변환 파이프라인 끝에서 무조건 Tensor+Normalize 보장"""
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = TF.to_tensor(ensure_rgb(x))     # [0,1] float tensor
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype != torch.float32:
                x = x.float()
            # ndarray가 0~255면 0~1로
            if x.max() > 1.0:
                x = x / 255.0
        assert isinstance(x, torch.Tensor), f"Transform must return Tensor, got {type(x)}"
        return TF.normalize(x, self.mean, self.std)

to_rgb = transforms.Lambda(ensure_rgb)
to_tensor_norm = EnsureTensorAndNormalize(IMAGENET_MEAN, IMAGENET_STD)

# --- 학습 ---
transforms_train = transforms.Compose([
    to_rgb,
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    # 붙을 때까지 과한 증강은 잠시 OFF:
    # transforms.RandomPerspective(0.4, p=0.5),
    # transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    # transforms.RandomGrayscale(p=0.1),
    to_tensor_norm,  # ★ 마지막에 무조건 Tensor+Normalize
])

# --- 검증/테스트 ---
transforms_test = transforms.Compose([
    to_rgb,
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    to_tensor_norm,  # ★ 마지막에 무조건 Tensor+Normalize
])




# 이미지 파일 오류 로그 찍기
class KFoodDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.transforms = transforms
        self.image_paths = glob.glob(os.path.join(root_dir, '**', '*.tif'), recursive=True)
        # 이미지 파일명에서 클래스 레이블 추출
        self.labels = [os.path.basename(os.path.dirname(p)) for p in self.image_paths]
        # 클래스 목록을 정렬하여 저장
        self.classes = sorted(list(set(self.labels)))
        # 클래스 이름을 인덱스로 매핑
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.class_to_idx[self.labels[index]]
        
        try:
            # 파일 로드
            image = Image.open(image_path)
            # 팔레트 이미지 경고를 피하기 위해 RGBA로 변환
            image = image.convert('RGBA')       
            # transforms가 있다면 적용
            if self.transforms:
                image = self.transforms(image)
        
        except Exception as e:
            # Truncated File Read 등의 오류 발생 시 실행되는 부분
            # 문제가 되는 파일의 경로를 출력
            print(f"Error loading image: {image_path}, Reason: {e}")
            return None, None
        
        return image, label
    
# 이미지를 RGBA로 변환하는 커스텀 변환
class ConvertToRGBA:
    def __call__(self, img):
        # PIL Image 객체에 대해 .convert("RGBA")를 호출
        return img.convert("RGBA")

# # 학습 데이터에 사용할 변환 (데이터 증강 추가) ,0825 이것도 증강을 추가하므로 일단 제외
# transforms_train = transforms.Compose([
#     # ConvertToRGBA(),
#     transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),    # 랜덤 크롭
#     transforms.RandomHorizontalFlip(),  # 좌우 반전
#     transforms.RandomRotation(15),  # 회전
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # 밝기/색감 왜곡
#     transforms.RandomPerspective(distortion_scale=0.4, p=0.5),  # 찌그러짐
#     transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3), # 흐림 효과
#     transforms.RandomGrayscale(p=0.1), # 흑백 변환 (가끔)
#     # base_transform
# ])

# # 테스트 데이터에 사용할 변환(Val/Test: 증강 X)
# transforms_test = transforms.Compose([
#     # ConvertToRGBA(), 
#     # base_transform
# ])

# 잘 적용되는지 확인
print(transforms_train)

# --- 데이터셋 만들기 ---
train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")
test_dir  = os.path.join(DATA_DIR, "test")

train_datasets = datasets.ImageFolder(train_dir, transform=transforms_train)
val_datasets   = datasets.ImageFolder(val_dir,   transform=transforms_test)
test_datasets  = datasets.ImageFolder(test_dir,  transform=transforms_test)

# --- 라벨 인덱스 통일: train 기준으로 강제 매핑 ---
train_c2i = train_datasets.class_to_idx
train_classes = set(train_c2i.keys())

missing_val  = set(val_datasets.classes)  - train_classes
missing_test = set(test_datasets.classes) - train_classes
if missing_val:
    print("[WARN] val에 train에 없는 클래스:", sorted(list(missing_val)))
if missing_test:
    print("[WARN] test에 train에 없는 클래스:", sorted(list(missing_test)))

def make_target_mapper(src_c2i, dst_c2i):
    idx2class = {v: k for k, v in src_c2i.items()}
    def _map(i):
        cls = idx2class[i]
        return dst_c2i[cls]   # KeyError면 해당 클래스 폴더를 train에도 만들어 두세요(비어 있어도 됨)
    return _map

val_datasets.target_transform  = make_target_mapper(val_datasets.class_to_idx,  train_c2i)
test_datasets.target_transform = make_target_mapper(test_datasets.class_to_idx, train_c2i)

class_names = train_datasets.classes
num_classes = len(class_names)
print(f"[OK] 통일된 클래스 개수: {num_classes}")

# --- DataLoader ---
train_dataLoader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
val_dataLoader   = DataLoader(val_datasets,   batch_size=BATCH_SIZE, shuffle=False)
test_dataLoader  = DataLoader(test_datasets,  batch_size=BATCH_SIZE, shuffle=False)




# # os.path.join(DATA_DIR, "train") 경로가 이미지를 포함하고 있는지 확인합니다.
# train_datasets = ImageFolder(os.path.join("kfood_processed", "train"), transform=transforms_train)

# # 데이터셋의 길이를 출력합니다.
# print(f"Number of samples in the training dataset: {len(train_datasets)}")

# # 만약 이 코드를 실행했을 때 0이 출력된다면, 여전히 ImageFolder가
# # 경로에서 이미지 파일을 찾지 못했다는 뜻입니다.
# if len(train_datasets) == 0:
#     print("Warning: The training dataset is empty. Please check your data path and file types.")

# train_dataLoader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)

# # 데이터셋의 길이를 출력합니다.
# print(f"Number of samples in the training dataset: {len(train_datasets)}")



# 데이터셋, 데이터 불러오기 (ImageFolader 사용)
# train_datasets = KFoodDataset(root_dir=os.path.join(DATA_DIR, "train"), transforms=transforms_train)
# val_datasets = KFoodDataset(root_dir=os.path.join(DATA_DIR, "val"), transforms=transforms_test)
# test_datasets = KFoodDataset(root_dir=os.path.join(DATA_DIR, "test"), transforms=transforms_test)

# train_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transforms_train)
# val_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transforms_test)
# test_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transforms_test)

# # --- 라벨 인덱스 통일: train 기준으로 강제 매핑 ---   0825 여기가 위치가 안맞음 
# train_c2i = train_datasets.class_to_idx
# train_classes = set(train_c2i.keys())

# missing_val  = set(val_datasets.classes)  - train_classes
# missing_test = set(test_datasets.classes) - train_classes
# if missing_val:
#     print("[WARN] val에 train에 없는 클래스:", sorted(list(missing_val)))
# if missing_test:
#     print("[WARN] test에 train에 없는 클래스:", sorted(list(missing_test)))

# def make_target_mapper(src_c2i, dst_c2i):
#     idx2class = {v: k for k, v in src_c2i.items()}
#     def _map(i):
#         cls = idx2class[i]
#         return dst_c2i[cls]   # KeyError나면 해당 클래스 폴더를 train에도 만들어주세요(비어있어도 됨)
#     return _map

# val_datasets.target_transform  = make_target_mapper(val_datasets.class_to_idx,  train_c2i)
# test_datasets.target_transform = make_target_mapper(test_datasets.class_to_idx, train_c2i)

# class_names = train_datasets.classes
# num_classes = len(class_names)
# print(f"[OK] 통일된 클래스 개수: {num_classes}")

# train_dataLoader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
# val_dataLoader = DataLoader(val_datasets, batch_size=BATCH_SIZE, shuffle=False)
# test_dataLoader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False)

# 클래스 이름
# class_names = train_datasets.classes
# num_classes = len(class_names)
# print(f"데이터셋 준비완료: {num_classes}개 클래스")

####### Mixup & CutMix 함수 ########

# CutMix: 랜덤 박스 좌표 생성
def rand_bbox(size, lam):

    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

# Mixup: 두 이미지와 라벨을 섞음
def mixup_data(x, y, alpha=1.0):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# CutMix: 이미지 일부를 잘라서 다른 이미지 붙이기
def cutmix_data(x, y, alpha=1.0):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam

###### 모델 정의 (Vgg16 파인튜닝) #######
model = vgg16(weights=weights)
# num_classes = 149
# 마지막 분류 레이어의 입력 피쳐(in_features) 확인
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model.to(DEVICE)
print(model.classifier)

# 손실 함수와 최적화 도구 설정
optimizer = optim.AdamW([
    {"params": model.classifier.parameters(), "lr": HEAD_LR},    # head
    {"params": model.features.parameters(), "lr": BACKBONE_LR},     # backborn
])

optimizer = optim.AdamW([
    {"params": model.classifier.parameters(), "lr": HEAD_LR},    # head
    {"params": model.features.parameters(), "lr": BACKBONE_LR},  # backbone
])

# === NEW: LR 스케줄러 & GradScaler ===
SCHEDULER = "cosine"   # "plateau" 로 바꾸면 ReduceLROnPlateau 사용
if SCHEDULER == "plateau":
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
        min_lr=ETA_MIN, verbose=True
    )
else:
    scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=ETA_MIN
    )

scaler = amp.GradScaler(enabled=(DEVICE.type == "cuda"))

########## 재개 로직 #############
start_epoch, best_f1 = 0, 0.0
## 가중치만 불러오기
if args.resume == "weights" and os.path.exists(RESUME_WEIGHTS):
    print(f"Pretrained weights 불러오기: {RESUME_WEIGHTS}")
    model.load_state_dict(torch.load(RESUME_WEIGHTS, map_location=DEVICE, weights_only=False))
## 체크포인트 불러오기
# elif args.resume == "checkpoint" and os.path.exists(RESUME_CKPT):
#     print(f"checkpoint 불러오기: {RESUME_CKPT}")
#     ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
#     model.load_state_dict(ckpt["model_state"])
#     optimizer.load_state_dict(ckpt["optimizer_state"])
#     scheduler.load_state_dict(ckpt["scheduler_state"])
#     scaler.load_state_dict(ckpt["scaler_state"])
#     start_epoch = ckpt["epoch"] + 1
# # 0824 체크포인트 uv run Vgg16_0824.py --resume checkpoint 안되서 코드 추가
#     START_EPOCH = start_epoch
    
#     best_f1 = ckpt["best_f1"]

elif args.resume == "checkpoint" and os.path.exists(RESUME_CKPT):
    print(f"checkpoint 불러오기: {RESUME_CKPT}")
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt["epoch"] + 1
    START_EPOCH = start_epoch
    best_f1 = ckpt["best_f1"]

    
else:
    print("처음부터 학습 시작!")

######## 훈련 ########
def train_model(model, train_loader, val_loader, use_amp = True):
    # 텐서보드 활용하여, 최적화 과정을 시각화
    writer = SummaryWriter(LOG_DIR)
    best_f1 = 0.0
    patience_counter = 0

    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)

    scalar = amp.GradScaler()

    device_type = DEVICE.type if isinstance(DEVICE, torch.device) else str(DEVICE)
    use_amp = use_amp and (device_type == "cuda")   # AMP는 cuda에서만 유의미

    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        running_loss, total, correct = 0, 0, 0
        batch_count = 0

        y_true, y_pred = [], []

        # mixup/cutmix 적용 배치는 train-accuracy 계산에서 제외(또는 분리해서 계산)
        nonmixed_true, nonmixed_pred = [], []

        pbar = tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()   # 옵티마이저 초기화

            # Mixup 또는 CutMix 적용
            r = float(np.random.rand()) # 스칼라

            mixed = False

            # forward (AMP 지원)
            with amp.autocast(device_type=DEVICE.type):
                if r < MIXUP_PROB:
                    mixed = True
                    mixed_x, targets_a, targets_b, lam = mixup_data(images, labels, alpha=MIXUP_ALPHA)
                    outputs = model(mixed_x)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                elif r < MIXUP_PROB + CUTMIX_PROB:
                    mixed = True
                    mixed_x, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=CUTMIX_ALPHA)
                    outputs = model(mixed_x)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            
            # backward (GradScaler 사용)
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()

            running_loss += loss.item()
            batch_count += 1

            # train metrics: 혼합된 배치는 레이블 해석이 어려우므로 비혼합 배치만 모음
            if not mixed:
                preds = outputs.argmax(dim=1)
                nonmixed_true.extend(labels.detach().cpu().numpy())
                nonmixed_pred.extend(preds.detach().cpu().numpy())
            
            pbar.set_postfix(loss=running_loss / max((1, batch_count)))
        
        # epoch end: train metrics (비혼합 배치만)
        train_loss = running_loss / max(1, batch_count)
        if len(nonmixed_true) > 0:
            train_acc = (np.array(nonmixed_true) == np.array(nonmixed_pred)).mean()
            train_f1 = f1_score(nonmixed_true, nonmixed_pred, average="macro")
        else:
            train_acc = None
            train_f1 = None
        
        ############ Validation 검증단계 ###################
        # evaluate 함수 사용, 예상 반환: dict with acc, f1, loss) 
        # evaluate가 evaluate(model, loader, use_amp)같이 동작
        val_res = evaluate(model, val_dataLoader)
        val_acc = val_res.get("acc")
        val_f1 = val_res.get("f1_macro")
        val_loss = val_res.get("loss")

        # 스케줄러 업데이트
        # ReduceLROnPlateau는 val_loss를 넣어야 함
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # TensorBoard 로그
        if writer is not None:
            writer.add_scalar("Loss/train(Vgg16)", train_loss, epoch)
            if train_acc is not None: writer.add_scalar("Acc/train(Vgg16)", train_acc, epoch)
            if train_f1 is not None: writer.add_scalar("F1/train(Vgg16)", train_f1, epoch)
            if val_loss is not None: writer.add_scalar("Loss/val(Vgg16)", val_loss, epoch)
            writer.add_scalar("Acc/val(Vgg16)", val_acc, epoch)
            writer.add_scalar("F1/val(Vgg16)", val_f1, epoch)
        
        # 출력
        print_str = f"[Epoch {epoch+1}/{EPOCHS}] train_loss={train_loss:.4f}"
        if train_acc is not None: print_str += f" train_acc={train_acc:.4f}"
        if train_f1 is not None: print_str += f" train_f1={train_f1:.4f}"
        print_str += f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        print(print_str)

        # 체크 포인트(전체 상태) 저장: 항상 최신 상태로 덮어쓰기 (재개용)
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_f1": best_f1
        }
        torch.save(ckpt, CKPT_PATH)

        # 주기적으로 가중치만 저장
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"models2/model_Vgg16_epoch{epoch+1}.pth")

        # best model 저장 (가중치만)
        if val_f1 is not None and val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), BEST_PATH)
            patience_counter = 0
            print(f"뉴뉴!! best_f1={best_f1:.4f}, saved to {BEST_PATH}")
        else:
            if EARLYSTOP_PATIENCE is not None:
                patience_counter += 1
                print(f"no improvement for {patience_counter}/{EARLYSTOP_PATIENCE} epoch")
                if patience_counter >= EARLYSTOP_PATIENCE:
                    print("EarlyStopping triggered")
                    break


        # 마지막 가중치 저장
        torch.save(model.state_dict(), SAVE_LAST_PATH)
        torch.save(model, SAVE_LAST_ALL_PATH)

        if writer is not None:
            writer.flush()
    writer.close()        
    return {"best__f1": best_f1, "last_epoch": EPOCHS}

# ######## 평가 함수 ##########  0825 텐서보드가 오류나서 다시 고침
# def evaluate(model, loader, use_amp=True):

#     model.eval()
#     y_true, y_pred = [], []
#     loss_sum, n_batches = 0.0, 0

#     # torch.no_grad(): 평가 시 그래디언트 비활성화(메모리/속도)
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
            
#             # AMP: 자동 혼합 정밀도 (추론에도 사용 가능)
#             if use_amp and torch.cuda.is_available():
#                 with amp.autocast(device_type =DEVICE.type):
#                     outputs = model(images)
#                     loss = criterion(outputs, labels) if criterion is not None else None
#             else:
#                 outputs = model(images)
#                 loss = criterion(outputs, labels) if criterion is not None else None
            
#             # 손실 누적있을 때만
#             if loss is not None:
#                 loss_sum += loss.item()
#                 n_batches += 1

#             # 예측값(top-1). softmax 불필요(순위만 보면 됨)
#             preds = outputs.argmax(dim=1)

#             # 리스트로 수집(넘파이 변환/CPU 이동)
#             y_true.extend(labels.detach().cpu().numpy())
#             y_pred.extend(preds.detach().cpu().numpy())

#             # 지표 계산(배치 단위가 아니라 전체 집계 후 단 한 번 계산하는게 정확함)
#             if len(y_true) == 0:
#                 return {"acc": 0.0, "f1_macro": 0.0, "loss": None if criterion is None else float("nan")}
            
#             y_true_np = np.array(y_true)
#             y_pred_np = np.array(y_pred)

#             acc = (y_true_np == y_pred_np).mean()
#             f1 = f1_score(y_true_np, y_pred_np, average="macro")
#             avg_loss = (loss_sum / max(1, n_batches)) if criterion is not None else None

#             return {"acc": acc, "f1_macro": f1, "loss": avg_loss}

######## 평가 함수 ##########
def evaluate(model, loader, use_amp=True):
    model.eval()
    y_true, y_pred = [], []
    loss_sum, n_batches = 0.0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if use_amp and torch.cuda.is_available():
                with amp.autocast(device_type=DEVICE.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels) if criterion is not None else None
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) if criterion is not None else None

            if loss is not None:
                loss_sum += loss.item()
                n_batches += 1

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    if len(y_true) == 0:
        return {"acc": 0.0, "f1_macro": 0.0, "loss": None if criterion is None else float("nan")}

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    acc = (y_true_np == y_pred_np).mean()
    f1 = f1_score(y_true_np, y_pred_np, average="macro")
    avg_loss = (loss_sum / max(1, n_batches)) if criterion is not None else None
    return {"acc": acc, "f1_macro": f1, "loss": avg_loss}

# 학습 실행
train_model(model, train_dataLoader, val_dataLoader)
torch.save(model.state_dict(), SAVE_PATH)
torch.save(model, "models2/all_0824.pth")

y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_dataLoader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

# classification_report 출력
print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=train_datasets.classes,  # 클래스 이름 매핑
    digits=4                              # 소수점 자
))
