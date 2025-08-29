import argparse
import datetime
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
from torchvision.models import resnet34, ResNet34_Weights
# import torchvision.transforms as T
from sklearn.metrics import f1_score, classification_report
import tqdm

from tensorboardX import SummaryWriter
# from torchvision import modelsDEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = f"tensorboard/Vgg16/Vgg16_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

torch.cuda.set_device(1)          # cuda 1
print(DEVICE.type)

###### 인자(옵션) 설정
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default="none",
                    choices=["none", "weights", "checkpoint"],
                    help="학습 재개 모드 선택 (none=새로학습, weights=가중치만 불러서 학습시작, checkpoint=체크포인트 이어서 학습)")
args = parser.parse_args()

###### 하이퍼파라미터(hyper parameters) 및 경로설정 ########
DATA_DIR = "kfood_processed"  # train/val/test 폴더 루트
SAVE_DIR = "models"
os.makedirs("models", exist_ok=True)

SAVE_PATH = "models/resnet34_test_0825_1.pth" # 모델 저장 경로
SAVE_LAST_ALL_PATH = "models/resnet34_last_all_model_0825_1.pth"
BEST_PATH = "models/best_resnet34_test_0825_1.pth" # 최적 모델 저장 경로
CLASS_MAPPING_PATH = "class_mapping.json"   # 클래스 인덱스 매핑
CKPT_PATH = "models/checkpoint_resnet34_0825.pth" # 전체 상테 저장(재개용)

RESUME_WEIGHTS = "models/resnet34_last_all_model_0824_2"  # 이어서 학습할 가중치
RESUME_CKPT = CKPT_PATH

SAVE_LAST_PATH="models/resnet34_last_model_0824_3.pth"

START_EPOCH = 0
EPOCHS = 10
BATCH_SIZE = 32
# LR = 1e-3
SEED = 42
IMG_SIZE = 224

# Mixup / CutMix 관련
MIXUP_PROB, CUTMIX_PROB = 0.25, 0.25
MIXUP_ALPHA, CUTMIX_ALPHA = 1.0, 1.0
EARLYSTOP_PATIENCE = 20    # EarlyStopping patience

# 파인튜닝 하이퍼파라이터(권장)
HEAD_LR = 1e-3
BACKBONE_LR = HEAD_LR / 10
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS_BASE = 3                        # 기본 워밍업 에폭
ETA_MIN = 1e-6                                # 코사인 최저 lr

SAVE_EVERY = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = f"tensorboard/resnet34/resnet34_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

torch.cuda.set_device(1)

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
# ResNet34 기본 변환 가져오기
weights = ResNet34_Weights.IMAGENET1K_V1
base_transform = weights.transforms()   # Resize/CenterCrop/ToTensor/Normalize(이미지넷 정규화)

#  option: 잘못된 이미지 파일을 처리하기 위한 Dataset 커스텀 클래스
class KFoodDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.transforms = transforms
        self.image_paths = glob.glob(os.path.join(root_dir, '**', '*.tif'), recursive=True)
        self.labels = [os.path.basename(os.path.dirname(p)) for p in self.image_paths]
        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.class_to_idx[self.labels[index]]
        
        try:
            image = Image.open(image_path).convert('RGB')   # ResNet은 RGB 3채널 입력
            if self.transforms:
                image = self.transforms(image)
        except Exception as e:
            print(f"Error loading image: {image_path}, Reason: {e}")
            return None, None
        
        return image, label

# 학습 데이터에 사용할 변환 (데이터 증강 추가)
transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),    # 랜덤 크롭 & 리사이즈
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.RandomRotation(15),  # 회전
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # 밝기/색감 왜곡
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),  # 찌그러짐
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3), # 흐림(블러) 효과
    transforms.RandomGrayscale(p=0.1), # 흑백 변환 (가끔)
    base_transform  # 기본 변환
])

# 테스트 데이터에 사용할 변환(Val/Test: 증강 X)
transforms_test = base_transform


# 잘 적용되는지 확인
print(transforms_train)

train_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transforms_train)
val_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transforms_test)
test_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transforms_test)

train_dataLoader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
val_dataLoader = DataLoader(val_datasets, batch_size=BATCH_SIZE, shuffle=False)
test_dataLoader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False)

# 클래스 이름
class_names = train_datasets.classes
num_classes = len(class_names)
print(f"데이터셋 준비완료: {num_classes}개 클래스")

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

###### 모델 정의 (Resnet34 파인튜닝) #######
model = resnet34(weights=weights)
# num_classes = 149
# 원래 마지막 분류(fc)의 입력 피쳐 수
in_features = model.fc.in_features 

# 새로운 분류기 정의 (Sequential 활용)
model.fc = nn.Sequential(
    nn.Linear(in_features, in_features // 2),  # 1단계 축소(in__features -> 중간 크)
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features // 2, num_classes)  # 최종 클래스 수로 출력
)

model.to(DEVICE)

# 교체된 분류기 출력 확인
print(model.fc)

# 손실 함수와 최적화 도구 설정
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer (head와 backbone 구분)
optimizer = optim.AdamW([
    {"params": model.fc.parameters(), "lr": HEAD_LR},            # head (분류기)
    {"params": model.layer1.parameters(), "lr": BACKBONE_LR},    # backbone 일부
    {"params": model.layer2.parameters(), "lr": BACKBONE_LR},
    {"params": model.layer3.parameters(), "lr": BACKBONE_LR},
    {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
], weight_decay=WEIGHT_DECAY)

# LR 스케줄러
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# AMP 스케일러
scaler = amp.GradScaler()

########## 재개 로직 #############
start_epoch, best_f1 = 0, 0.0
# 가중치만 불러오기
if args.resume == "weights" and os.path.exists(RESUME_WEIGHTS):
    print(f"Pretrained weights 불러오기: {RESUME_WEIGHTS}")
    model.load_state_dict(torch.load(RESUME_WEIGHTS, map_location=DEVICE, weights_only=False))
# 체크포인트 불러오기
elif args.resume == "checkpoint" and os.path.exists(RESUME_CKPT):
    print(f"checkpoint 불러오기: {RESUME_CKPT}")
    
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
    print("Checkpoint keys:", ckpt.keys())

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt["epoch"] + 1
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
            writer.add_scalar("Loss/train(Resnet34)", train_loss, epoch)
            if train_acc is not None: writer.add_scalar("Acc/train(Resnet34)", train_acc, epoch)
            if train_f1 is not None: writer.add_scalar("F1/train(Resnet34)", train_f1, epoch)
            if val_loss is not None: writer.add_scalar("Loss/val(Resnet34)", val_loss, epoch)
            writer.add_scalar("Acc/val(Resnet34)", val_acc, epoch)
            writer.add_scalar("F1/val(Resnet34)", val_f1, epoch)
        
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
        if (EPOCHS + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"models/model_resnet34_0824_epoch{epoch+1}.pth")

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

######## 평가 함수 ##########
def evaluate(model, loader, use_amp=True):

    model.eval()
    y_true, y_pred = [], []
    loss_sum, n_batches = 0.0, 0

    # torch.no_grad(): 평가 시 그래디언트 비활성화(메모리/속도)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # AMP: 자동 혼합 정밀도 (추론에도 사용 가능)
            if use_amp and torch.cuda.is_available():
                with amp.autocast(device_type =DEVICE.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels) if criterion is not None else None
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) if criterion is not None else None
            
            # 손실 누적있을 때만
            if loss is not None:
                loss_sum += loss.item()
                n_batches += 1

            # 예측값(top-1). softmax 불필요(순위만 보면 됨)
            preds = outputs.argmax(dim=1)

            # 리스트로 수집(넘파이 변환/CPU 이동)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

            # 지표 계산(배치 단위가 아니라 전체 집계 후 단 한 번 계산하는게 정확함)
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
torch.save(model, "models/all_Resnet34_0824_3.pth")

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
