import os
import json
import matplotlib.pyplot as plt
import random, numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm

from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision
from torchvision import datasets
from torchvision.transforms import Compose
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Normalize
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as T
from sklearn.metrics import f1_score, classification_report
import tqdm

from tensorboardX import SummaryWriter
writer = SummaryWriter()

from torchvision import models

# 하이퍼파라미터 및 경로설정
DATA_DIR = "kfood_processed"  # train/val/test 폴더 루트
os.makedirs("models", exist_ok=True)
SAVE_PATH = "models/vgg16_test_0822.pth" # 모델 저장 경로
BEST_PATH = "models/best_vgg16_test_0822.pth" # 최적 모델 저장 경로
CLASS_MAPPING_PATH = "class_mapping.json"   # 클래스 인덱스 매핑

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
PATIENCE = 5
SEED = 42
IMG__SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 재현성(Seed) 고정
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deteministic = True
torch.backends.cudnn.benchmark = False

# 데이터 전처리(transforms)
# vgg16 모델의 기본 변환 가져오기
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

weights = VGG16_Weights.IMAGENET1K_V1
base_transform = weights.transforms()   # Resize/CenterCrop/ToTensor/Normalize(이미지넷 정규화)

# 학습 데이터에 사용할 변환 (데이터 증강 추가)
transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),    # 랜덤 크롭
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.RandomRotation(15),  # 회전
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # 밝기/색감 왜곡
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),  # 찌그러짐
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3), # 흐림 효과
    transforms.RandomGrayscale(p=0.1), # 흑백 변환 (가끔)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# 테스트 데이터에 사용할 변환
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# 잘 적용되는지 확인
print(transforms_train)

# 데이터셋, 데이터 불러오기 (ImageFolader 사용)
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

# 모델 정의 (vgg16 파인튜닝)
model = vgg16(weights=weights)
# num_classes = 149
# 마지막 분류 레이어의 입력 피쳐(in_features) 확인
in_features = model.classifier[6].in_features

# 새로운 classifier 정의
# nn.Sequential을 사용하여 여러 레이어를 순차적으로 구성
model.classifier = nn.Sequential(
    # 첫 번째 선형레이어: in_features -> 중간크기
    nn.Linear(in_features, in_features // 2),
    nn.ReLU(),
    nn.Dropout(p=0.2), # 과적합 방지를 위한 드롭 아웃 추가
    # 두 번째 선형레이어: 중간크기 -> 최종클래스 개수
    nn.Linear(in_features // 2, num_classes)
)
model.to(DEVICE)
print(model.classifier)


# 손실 함수와 최적화 도구 설정
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# 훈련
def train_model(model, train_loader, val_loader, epochs):
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, total, correct = 0, 0, 0
        y_true, y_pred = [], []

        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_f1 = f1_score(y_true, y_pred, average="macro")

        # 검증 단계
        val_acc, val_f1 = evaluate(model, val_loader)
        
        if (epoch) % 2 == 0:
            save_name = f"vgg_16_test_0822_{epoch}.pth"
            torch.save(model.state_dict(), save_name)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} "
              f"| Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} "
              f"| Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Best checkpoint 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), BEST_PATH)

        scheduler.step()

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1

# 학습 실행
train_model(model, train_dataLoader, val_dataLoader, EPOCHS, mix_strategy="cutmix")  # 또는 "mixup"

torch.save(model.state_dict(), SAVE_PATH)
torch.save(model, "models/vgg_test_0822.pth")

y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_dataLoader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=train_datasets.classes))