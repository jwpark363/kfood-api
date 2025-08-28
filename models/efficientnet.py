# efficientent.py: 모델 아키텍처 정의 (GradCAM에 필요)
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# --- 모델 정의 (사용자 제공 코드 기반) ---
# 이 클래스 정의는 학습 시 사용된 것과 정확히 일치해야 합니다.
# torch.hub.load에서 불러올 수 없으므로, 직접 정의합니다.

class EfficientNetV2S(nn.Module):
    """
    사용자가 정의한 EfficientNetV2S 모델 클래스.
    """
    def __init__(self, num_classes: int = 149, pretrained: bool = False):
        super().__init__()
        # 사전 학습 가중치 없이 모델 구조만 불러옵니다.
        self.model = efficientnet_v2_s(weights=None)

        # 마지막 분류 레이어의 입력 피쳐(in_features)를 가져옵니다.
        in_features = self.model.classifier[1].in_features
        
        # 새로운 분류기(classifier) 레이어를 정의하고 교체합니다.
        # 이 구조는 학습 시 사용한 것과 동일해야 합니다.
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features // 4, num_classes),
        )
        
    def forward(self, x):
        return self.model(x)

# class EfficientNetV2S(nn.Module):
#     def __init__(self, num_classes: int = 149, pretrained: bool = False):   # 음식 클래스 수
#         super().__init__()
#         weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
#         in_features = self.classifier[1].in_features
#         self.classifier = nn.Sequential(
#             # 첫 번째 선형레이어: in_features -> 중간크기1
#             nn.Linear(in_features, in_features // 2),
#             nn.ReLU(),
#             nn.Dropout(p=0.2), # 과적합 방지를 위한 드롭 아웃 추가
#             # 두 번째 선형레이어: 중간크기1 -> 중간크기2
#             nn.Linear(in_features // 2, in_features // 4),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),  # 추가적인 드롭 아웃
#             # 세 번째 선형레이어: 중간크기2 -> 최종클래스 개수
#             nn.Linear(in_features // 4, num_classes)
#         )
#         # self.backbone = efficientnet_v2_s(weights=weights)  # pretrained은 학습 시점에 적용
#         # in_features = self.backbone.classifier[1].in_features
#         # self.backbone.classifier = nn.Sequential(
#         #     nn.Linear(in_features, in_features // 2),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.2),
#         #     nn.Linear(in_features // 2, in_features // 4),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.2),
#         #     nn.Linear(in_features // 4, num_classes),
#         # )

#     def forward(self, x):
#         return self.backbone(x)