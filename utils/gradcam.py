#############################################################################################
##############################     Grad-CAM 유틸      #######################################
#  * 마지막 Conv 블록의 활성값(activation)과 그라디언트(gradient)를 이용해 CAM(히트맵)을 만든다.
#  * 안전 포인트!
#    - "요청마다" 훅을 등록/제거하여, 웹서버 동시성에서 값이 섞이지 않도록 한다.
#  * 사용 흐름
#    - 1) GradCAM(model, target_layer) # 인스턴트 생성
#    - 2) cam = gcam(x, class_idx=None) # x: (1, C, H, M), class_idx 생략 시 예측 top-1
#    - 3) overlay = overlay_heatmap(original_pil_image, cam)
#  * 주의
#    - Grad-CAM 계산에는 backward()가 필요하므로 torch.no_grad() 금지
#    - 배치 사이즈는 1이 가장 안전(여러 장 처리는 별도 코드 필요)
#  target_layer에 마지막 Conv 블록 모듈을 전달
#  __call__(tensor, class_idx) -> CAM(0~1)
#  overlay(image, cam) -> 원본+히트맵 합성 이미지
#############################################################################################

from __future__ import annotations
import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple



def _find_last_conv_layer(module: nn.Module) -> nn.Module:
    last_conv = None
    for _, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            last_conv = child
        else:
            cand = _find_last_conv_layer(child)
            if cand is not None:
                last_conv = cand
    return last_conv

# * Gard-CAM 계산기 클래스
#  - model: 학습 완료된 분류 모델 (eval() 권장)
#  - target_layer: 마지막 Conv블록(또는 Conv가 포함된 모듈). 여기가 Cam을 만들 "특징맵" 위치
#  ex) torchvision EfficientNetV2-S: model.features[-1]
#      ResNet-34(기본블록): model.layer4[-1].conv2
#      ResNet-50(병목블록): model.layer4[-1].conv3
#      ConvNeXt-Tiny: model.features[-1].block[-1].pwconv2 (모델마다 다름, 마지막 conv 유사 지점)
class GradCAM:   

    def __init__(self, model:torch.nn.Module, target_layer: torch.nn.Module):
    
        self.model = model
        self.target_layer = target_layer
        self.model.eval()   # 드롭아웃/배치정규화를 평가모드로

    # 요청(호출) 단위로 forward/backward 훅을 만들고 등록한 뒤, 사용 후 즉시 제거
    #  - x: (1, C, H, W) 모델과 같은 디바이에에스에 있어야 함(ex: cuda면 둘 다 cuda)
    #  - class_idx: CAM을 만들 타깃 클래스 인덱스 (Nome이면 예측 확률이 가장 큰 클래스로)
    # 반환: 
    #  - activations(act): target_layer의 활성값(activation) 텐서
    #  - gradients(grad): 위 활성값에 대한 dL/dA 그라디언트 텐서
    #  - chosen_class_idx(class_idx): 최종 사용한 클래스 인덱스(int)
    def _register_hooks_and_forward(self, x: torch.Tensor, class_idx: Optional[int]):

        activations = {}
        gradients = {}

        # forward hook: 순전파 중 targe_layer의 출력(특징맵)을 저장
        def _forward_hook(module, input, output):
            # output: Tensor (B, C, H, W)
            activations['value'] = output.detach()
        
        # backward hook: 뎍전파 중 target_layer 출력에 대한 그라디언트 저장
        def _backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: (B, C, H, W)
            gradients['value'] = grad_output[0].detach()
        
        # 훅 등록(핸들 저장)
        fh = self.target_layer.register_forward_hook(_forward_hook)
        # Pytorch 버전에 따라 fill_backward_hook 지원이 다름 + 지원 시 우선 사용, 아니면 기존 backward_hook로 폴백
        try:
            bh = self.target_layer.register_full_backward_hook(_backward_hook)
        except Exception:
            bh = self.target_layer.register_backward_hook(_backward_hook) 
        
        ### 순전파(forward) ###
        self.model.zero_grad(set_to_none=True)  # 이전 그래디언트 초기화 
        outputs = self.model(x) # 여기서는 torch.no_grad() 쓰면 안됨 (뒤에 backward 필요)

        # class_idx가 없으면 예측 최대 확률 클래스를 사용
        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1).item())
        
        ### 역전파(backward: 해당 클래스 스칼라 로스를 만들어 그라디언트 계산 ###
        loss = outputs[0, class_idx]
        loss.backward()

        # 훅으로 캡쳐한 활성값/그라디언트 꺼내기
        act = activations.get("value", None)
        grad = gradients.get("value", None)

        # 훅 제거(매모리릭(memory leak)/동시성 오염(cross-request contamination) 방지)
        try:
            fh.remove()
        except Exception:
            pass

        try:
            bh.remove()
        except Exception:
            pass

        return act, grad, int(class_idx)

    # CAM을 계산해 (Hf, Wf)의 2D numpy 배열(0~1 정규화)을 반환
    #  - x: (1, C, H, W) 1장만 넣는 것을 권장
    #  class_idx: None이면 예측 top-1
    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:

        assert x.dim() == 4 and x.size(0) == 1, "Grad-CAM 입력은 배치 크기 1을 권장"

        # 훅 등록->forward/backward->훅 제거를 한 번에 수정
        act, grad, class_idx = self._register_hooks_and_forward(x, class_idx)

        if act is None or grad is None:
            raise RuntimeError("Grad-CAM: 활성값 또는 그라디언트 캡처에 실패했습니다.")
        
        # 채널 가중치: 그라디언트를 H, W 평균(GAP)하여 (1, C, 1, 1) 형태로
        weights = grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # 가중합으로 CAM 생성: (1, 1, Hf, Wf)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)   # 음수는 0으로 (ReLU)

        # numpy로 변환 + [0, 1] 정규화
        cam = cam.squeeze().cpu().numpy()   # (Hf, Wf)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

# 원본 PIL 이미지에 CAM 히트맵을 오버레이해서 PIL 이미지로 반환
# - pil_img: RGB
# - cam: (h, w) normalized [0, 1]
def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.5, input_size: int = 224) -> Image.Image:

    # 1) 원본을 모델 입력 크기(예: 224x224)로 맞춥니다.
    base = pil_img.convert("RGB").resize((input_size, input_size))
    base_np = np.array(base)  # (H, W, 3)

    # 2) CAM(특징맵 크기)을 베이스 크기와 동일하게 리사이즈
    cam_resized = cv2.resize(cam.astype(np.float32), (base_np.shape[1], base_np.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 3) [0,1] → [0,255]로 변환 후 컬러맵 적용(JET)
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # (H,W,3) BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)       # RGB로 변환

    # 4) 원본과 히트맵을 가중합(overlay)
    overlay = cv2.addWeighted(base_np, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlay)
