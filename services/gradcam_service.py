# gradcam_service.py: GradCAM 로직 실행

import os
import time
from typing import Optional, Dict

from PIL import Image
import torch

from utils.gradcam import GradCAM, overlay_heatmap_on_image, pil_to_base64


class GradCAMService:
    """
    - 모델과 전처리 함수를 받아 한 번만 초기화
    - 요청마다 Grad-CAM을 생성하여 파일 및 base64로 반환
    """
    def __init__(self, model: torch.nn.Module, preprocess, static_dir: str = "static"):
        self.model = model
        self.preprocess = preprocess
        self.static_dir = static_dir
        os.makedirs(self.static_dir, exist_ok=True)

        # GradCAM 인스턴스(타깃 레이어 자동 탐색)
        self.cam = GradCAM(self.model)

    def run(self, image_pil: Image.Image, target_category: Optional[int] = None, alpha: float = 0.4) -> Dict:
        """
        Returns:
          {
            "target": int,
            "image_path": str,      # 저장된 파일 경로
            "image_base64": str,    # base64 PNG
          }
        """
        # 1) 전처리 + 배치 차원
        input_tensor = self.preprocess(image_pil).unsqueeze(0).to(next(self.model.parameters()).device)

        # 2) forward (예측)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_idx = int(logits.argmax(dim=1).item())

        target = pred_idx if target_category is None else target_category

        # 3) Grad-CAM (grad 필요!)
        #   주의: CAM 계산은 grad가 필요하므로 enable_grad로 다시 계산
        input_tensor.requires_grad_(True)
        heatmap = self.cam(input_tensor, target_category=target)  # (H', W') in [0,1]

        # 4) 오버레이 생성
        overlay = overlay_heatmap_on_image(heatmap, image_pil=image_pil, alpha=alpha)

        # 5) 파일 저장 + base64
        ts = int(time.time() * 1000)
        filename = f"gradcam_{ts}.png"
        out_path = os.path.join(self.static_dir, filename)
        overlay.save(out_path)

        return {
            "target": target,
            "image_path": out_path,
            "image_base64": pil_to_base64(overlay),
        }

# from utils.gradcam import GradCAM, overlay_heatmap
# from services.model_service import ModelService, DEVICE
# from PIL import Image
# import torch

# # target_layer: EfficientNetV2 마지막 Conv 블록
# target_layer = ModelService.backbone.features[-1]
# gcam = GradCAM(ModelService, target_layer)

# def generate_gradcam(image: Image.Image, class_idx=None):
#     # 입력 준비
#     from services.model_service import transform
#     tensor = transform(image).unsqueeze(0).to(DEVICE)

#     cam = gcam(tensor, class_idx)
#     overlay = overlay_heatmap(image, cam, alpha=0.6, input_size=224)
#     return overlay