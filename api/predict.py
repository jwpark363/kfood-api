
from fastapi import APIRouter, UploadFile, File
# from services.model_service import predict_image
# from services.gradcam_service import generate_gradcam
from PIL import Image

from services.model_service import ModelService
# from schemas.prediction import PredictionResponse

router = APIRouter()

# 서버 시작 시 1회 초기화 (경로, 클래스 수는 프로젝트에 맞게 조정)
MODEL_PATH = "data/efficientnet_test_0826_1.pth"
CLASS_MAPPING_CSV_PATH = "data/food_class_mapping.csv"
NUM_CLASSES = 149

_model_service = ModelService(MODEL_PATH, num_classes=NUM_CLASSES, class_mapping_csv_path=CLASS_MAPPING_CSV_PATH)
# _cam_service = GradCAMService(_model_service.model, _model_service.preprocess, static_dir="static")
print('model',_model_service)

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    print(image)
    result = _model_service.predict(image)
    print(result)
    return result


# @router.post("/predict-with-cam")
# async def predict_with_cam(file: UploadFile = File(...), alpha: float = 0.40):
#     """
#     반환:
#     - class_id / class_name / probability
#     - gradcam_path: 서버 내 저장된 파일 경로
#     - gradcam_url : 브라우저에서 접근 가능한 URL (/static/...)
#     - gradcam_base64 : 직접 이미지 데이터를 받고 싶을 때(프론트에서 바로 <img src="data:image/png;base64,..." />)
#     """
#     image = Image.open(file.file).convert("RGB")

#     pred = _model_service.predict(image)
#     cam = _cam_service.run(image, target_category=pred["class_id"], alpha=alpha)

#     # static 마운트 기준으로 URL 생성
#     # 파일 경로가 "static/xxx.png" 형태이므로, URL은 "/static/xxx.png"
#     fname = cam["image_path"].split("static/")[-1]
#     url = f"/static/{fname}"

#     return {
#         **pred,
#         "gradcam_path": cam["image_path"],
#         "gradcam_url": url,
#         "gradcam_base64": cam["image_base64"],
#     }

# # 전역 모델 서비스 (서버 시작 시 한 번만 로드)
# model_service = ModelService("data/efficientnet_test_0823_3.pth", num_classes=149)

# @router.post("/predict", response_model=PredictionResponse)
# async def predict(file: UploadFile = File(...)):
#     image = Image.open(file.file).convert("RGB")
#     pred_idx, class_name, probability = predict_image(image)

#     # Grad-CAM 저장 (static/)
#     gradcam_img = generate_gradcam(image, pred_idx)
#     gradcam_path = f"static/gradcam_{pred_idx}.png"
#     gradcam_img.save(gradcam_path)

#     return PredictionResponse(class_idx=pred_idx, class_name=class_name, probability=probability)