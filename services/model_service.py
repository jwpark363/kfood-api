# model_service.py: 모델 로드 및 추론 로직
# 
# services/model_service.py

# services/model_service.py

import json, os, csv
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights
from models.efficientnet import EfficientNetV2S
# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelService:
    def __init__(self, model_path: str, num_classes: int = 149, class_mapping_csv_path: str = 'food_class_mapping.csv'):
        # 모델 클래스 인스턴스화
        self.model = EfficientNetV2S(num_classes=num_classes)
        
        # 모델 가중치 로드
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # 로드된 state_dict의 키를 현재 모델 구조에 맞게 수정합니다.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("features") or k.startswith("classifier"):
                new_state_dict["model." + k] = v
            elif k.startswith("backbone."):
                new_state_dict[k.replace("backbone.", "model.")] = v
            else:
                new_state_dict[k] = v

        try:
            # 키가 조정된 state_dict를 모델에 로드합니다.
            self.model.load_state_dict(new_state_dict, strict=True)
            print("모델 가중치 로드 성공!")
        except Exception as e:
            print(f"모델 가중치 로드 실패: {e}")
            raise RuntimeError(f"Failed to load model state_dict: {e}")

        self.model.to(DEVICE)
        self.model.eval()
        
        self.preprocess = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()

        # CSV 파일에서 매핑 정보를 읽어 딕셔너리로 저장합니다.
        self.idx_to_name_mapping: Dict[int, str] = {}
        self.name_to_code_mapping: Dict[str, str] = {}
        
        if os.path.exists(class_mapping_csv_path):
            with open(class_mapping_csv_path, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    idx = int(row['id'])
                    name = row['name']
                    code = row['code']
                    
                    self.idx_to_name_mapping[idx] = name
                    self.name_to_code_mapping[name] = code
            print("CSV 파일에서 매핑 정보 로드 성공!")
        else:
            print(f"경로에 CSV 파일이 없습니다: {class_mapping_csv_path}")
            
    def predict(self, image_pil):
        # 이미지 전처리 및 모델 예측
        x = self.preprocess(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)
            top_prob, top_idx = probs.max(dim=1)
        
        idx = int(top_idx.item())
        
        # 예측된 인덱스로부터 클래스 이름을 가져옵니다.
        name = self.idx_to_name_mapping.get(idx)
        
        # 클래스 이름으로부터 D-코드를 가져옵니다.
        code = self.name_to_code_mapping.get(name)

        return {
            "class_id": idx,
            "class_name": name,
            "class_code": code,
            "probability": float(top_prob.item())
        }


# import json, os
# from typing import Optional, List, Dict

# import torch
# import torch.nn.functional as F
# from torchvision.models import EfficientNet_V2_S_Weights

# from models.efficientnet import EfficientNetV2S
# # GPU 사용 가능 여부 확인
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ModelService:
#     def __init__(self, model_path: str, num_classes: int = 149, class_mapping_path: Optional[str] = None):
#         # 모델 클래스 인스턴스화
#         self.model = EfficientNetV2S(num_classes=num_classes, pretrained=False)
        
#         # 모델 가중치 로드
#         state_dict = torch.load(model_path, map_location=DEVICE)
        
#         # 로드된 state_dict의 키를 현재 모델 구조에 맞게 수정합니다.
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith("features") or k.startswith("classifier"):
#                 new_state_dict["model." + k] = v
#             elif k.startswith("backbone."):
#                 new_state_dict[k.replace("backbone.", "model.")] = v
#             else:
#                 new_state_dict[k] = v

#         try:
#             # 키가 조정된 state_dict를 모델에 로드합니다.
#             self.model.load_state_dict(new_state_dict, strict=True)
#             print("모델 가중치 로드 성공!")
#         except Exception as e:
#             print(f"모델 가중치 로드 실패: {e}")
#             raise RuntimeError(f"Failed to load model state_dict: {e}")

#         self.model.to(DEVICE)
#         self.model.eval()
        
#         self.preprocess = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()

#         # 클래스 이름과 D-코드 매핑을 위한 딕셔너리 초기화
#         self.class_names: Optional[List[str]] = None
#         self.name_to_code_mapping: Optional[Dict[str, str]] = None

#         if class_mapping_path and os.path.exists(class_mapping_path):
#             with open(class_mapping_path, "r", encoding="utf-8") as f:
#                 # JSON 파일을 그대로 읽어와서 음식 이름 -> D-코드 매핑을 생성합니다.
#                 self.name_to_code_mapping = json.load(f)

#             # D-코드 값(value)을 기준으로 정렬하여 클래스 이름 리스트를 생성합니다.
#             # D-코드는 정렬이 가능한 문자열이므로, 이를 활용하여 클래스 순서를 맞춥니다.
#             sorted_items = sorted(self.name_to_code_mapping.items(), key=lambda item: item[1])
#             self.class_names = [item[0] for item in sorted_items]

#     def predict(self, image_pil):
#         # 이미지 전처리 및 모델 예측
#         x = self.preprocess(image_pil).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             outputs = self.model(x)
#             probs = F.softmax(outputs, dim=1)
#             top_prob, top_idx = probs.max(dim=1)
        
#         idx = int(top_idx.item())
        
#         # 예측된 인덱스로부터 클래스 이름을 가져옵니다.
#         name = self.class_names[idx] if self.class_names and idx < len(self.class_names) else None
        
#         # 클래스 이름으로부터 D-코드를 가져옵니다.
#         code = self.name_to_code_mapping.get(name) if name else None

#         return {
#             "class_id": idx,
#             "class_name": name,
#             "class_code": code,
#             "probability": float(top_prob.item())
#         }
# import json, os
# from typing import Optional, List

# import torch
# import torch.nn.functional as F
# from torchvision.models import EfficientNet_V2_S_Weights

# from models.efficientnet import EfficientNetV2S

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ModelService:
#     def __init__(self, model_path: str, num_classes: int = 149, class_mapping_path: Optional[str] = None):
#         self.model = EfficientNetV2S(num_classes=num_classes, pretrained=False)
#         # self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#         state = torch.load(model_path, map_location=DEVICE)
#         self.model.load_state_dict(state, strict=True)
#         self.model.to(DEVICE)
#         self.model.eval()
#         # 추론용 전처리
#         self.preprocess = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()

#         # 클래스 인덱스->이름 매핑
#         self.class_names: Optional[List[str]] = None
#         if class_mapping_path and os.path.exists(class_mapping_path):
#             with open(class_mapping_path, "r", encoding="utf-8") as f:
#                 mapping = json.load(f)
#             # 키가 문자열이라 인덱스순으로 정라
#             self.class_names = [mapping[str(i)] for i in range(len(mapping))]

#     def predict(self, image_pil):
#         x = self.preprocess(image_pil).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             outputs = self.model(x)
#             probs = F.softmax(outputs, dim=1)
#             top_prob, top_idx = probs.max(dim=1)
#         idx = int(top_idx.item())
#         name = self.class_names[idx] if self.class_names else None
#         return {
#             "class_id": idx,
#             "class_name": name,
#             "probability": float(top_prob.item())
#         }
