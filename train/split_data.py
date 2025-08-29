# 데이터 샘플링 및 분할 스크립트

import os
import shutil
import random
import splitfolders

# 사용자 설정
INPUT_FOLDER = "kfood_original"
TEMP_SAMPLE_FOLDER = "kfood_sample"
OUTPUT_FOLDER = "kfood_processed"
SAMPLE_SIZE = 1000
RATIO = (0.8, 0.1, 0.1) # 훈련:검증:테스트 비율
RANDOM_SEED = 42

# 원본 데이터에서 지정된 개수만큼 이미지를 샘플링
print(f"[{INPUT_FOLDER}] 폴더에서 각 클래스별로 {SAMPLE_SIZE}개 이미지를 샘플링")
if os.path.exists(TEMP_SAMPLE_FOLDER):
    shutil.rmtree(TEMP_SAMPLE_FOLDER)
os.makedirs(TEMP_SAMPLE_FOLDER)

class_folders = [d for d in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, d))]
for class_name in class_folders:
    class_path = os.path.join(INPUT_FOLDER, class_name)
    output_class_path = os.path.join(TEMP_SAMPLE_FOLDER, class_name)
    os.makedirs(output_class_path)
    
    all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sampled_images = random.sample(all_images, min(len(all_images), SAMPLE_SIZE))
    
    for image_name in sampled_images:
        src_path = os.path.join(class_path, image_name)
        dst_path = os.path.join(output_class_path, image_name)
        shutil.copy(src_path, dst_path)

print("\n이미지 샘플링 완료!")

# 2. 샘플링된 데이터를 훈련, 검증, 테스트셋으로 분할
print(f"'{TEMP_SAMPLE_FOLDER}' 폴더의 데이터를 분할합니다...")
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)

splitfolders.ratio(
    input=TEMP_SAMPLE_FOLDER,
    output=OUTPUT_FOLDER,
    seed=RANDOM_SEED,
    ratio=RATIO,
    # group_prefix=2
)
print("데이터 분할 완료!")