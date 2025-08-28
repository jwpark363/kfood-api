import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api import predict

print('start api......')
app = FastAPI(title="KFood Demon Hunters", vision="1.0.0")
origins = [
    "http://localhost:5173",  # Your frontend's origin
    "http://127.0.0.1:5173",  # Or this one, depending on how your React app runs
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=[""],  # Allows all headers
)

# 라우터 등록
app.include_router(predict.router, prefix="/api", tags=["prediction"])

# /static/* 로 이미지 제공
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
