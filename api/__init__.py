# backend/api/__init__.py

# from . import predict

from fastapi import APIRouter
from .predict import router as predict_router

api_router = APIRouter()
api_router.include_router(predict_router, prefix="/predict")
