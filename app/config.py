import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "final_model.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
