import joblib
from app.config import MODEL_PATH, PIPELINE_PATH, ENCODER_PATH
from app.feature_engineering import StellarFeatureEngineer  # noqa: F401 (needed for unpickling)

_model = None
_pipeline = None
_label_encoder = None


def load_artifacts():
    global _model, _pipeline, _label_encoder
    _model = joblib.load(MODEL_PATH)
    _pipeline = joblib.load(PIPELINE_PATH)
    _label_encoder = joblib.load(ENCODER_PATH)


def get_model():
    return _model


def get_pipeline():
    return _pipeline


def get_label_encoder():
    return _label_encoder
