import pandas as pd
from fastapi import APIRouter

from app.schemas import StellarObjectInput, PredictionOutput
from app.model_loader import get_model, get_pipeline, get_label_encoder

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionOutput)
def predict(payload: StellarObjectInput):
    model = get_model()
    pipeline = get_pipeline()
    label_encoder = get_label_encoder()

    input_df = pd.DataFrame([payload.model_dump()])
    X_processed = pipeline.transform(input_df)

    pred_encoded = model.predict(X_processed)[0]
    pred_proba = model.predict_proba(X_processed)[0]

    predicted_class = label_encoder.inverse_transform([pred_encoded])[0]
    probabilities = {cls: float(p) for cls, p in zip(label_encoder.classes_, pred_proba)}

    return PredictionOutput(predicted_class=predicted_class, probabilities=probabilities)
