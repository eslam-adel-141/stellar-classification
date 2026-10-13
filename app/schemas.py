from pydantic import BaseModel


class StellarObjectInput(BaseModel):
    alpha: float
    delta: float
    u: float
    g: float
    r: float
    i: float
    z: float
    redshift: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "alpha": 135.68, "delta": 32.49,
                "u": 23.87, "g": 22.27, "r": 20.39, "i": 19.01, "z": 18.52,
                "redshift": 0.634
            }
        }
    }


class PredictionOutput(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
