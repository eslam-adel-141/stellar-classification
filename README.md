# Stellar Classification API

FastAPI service that classifies SDSS17 astronomical objects as `GALAXY`, `STAR`, or `QSO`
from photometric magnitudes (u, g, r, i, z), sky position (alpha, delta) and redshift.

## Run locally

    pip install -r requirements.txt
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs

## Run with Docker

    docker build -t stellar-api .
    docker run -p 8000:8000 stellar-api

## Endpoints

- `GET /health`: health check
- `POST /predict`: predict the class of an object

Example body:

    {"alpha": 135.68, "delta": 32.49, "u": 23.87, "g": 22.27, "r": 20.39,
     "i": 19.01, "z": 18.52, "redshift": 0.634}

## Note

The model relies on `redshift`, a spectroscopic measurement, so it is not meant for
objects that only have photometry.

## Training

See `notebooks/stellar_classification.ipynb` for EDA, feature engineering, model training and evaluation.
