from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.model_loader import load_artifacts
from app.routers import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Stellar Classification API", version="1.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(predict.router)
