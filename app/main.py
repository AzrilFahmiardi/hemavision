"""API lokal untuk model konjungtiva dan palm.

Menyediakan endpoint POST /predict (konjungtiva, foto mata mentah) dan
POST /predict/palm (palm, video telapak tangan mentah) beserta variabel
demografi, lalu mengembalikan estimasi hemoglobin, status anemia, dan
severity. Dijalankan lokal lewat uvicorn untuk pengujian sebelum
dipertimbangkan deploy, lihat docs/DEPLOYMENT.md untuk opsi hosting gratis.

Jalankan lokal:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.predict import predict
from app.predict_palm import predict_palm

app = FastAPI(
    title="Hemavision API",
    description="Skrining anemia non-invasif berbasis citra konjungtiva dan palm.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Pemeriksaan kesehatan sederhana untuk memastikan layanan berjalan."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    age_years: float = Form(...),
    gender: str = Form(...),
    site: str = Form("ghana"),
    include_stages: bool = Form(False),
) -> JSONResponse:
    """Terima unggahan foto mata mentah dan kembalikan hasil skrining anemia.

    Bila include_stages diisi true, response turut menyertakan stage_images
    (PNG base64 per tahap pipeline) untuk ditampilkan pada dashboard klinis.
    """
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(await image.read())
        temp_file.flush()
        result = predict(
            temp_file.name, age_years=age_years, gender=gender, site=site, include_stages=include_stages
        )
    return JSONResponse(content=result)


@app.post("/predict/palm")
async def predict_palm_endpoint(
    video: UploadFile = File(...),
    age_years: float = Form(...),
    gender: str = Form(...),
    include_stages: bool = Form(False),
) -> JSONResponse:
    """Terima unggahan video telapak tangan mentah dan kembalikan hasil skrining anemia.

    Bila include_stages diisi true, response turut menyertakan stage_images
    (PNG base64 per tahap pipeline) untuk ditampilkan pada dashboard klinis.
    """
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(await video.read())
        temp_file.flush()
        result = predict_palm(
            temp_file.name, age_years=age_years, gender=gender, include_stages=include_stages
        )
    return JSONResponse(content=result)
