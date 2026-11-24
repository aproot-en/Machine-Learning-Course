"""Health check and optional handwritten-letter prediction."""

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from PIL import Image

from exam_src.api.deps import get_current_user

router = APIRouter(prefix="/api", tags=["utility"])


@router.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "inference_wrapper", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@router.post("/predict/single")
async def predict_single(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Demo/teaching endpoint: classify one handwritten A-Z character image.
    Not part of the real grading path -- see exam_docs/answer_sheet_real_template.md."""
    wrapper = getattr(request.app.state, "inference_wrapper", None)
    if wrapper is None:
        raise HTTPException(503, "model not loaded")

    try:
        img = Image.open(file.file).convert("L")
    except Exception:
        raise HTTPException(400, "could not read image file")

    result = wrapper.classify([np.array(img)])[0]
    return result
