import io, uuid
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import preprocessing as pp
from meta_classifier import MetaClassifier
from meta_regressor import MetaRegressor

app = FastAPI(title="Meta-Learning AutoML Recommender")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RECS = {"classification": MetaClassifier(), "regression": MetaRegressor()}
SESSIONS = {}   # in-memory; fine for a single-user demo

class PreIn(BaseModel):
    session_id: str
    numeric_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    constant: str = "Unknown"

class RecIn(BaseModel):
    session_id: str
    target: str

class EvalIn(BaseModel):
    session_id: str
    model: Optional[str] = None
    
def session(sid):
    if sid not in SESSIONS: raise HTTPException(404, "Unknown session. Upload a dataset first.")
    return SESSIONS[sid]

@app.post("/upload")
def upload(file: UploadFile = File(...), task: str = Form(...)):
    if task not in RECS: raise HTTPException(400, "task must be 'classification' or 'regression'")
    try:
        df = pp.read_table(io.BytesIO(file.file.read()), file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"task": task, "raw": df}
    return {"session_id": sid, **pp.inspect(df)}

@app.post("/preprocess")
def preprocess(b: PreIn):
    s = session(b.session_id)
    try:
        s["clean"] = pp.apply_imputation(s["raw"], b.numeric_strategy, b.categorical_strategy, b.constant)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return pp.inspect(s["clean"])

@app.post("/recommend")
def recommend(b: RecIn):
    s = session(b.session_id)
    if "clean" not in s: raise HTTPException(400, "Run /preprocess first.")
    try:
        X, X_raw, y = pp.prepare_xy(s["raw"], s["clean"], b.target)
    except ValueError as e:
        raise HTTPException(400, str(e))
    errs = pp.validate_xy(X, y, s["task"])
    if errs: raise HTTPException(422, {"errors": errs})
    rec = RECS[s["task"]].recommend(X_raw, y)
    s.update(X=X, y=y, rec=rec)
    return {"n_samples": len(X), "n_features": X.shape[1], **rec}

@app.post("/evaluate")
def evaluate(b: EvalIn):
    s = session(b.session_id)
    if "rec" not in s: raise HTTPException(400, "Run /recommend first.")
    r = RECS[s["task"]]
    name = b.model or s["rec"]["recommended"]
    if name not in r.candidates(): raise HTTPException(400, f"Unknown model '{name}'")
    return r.evaluate(s["X"], s["y"], name)

FRONT = Path(__file__).resolve().parent.parent / "frontend"
if FRONT.exists():                                       # must stay last, after the API routes
    app.mount("/", StaticFiles(directory=FRONT, html=True), name="ui")