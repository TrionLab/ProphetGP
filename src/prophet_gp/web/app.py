from __future__ import annotations

import os
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prophet_gp.config import AppConfig, load_config
from prophet_gp.data.dataset import ReactionDatasetService
from prophet_gp.pipeline.trainer import ProphetGPPipeline, TrainingArtifacts

_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(yaml|yml)$")
_MAX_SESSIONS = 32


@dataclass
class SessionState:
    pipeline: ProphetGPPipeline
    artifacts: TrainingArtifacts


_sessions: "OrderedDict[str, SessionState]" = OrderedDict()


def _project_root() -> Path:
    env = os.environ.get("PROPHET_GP_ROOT")
    if env:
        return Path(env).resolve()
    # src/prophet_gp/web/app.py → repo root
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _configs_dir() -> Path:
    d = CONFIGS_DIR.resolve()
    if not d.is_dir():
        raise HTTPException(status_code=500, detail=f"configs directory missing: {d}")
    return d


def _safe_config_basename(name: str) -> str:
    base = Path(name).name
    if not _CONFIG_NAME_RE.fullmatch(base):
        raise HTTPException(status_code=400, detail="Invalid config file name.")
    return base


def _resolve_data_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (PROJECT_ROOT / p).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Data path must stay under project root.") from exc
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"Data file not found: {resolved}")
    return resolved


def _store_session(state: SessionState) -> str:
    sid = str(uuid.uuid4())
    while len(_sessions) >= _MAX_SESSIONS:
        _sessions.popitem(last=False)
    _sessions[sid] = state
    _sessions.move_to_end(sid)
    return sid


def _get_session(session_id: str) -> SessionState:
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session; train again.")
    _sessions.move_to_end(session_id)
    return state


def _resolve_output_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (PROJECT_ROOT / p).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Output path must stay under project root.") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


class TrainRequest(BaseModel):
    config_name: str = Field(..., description="configs/*.yaml basename")
    data_path: str = Field(
        ...,
        description="CSV path relative to project root or absolute under project root.",
    )


class TrainResponse(BaseModel):
    session_id: str
    train_rows: int
    feature_dims: int
    target_columns: List[str]
    molecular_unique_rows: int


class SuggestRequest(BaseModel):
    session_id: str
    n_candidates: Optional[int] = None
    strategy: Optional[str] = Field(None, description="best_output | best_information")


class AppendRequest(BaseModel):
    base_data_path: str
    new_data_path: str
    out_path: str
    config_name: str


app = FastAPI(title="ProphetGP Web", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "project_root": str(PROJECT_ROOT)}


@app.get("/api/configs")
def list_configs() -> Dict[str, Any]:
    d = _configs_dir()
    names = sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".yaml", ".yml"))
    return {"configs_dir": str(d), "files": names}


@app.get("/api/config/{name}")
def get_config_text(name: str) -> JSONResponse:
    base = _safe_config_basename(name)
    path = _configs_dir() / base
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Config not found: {base}")
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    validated = AppConfig.model_validate(parsed)
    return JSONResponse(
        {
            "name": base,
            "text": text,
            "validated_summary": validated.model_dump(),
        }
    )


@app.put("/api/config/{name}")
async def put_config_text(name: str, request: Request) -> Dict[str, str]:
    base = _safe_config_basename(name)
    body_bytes = await request.body()
    try:
        text = body_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Body must be UTF-8 text.") from exc
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    try:
        AppConfig.model_validate(parsed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Config validation failed: {exc}") from exc
    path = _configs_dir() / base
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"status": "saved", "path": str(path)}


@app.get("/api/featurisers")
def list_featurisers() -> Dict[str, List[str]]:
    pipeline = ProphetGPPipeline(AppConfig())
    return {"featurisers": pipeline.featurizers.available()}


@app.post("/api/train", response_model=TrainResponse)
def train_endpoint(body: TrainRequest) -> TrainResponse:
    cfg_path = _configs_dir() / _safe_config_basename(body.config_name)
    if not cfg_path.is_file():
        raise HTTPException(status_code=404, detail=f"Config not found: {cfg_path.name}")
    data_file = _resolve_data_path(body.data_path)
    config = load_config(cfg_path)
    pipeline = ProphetGPPipeline(config)
    artifacts = pipeline.train_from_csv(str(data_file))
    mol_part = artifacts.x_train[:, : artifacts.mol_feature_dim]
    unique_mol = int(np.unique(np.asarray(mol_part), axis=0).shape[0])
    sid = _store_session(SessionState(pipeline=pipeline, artifacts=artifacts))
    return TrainResponse(
        session_id=sid,
        train_rows=int(artifacts.x_train.shape[0]),
        feature_dims=int(artifacts.x_train.shape[1]),
        target_columns=list(artifacts.target_columns),
        molecular_unique_rows=unique_mol,
    )


@app.post("/api/suggest")
def suggest_endpoint(body: SuggestRequest) -> Dict[str, Any]:
    state = _get_session(body.session_id)
    n_cand = body.n_candidates or state.pipeline.config.optimization.n_candidates
    if n_cand < 1:
        raise HTTPException(status_code=400, detail="n_candidates must be >= 1")
    strategy = body.strategy or state.pipeline.config.optimization.suggestion_strategy
    if strategy not in ("best_output", "best_information"):
        raise HTTPException(status_code=400, detail="strategy must be best_output or best_information")
    suggestions = state.pipeline.suggest_next_experiments(
        state.artifacts,
        n_candidates=int(n_cand),
        strategy=strategy,
    )
    decoded = [_json_safe(row) for row in suggestions.decoded_candidates]
    return {
        "strategy": strategy,
        "n_candidates": int(n_cand),
        "raw_candidates_shape": list(suggestions.raw_candidates.shape),
        "candidates": decoded,
    }


@app.post("/api/append")
def append_endpoint(body: AppendRequest) -> Dict[str, Any]:
    cfg_path = _configs_dir() / _safe_config_basename(body.config_name)
    if not cfg_path.is_file():
        raise HTTPException(status_code=404, detail=f"Config not found: {cfg_path.name}")
    base = _resolve_data_path(body.base_data_path)
    new = _resolve_data_path(body.new_data_path)
    out = _resolve_output_path(body.out_path)
    config = load_config(cfg_path)
    ds = ReactionDatasetService(config.data)
    merged = ds.append_csv(str(base), str(new), str(out))
    return {"status": "ok", "rows": int(len(merged)), "output": str(out)}


@app.get("/")
def serve_index() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=500, detail=f"Missing UI file: {index}")
    return FileResponse(index)


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def main() -> None:
    import uvicorn

    host = os.environ.get("PROPHET_GP_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("PROPHET_GP_WEB_PORT", "8765"))
    uvicorn.run(
        "prophet_gp.web.app:app",
        host=host,
        port=port,
        reload=os.environ.get("PROPHET_GP_WEB_RELOAD", "").lower() in ("1", "true", "yes"),
    )
