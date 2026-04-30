"""Dataset upload + overview routes."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api import state
from api.auth import get_current_user_id
from api.chat_session import ChatSession, set_session
from api.tools import get_data_overview

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _read_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buf = io.BytesIO(raw)
    if suffix == ".csv":
        return pd.read_csv(buf)
    if suffix == ".tsv":
        return pd.read_csv(buf, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(buf)
    if suffix == ".json":
        return pd.read_json(buf)
    if suffix == ".parquet":
        return pd.read_parquet(buf)
    raise HTTPException(400, f"Unsupported file type: {suffix}")


def _to_response(rec: state.DatasetRecord) -> dict:
    return {
        "id": rec.id,
        "name": rec.name,
        "columns": rec.columns,
        "rowCount": rec.row_count,
        "sizeBytes": rec.size_bytes,
        "createdAt": rec.created_at,
    }


@router.post("")
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 50 MB limit")
    try:
        df = _read_dataframe(file.filename or "data.csv", raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {e}")

    rec = state.create_dataset(
        user_id=user_id,
        name=file.filename or "uploaded_data",
        df=df,
        size_bytes=len(raw),
    )
    return _to_response(rec)


@router.get("/{dataset_id}/overview")
async def overview(dataset_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    rec = state.get_dataset(dataset_id, user_id)
    if not rec:
        raise HTTPException(404, "Dataset not found")
    df = state.load_dataframe(rec.parquet_path)
    set_session(ChatSession(chat_id="overview", df=df, data_name=rec.name))
    return {"overview": get_data_overview()}


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    rec = state.get_dataset(dataset_id, user_id)
    if not rec:
        raise HTTPException(404, "Dataset not found")
    return _to_response(rec)
