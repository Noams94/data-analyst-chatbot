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


@router.get("")
async def list_datasets(user_id: str = Depends(get_current_user_id)) -> list[dict]:
    return [_to_response(rec) for rec in state.list_datasets(user_id)]


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


@router.get("/{dataset_id}/preview")
async def preview(dataset_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    """Structured preview: per-column stats + numeric summary + 5 sample rows.
    Used by the dataset-detail page to render a 'know your data' panel before
    the user starts chatting.
    """
    import math
    import pandas as pd

    rec = state.get_dataset(dataset_id, user_id)
    if not rec:
        raise HTTPException(404, "Dataset not found")
    df = state.load_dataframe(rec.parquet_path)
    n = max(int(df.shape[0]), 1)

    columns = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        sample_values = (
            s.dropna().head(3).astype(str).tolist()
        )
        columns.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "nullCount": nulls,
            "nullPct": round(nulls / n * 100, 1),
            "uniqueCount": unique,
            "sampleValues": sample_values,
        })

    # Numeric summary — describe(), but JSON-friendly.
    num_cols = df.select_dtypes(include="number").columns
    numeric_summary: dict[str, dict[str, float]] = {}
    if len(num_cols):
        desc = df[num_cols].describe().round(4)
        for col in num_cols:
            stats = {}
            for stat in ("mean", "std", "min", "25%", "50%", "75%", "max"):
                val = desc.loc[stat, col]
                # Replace NaN/inf with None so it's JSON-encodable.
                if pd.isna(val) or (isinstance(val, float) and math.isinf(val)):
                    stats[stat] = None
                else:
                    stats[stat] = float(val)
            numeric_summary[str(col)] = stats

    # Sample rows — JSON-safe.
    head = df.head(5).where(pd.notna(df.head(5)), None)
    sample_rows = head.to_dict(orient="records")

    return {
        "id": rec.id,
        "name": rec.name,
        "rowCount": rec.row_count,
        "createdAt": rec.created_at,
        "columns": columns,
        "numericSummary": numeric_summary,
        "sampleRows": sample_rows,
    }


@router.delete("/{dataset_id}", status_code=204, response_model=None)
async def delete_dataset(dataset_id: str, user_id: str = Depends(get_current_user_id)) -> None:
    if not state.delete_dataset(dataset_id, user_id):
        raise HTTPException(404, "Dataset not found")


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    rec = state.get_dataset(dataset_id, user_id)
    if not rec:
        raise HTTPException(404, "Dataset not found")
    return _to_response(rec)
