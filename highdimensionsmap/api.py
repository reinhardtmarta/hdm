from typing import List, Union, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .scanner import HDMScanner
from .compressor import HDMCompressor

app = FastAPI(title="HDM API", version="0.3.0")
scanner = HDMScanner(max_dim=16384)
compressor = HDMCompressor(max_dim=65536)


class TransformRequest(BaseModel):
    input: Union[List[float], List[List[float]]]


class QueryRequest(BaseModel):
    dataset: List[List[float]]
    query: List[float]
    k: int = 5


class CompressRequest(BaseModel):
    input: Union[List[float], List[List[float]]]
    target_energy: float = 0.999
    output_dir: Optional[str] = None
    verbose: bool = False


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "scanner_max_dim": scanner.max_dim,
        "compressor_max_dim": compressor.max_dim,
    }


@app.post("/transform")
def transform_endpoint(payload: TransformRequest):
    try:
        values = np.asarray(payload.input, dtype=np.float64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="'input' precisa ser numérico.") from exc

    if values.ndim not in (1, 2):
        raise HTTPException(status_code=400, detail="'input' deve ser um vetor ou uma matriz 2D.")

    was_1d = values.ndim == 1
    try:
        signature = scanner.transform(values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sig_array = np.asarray(signature, dtype=np.float64)

    # Garante que a estrutura dimensional da resposta corresponda à entrada
    if was_1d and sig_array.ndim > 1:
        sig_array = sig_array[0]
    elif not was_1d and sig_array.ndim == 1:
        sig_array = sig_array.reshape(1, -1)

    return {"signature": sig_array.tolist()}


@app.post("/query")
def query_endpoint(payload: QueryRequest):
    try:
        dataset = np.asarray(payload.dataset, dtype=np.float64)
        query = np.asarray(payload.query, dtype=np.float64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="'dataset' e 'query' precisam ser numéricos.") from exc

    if dataset.ndim != 2:
        raise HTTPException(status_code=400, detail="'dataset' deve ser uma matriz 2D.")
    if query.ndim != 1:
        raise HTTPException(status_code=400, detail="'query' deve ser um vetor 1D.")
    if dataset.shape[1] != query.shape[0]:
        raise HTTPException(status_code=400, detail="As dimensões de 'dataset' e 'query' devem bater.")
    if payload.k <= 0:
        raise HTTPException(status_code=400, detail="'k' precisa ser maior que zero.")
    if payload.k > dataset.shape[0]:
        raise HTTPException(status_code=400, detail="'k' não pode ser maior que o tamanho do dataset.")

    try:
        indices, distances = scanner.query(dataset, query, k=payload.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"indices": indices.tolist(), "distances": distances.tolist()}


@app.post("/compress")
def compress_endpoint(payload: CompressRequest) -> Dict[str, Any]:
    try:
        values = np.asarray(payload.input, dtype=np.float64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="'input' precisa ser numérico.") from exc

    try:
        local_compressor = HDMCompressor(
            target_energy=payload.target_energy,
            max_dim=compressor.max_dim,
        )
        reconstructed, k_retained, exec_time_ms = local_compressor.fit_transform(
            values,
            output_dir=payload.output_dir,
            verbose=payload.verbose,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    k_out = int(k_retained) if np.ndim(k_retained) == 0 else k_retained.tolist()

    return {
        "reconstructed": reconstructed.tolist(),
        "k_retained": k_out,
        "exec_time_ms": exec_time_ms,
        "residual_report": local_compressor.last_residual_report,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("highdimensionsmap.api:app", host="0.0.0.0", port=8000, reload=True)
    
