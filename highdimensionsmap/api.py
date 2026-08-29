from typing import List, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .scanner import HDMScanner

app = FastAPI(title="HDM API", version="0.1.0")
scanner = HDMScanner()


class TransformRequest(BaseModel):
    input: Union[List[float], List[List[float]]]


class QueryRequest(BaseModel):
    dataset: List[List[float]]
    query: List[float]
    k: int = 5


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/transform")
def transform_endpoint(payload: TransformRequest):
    try:
        values = np.asarray(payload.input, dtype=float)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="'input' precisa ser numérico.") from exc

    if values.ndim not in (1, 2):
        raise HTTPException(status_code=400, detail="'input' deve ser um vetor ou uma matriz 2D.")

    signature = scanner.transform(values)
    return {"signature": np.asarray(signature).tolist()}


@app.post("/query")
def query_endpoint(payload: QueryRequest):
    try:
        dataset = np.asarray(payload.dataset, dtype=float)
        query = np.asarray(payload.query, dtype=float)
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

    indices, distances = scanner.query(dataset, query, k=payload.k)
    return {"indices": indices.tolist(), "distances": distances.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("highdimensionsmap.api:app", host="0.0.0.0", port=8000, reload=True)
