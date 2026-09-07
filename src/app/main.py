from fastapi import FastAPI

app = FastAPI(title="akpedia-ml", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check simples usado por orquestração e pelo CI."""
    return {"status": "UP"}
