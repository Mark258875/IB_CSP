from fastapi import FastAPI, UploadFile
import numpy as np
from src.index.search import Searcher
# load detector weights, embedder, faiss index on startup
app = FastAPI()
searcher = Searcher("models/faiss_ivfpq.index", "outputs/index.json")

@app.post("/search")
async def search(crop: UploadFile):
    # 1) (optional) detect graffiti if full image
    # 2) embed crop -> vec
    # 3) kNN -> candidate cluster (via majority vote)
    return {"neighbors": [...]}

@app.get("/clusters/{cid}")
def cluster_info(cid: int):
    # return sightings, locations, last_seen
    return {...}
