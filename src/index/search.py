import faiss, numpy as np, json
from pathlib import Path

class Searcher:
    def __init__(self, index_path, names_json):
        self.index = faiss.read_index(index_path)
        self.names = json.loads(Path(names_json).read_text())
    def knn(self, vecs, k=30):
        D,I = self.index.search(vecs.astype('float32'), k)
        return D,I

# usage: embed new crop -> knn -> candidates for verification
