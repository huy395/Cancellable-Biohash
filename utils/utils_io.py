import os
import pickle
import zstandard as zstd

def save_pickle_zst(obj, path, level=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pkl = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    cctx = zstd.ZstdCompressor(level=level)
    comp = cctx.compress(pkl)
    with open(path, "wb") as f:
        f.write(comp)

def load_pickle_zst(path):
    with open(path, "rb") as f:
        comp = f.read()
    dctx = zstd.ZstdDecompressor()
    pkl = dctx.decompress(comp)
    return pickle.loads(pkl)
