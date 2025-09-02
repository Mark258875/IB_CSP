import torch, json
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

# choose backbone: "openai/clip-vit-large-patch14" via open_clip
import open_clip

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.eval().cuda()
    return model, preprocess

def embed_dir(crops_dir, out_npy):
    model, preprocess = load_model()
    crops = sorted([p for p in Path(crops_dir).glob("*.jpg")])
    feats = np.zeros((len(crops), 768), dtype=np.float32)  # CLIP-L=768
    for i,p in enumerate(tqdm(crops)):
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).cuda()
        with torch.no_grad():
            f = model.encode_image(img)
            f = torch.nn.functional.normalize(f, p=2, dim=-1).squeeze().float().cpu().numpy()
        feats[i] = f
    np.save(out_npy, feats)
    with open(Path(crops_dir)/"index.json","w") as f:
        json.dump([str(p.name) for p in crops], f)
if __name__ == "__main__":
    import sys; embed_dir(sys.argv[1], sys.argv[2])
