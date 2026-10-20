import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from src.visa_data import ROOT

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PatchExtractor:
    def __init__(self, size=(288, 384)):
        self.size = size  # (height, width)
        m = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3

    def load(self, rel):
        img = Image.open(ROOT / rel).convert("RGB")
        img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))

    @torch.no_grad()
    def features(self, batch):
        f2 = self.layer2(self.stem(batch))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)  # each patch also sees its neighbours
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # batch x 384 x H x W

    def extract_all(self, rels, batch=16):
        out = []
        for i in range(0, len(rels), batch):
            x = torch.stack([self.load(r) for r in rels[i:i + batch]])
            out.append(self.features(x).half().numpy())
            print(f"  features {min(i + batch, len(rels))}/{len(rels)}", end="\r", flush=True)
        print()
        return np.concatenate(out)


def build_bank(feats, fraction, seed=0):
    """Random subset of all training patches. feats: N x C x H x W."""
    n, c, h, w = feats.shape
    flat = feats.reshape(n, c, h * w)
    total = n * h * w
    k = max(1, int(total * fraction))
    idx = np.sort(np.random.default_rng(seed).choice(total, size=k, replace=False))
    return torch.from_numpy(flat[idx // (h * w), :, idx % (h * w)].astype(np.float32))


def patch_scores(q, bank, chunk=20000):
    """Distance from each query patch (P x C) to its nearest bank patch (M x C)."""
    qq = (q * q).sum(1, keepdim=True)
    best = torch.full((q.shape[0],), float("inf"))
    for s in range(0, bank.shape[0], chunk):
        b = bank[s:s + chunk]
        d = qq + (b * b).sum(1)[None, :] - 2 * (q @ b.T)
        best = torch.minimum(best, d.min(dim=1).values)
    return best.clamp_min(0).sqrt()


def score_maps(feats, bank):
    """Anomaly map (H x W of patch distances) for each image in feats."""
    n, c, h, w = feats.shape
    maps = np.zeros((n, h, w), dtype=np.float32)
    for i in range(n):
        q = torch.from_numpy(feats[i].reshape(c, h * w).T.astype(np.float32))
        maps[i] = patch_scores(q, bank).reshape(h, w).numpy()
    return maps