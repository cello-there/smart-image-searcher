from __future__ import annotations
import numpy as np
import warnings

# Soft-deps
try:
    import torch
    import open_clip
except Exception as e:  # noqa: BLE001
    torch = None
    open_clip = None


class _Normalizer:
    @staticmethod
    def l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return x / n


class ImageEmbedder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = None
        if open_clip is not None and torch is not None:
            model_name = cfg.get("clip_model", "ViT-B-32")
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.device = self._pick_device(cfg.get("device", "auto"))
                self.model.to(self.device)
                self.backend = "open_clip"
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"open_clip init failed: {e}. Falling back to numpy stub.")
        if self.backend is None:
            warnings.warn("Using random-stub embeddings for images. Install torch + open-clip-torch for real embeddings.")

    def _pick_device(self, pref: str):
        if pref == "cuda" and torch and torch.cuda.is_available():
            return "cuda"
        if pref in {"cpu", "cuda"}:
            return "cpu" if pref == "cpu" else ("cuda" if torch and torch.cuda.is_available() else "cpu")
        return "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    def encode_images(self, paths: list[str]) -> np.ndarray:
        if self.backend == "open_clip":
            from PIL import Image
            with torch.no_grad():
                imgs = []
                for p in paths:
                    img = Image.open(p).convert("RGB")
                    imgs.append(self.preprocess(img))
                batch = torch.stack(imgs).to(self.device)
                feats = self.model.encode_image(batch)
                feats = feats.cpu().numpy()
                return _Normalizer.l2(feats)
        # Fallback: deterministic random based on path hash
        vecs = []
        for p in paths:
            rng = np.random.default_rng(abs(hash(p)) % (2**32))
            vecs.append(rng.normal(size=512).astype("float32"))
        return _Normalizer.l2(np.vstack(vecs))


class TextEmbedder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = None
        if open_clip is not None and torch is not None:
            model_name = cfg.get("clip_model", "ViT-B-32")
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                self.backend = "open_clip"
            except Exception:
                pass
        if self.backend is None:
            warnings.warn("Using random-stub embeddings for text. Install torch + open-clip-torch for real embeddings.")

    def encode_text(self, text: str) -> np.ndarray:
        if self.backend == "open_clip":
            import torch
            with torch.no_grad():
                toks = self.tokenizer([text])
                toks = toks.to(self.device)
                feats = self.model.encode_text(toks)
                feats = feats.cpu().numpy().astype("float32")
                feats = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)
                return feats[0]
        # Fallback: deterministic random based on text hash
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=512).astype("float32")
        return v / (np.linalg.norm(v) + 1e-12)