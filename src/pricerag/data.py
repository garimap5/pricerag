"""Synthetic product catalogue with a known price-generating process.

Why synthetic: it lets every claim in the README be reproduced with zero downloads,
and the generator deliberately creates *product families* (siblings with similar text
and correlated prices) so retrieval has something real to exploit. `load_csv` accepts
a real catalogue (columns: description, price[, category]) so the same pipeline runs on
real data such as the public Amazon product-price dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLUFF = [
    "Great value.", "Ships fast.", "Brand new in box.", "Top rated by customers.",
    "Perfect gift.", "Limited stock.", "Premium build quality.", "Free returns.",
    "Lightweight and durable.", "Latest generation model.",
]


_LINE_A = ["Aurora", "Vortex", "Nimbus", "Zenith", "Apex", "Lumen", "Orbit", "Pulse", "Titan", "Echo",
           "Nova", "Helix", "Quasar", "Vertex", "Ember", "Cobalt", "Falcon", "Mirage", "Onyx", "Prism"]
_LINE_B = ["Pro", "Air", "Max", "Lite", "Plus", "Elite", "One", "Edge", "Core", "Neo"]


def _line_name(rng) -> str:
    """Product-line name shared by siblings (like 'ThinkPad X1'); lets retrieval find them."""
    return f"{rng.choice(_LINE_A)} {rng.choice(_LINE_B)} {rng.integers(1, 10)}"


def _code(rng: np.random.Generator) -> str:
    return f"{rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'))}{rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'))}-{rng.integers(100, 9999)}"


def _pick(rng, d):
    k = list(d)
    return k[rng.integers(len(k))]


def _laptop(rng, fam=None):
    b = {"Acer": .85, "Asus": 1.0, "HP": 1.0, "Dell": 1.05, "Lenovo": 1.0, "Apple": 1.8}
    c = {"Core i3": .8, "Core i5": 1.0, "Core i7": 1.35, "Ryzen 5": .95, "Ryzen 7": 1.25, "M2": 1.5}
    if fam is None:
        fam = dict(brand=_pick(rng, b), cpu=_pick(rng, c))
    ram = int(rng.choice([4, 8, 16, 32])); ssd = int(rng.choice([128, 256, 512, 1024]))
    scr = float(rng.choice([13.3, 14.0, 15.6, 17.3]))
    price = (180 + 22 * ram + .28 * ssd + 12 * scr) * b[fam["brand"]] * c[fam["cpu"]]
    text = f"{fam['brand']} laptop {{code}} with {fam['cpu']} processor, {ram}GB RAM, {ssd}GB SSD and {scr} inch display."
    return fam, text, price


def _headphones(rng, fam=None):
    b = {"JBL": .8, "Sony": 1.2, "Bose": 1.5, "Sennheiser": 1.3, "Anker": .6, "Skullcandy": .7}
    s = {"over-ear": 1.2, "in-ear": .9, "on-ear": 1.0}
    if fam is None:
        fam = dict(brand=_pick(rng, b), style=_pick(rng, s))
    anc = int(rng.integers(0, 2)); bat = int(rng.choice([10, 20, 30, 40, 60]))
    price = (25 + 1.2 * bat + 60 * anc) * b[fam["brand"]] * s[fam["style"]]
    text = (f"{fam['brand']} {fam['style']} wireless headphones {{code}}, "
            f"{'active noise cancelling, ' if anc else ''}{bat} hours battery life.")
    return fam, text, price


def _camera(rng, fam=None):
    b = {"Canon": 1.1, "Nikon": 1.1, "Sony": 1.2, "Fujifilm": 1.15, "Panasonic": 1.0}
    if fam is None:
        fam = dict(brand=_pick(rng, b))
    mp = int(rng.choice([12, 20, 24, 33, 45, 61]))
    lens = {"body only": 0, "18-55mm kit lens": 100, "24-70mm lens": 900, "70-200mm lens": 1400}
    lk = _pick(rng, lens)
    vid = {"1080p": 0, "4K": 250, "6K": 600}
    v = _pick(rng, vid)
    price = (200 + 9 * mp + vid[v]) * b[fam["brand"]] + lens[lk]
    text = f"{fam['brand']} mirrorless camera {{code}}, {mp} megapixel sensor, {v} video, {lk}."
    return fam, text, price


def _shoes(rng, fam=None):
    b = {"Nike": 1.3, "Adidas": 1.2, "Puma": 1.0, "Reebok": .9, "Asics": 1.15, "Skechers": .85}
    t = {"running": 80, "trail": 95, "casual": 60, "boots": 110, "basketball": 100}
    if fam is None:
        fam = dict(brand=_pick(rng, b), type=_pick(rng, t))
    m = {"mesh": 0, "leather": 25, "knit": 10, "suede": 20}
    mk = _pick(rng, m); cush = int(rng.integers(0, 2))
    price = (t[fam["type"]] + m[mk] + 20 * cush) * b[fam["brand"]]
    text = f"{fam['brand']} {fam['type']} shoes {{code}}, {mk} upper{', extra cushioned sole' if cush else ''}."
    return fam, text, price


def _watch(rng, fam=None):
    b = {"Garmin": 1.4, "Fitbit": 1.0, "Casio": .6, "Fossil": .9, "Samsung": 1.2, "Apple": 2.0}
    k = {"smart": 220, "analog": 80, "hybrid": 150}
    if fam is None:
        fam = dict(brand=_pick(rng, b), kind=_pick(rng, k))
    gps, amoled, lte = (int(x) for x in rng.integers(0, 2, 3)); wr = int(rng.choice([50, 100]))
    price = (k[fam["kind"]] + 40 * gps + 60 * amoled + 90 * lte + (10 if wr == 50 else 25)) * b[fam["brand"]]
    feats = ", ".join(x for x, f in (("GPS", gps), ("AMOLED display", amoled), ("LTE", lte)) if f) or "basic features"
    text = f"{fam['brand']} {fam['kind']} watch {{code}} with {feats}, water resistant to {wr}m."
    return fam, text, price


def _vacuum(rng, fam=None):
    b = {"Dyson": 2.2, "Shark": 1.0, "Roomba": 1.5, "Eufy": .9, "Bissell": .7, "Xiaomi": .75}
    k = {"robot": 250, "stick": 180, "canister": 150, "handheld": 60}
    if fam is None:
        fam = dict(brand=_pick(rng, b), kind=_pick(rng, k))
    pa = int(rng.choice([2000, 5000, 9000, 15000, 25000])); se, hepa = (int(x) for x in rng.integers(0, 2, 2))
    price = (k[fam["kind"]] + .008 * pa + 120 * se + 30 * hepa) * b[fam["brand"]]
    text = (f"{fam['brand']} {fam['kind']} vacuum cleaner {{code}}, {pa}Pa suction"
            f"{', self-emptying base' if se else ''}{', HEPA filter' if hepa else ''}.")
    return fam, text, price


CATEGORIES = {"laptop": _laptop, "headphones": _headphones, "camera": _camera,
              "shoes": _shoes, "watch": _watch, "vacuum": _vacuum}


def make_synthetic_catalog(n_families: int = 1500, seed: int = 0, fam_sigma: float = 0.08) -> pd.DataFrame:
    """Each family = 2-6 sibling products sharing brand/line and a latent price premium.

    fam_sigma = std of the log-premium shared by siblings but NOT visible in the text. It is
    the knob that decides whether retrieval can add information beyond the description."""
    rng = np.random.default_rng(seed)
    rows = []
    cats = list(CATEGORIES)
    for fid in range(n_families):
        cat = cats[rng.integers(len(cats))]
        fam, _, _ = CATEGORIES[cat](rng)
        line = _line_name(rng)
        fam_factor = np.exp(rng.normal(0, fam_sigma))  # shared latent "line" premium
        for _ in range(int(rng.integers(2, 7))):
            _, text, price = CATEGORIES[cat](rng, fam)
            price = price * fam_factor * np.exp(rng.normal(0, 0.10))
            text = text.replace(fam["brand"], f"{fam['brand']} {line}", 1)
            parts = [text.format(code=_code(rng))]
            parts += list(rng.choice(FLUFF, size=int(rng.integers(0, 3)), replace=False))
            rng.shuffle(parts[1:]) if len(parts) > 2 else None
            rows.append(dict(family_id=fid, category=cat, description=" ".join(parts),
                             price=round(float(price), 2)))
    return pd.DataFrame(rows)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"description", "price"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["description", "price"]).query("price > 0").reset_index(drop=True)
    if "family_id" not in df:
        df["family_id"] = np.arange(len(df))
    if "category" not in df:
        df["category"] = "all"
    return df


def split(df: pd.DataFrame, by_family: bool = False, seed: int = 0,
          fracs=(0.7, 0.1, 0.2)):
    """Return (train, calib, test) frames. by_family=True keeps whole families together,
    i.e. test products come from product lines never seen in training (harder, honest)."""
    rng = np.random.default_rng(seed)
    key = df["family_id"].values if by_family else np.arange(len(df))
    uniq = np.unique(key)
    rng.shuffle(uniq)
    n = len(uniq)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    which = {u: 0 for u in uniq[:a]} | {u: 1 for u in uniq[a:b]} | {u: 2 for u in uniq[b:]}
    part = np.array([which[k] for k in key])
    return tuple(df[part == i].reset_index(drop=True) for i in range(3))
