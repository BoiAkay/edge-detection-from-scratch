"""
============================================================
  Canny Edge Detection — From Scratch  (Pure NumPy, No OpenCV)
  Gradient Filters: Sobel (default), Prewitt, Scharr, Roberts
============================================================
  Usage examples:
    python canny_edge_detection.py                         # Sobel (default)
    python canny_edge_detection.py --filter prewitt
    python canny_edge_detection.py --filter scharr
    python canny_edge_detection.py --filter roberts
    python canny_edge_detection.py --filter sobel --url <your_image_url>
    python canny_edge_detection.py --filter sobel --gauss-sigma 2.0 --high-ratio 0.2
============================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for headless)
import matplotlib.pyplot as plt
import urllib.request
import argparse
import sys
import io
import ssl
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  STEP 0 ─ Download image
# ─────────────────────────────────────────────────────────────
SAMPLE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/board.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/img_with_chessboard.png",
]


def download_image(url: str | None = None,
                   save_path: str  = "input_image.jpg") -> np.ndarray:
    """Download an image from URL → NumPy float64 grayscale array (H × W)."""
    import random
    from PIL import Image

    if url is None:
        url = random.choice(SAMPLE_URLS)

    print(f"[↓] Downloading image from:\n    {url}")
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            raw = resp.read()
    except Exception as e:
        sys.exit(f"[✗] Download failed: {e}")

    img  = Image.open(io.BytesIO(raw)).convert("L")   # force grayscale
    arr  = np.array(img, dtype=np.float64)

    Path(save_path).write_bytes(raw)
    print(f"[✓] Image saved → {save_path}  |  shape={arr.shape}")
    return arr


# ─────────────────────────────────────────────────────────────
#  STEP 1 ─ Gaussian blur  (noise suppression)
# ─────────────────────────────────────────────────────────────
def gaussian_kernel(size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Construct a 2-D Gaussian kernel of given size and sigma."""
    k    = size // 2
    y, x = np.mgrid[-k : k+1, -k : k+1]
    g    = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()                       # normalised so kernel sums to 1


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2-D convolution with reflect padding — pure NumPy.
    Works for any kernel size; uses sliding window via stride tricks.
    """
    kh, kw  = kernel.shape
    ph, pw  = kh // 2, kw // 2
    padded  = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    H, W    = image.shape
    out     = np.zeros((H, W), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i+H, j : j+W]
    return out


def gaussian_blur(image: np.ndarray,
                  size:  int   = 5,
                  sigma: float = 1.4) -> np.ndarray:
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)


# ─────────────────────────────────────────────────────────────
#  STEP 2 ─ Gradient filters
# ─────────────────────────────────────────────────────────────
FILTERS: dict[str, dict] = {
    "sobel": {
        "Kx":  np.array([[-1,  0,  1],
                          [-2,  0,  2],
                          [-1,  0,  1]], dtype=np.float64),
        "Ky":  np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float64),
        "desc": "Sobel (3×3, weighted — DEFAULT)",
    },
    "prewitt": {
        "Kx":  np.array([[-1,  0,  1],
                          [-1,  0,  1],
                          [-1,  0,  1]], dtype=np.float64),
        "Ky":  np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float64),
        "desc": "Prewitt (3×3, uniform weights)",
    },
    "scharr": {
        "Kx":  np.array([[ -3,  0,  3],
                          [-10,  0, 10],
                          [ -3,  0,  3]], dtype=np.float64),
        "Ky":  np.array([[ -3, -10, -3],
                          [  0,   0,  0],
                          [  3,  10,  3]], dtype=np.float64),
        "desc": "Scharr (3×3, high-accuracy rotation-invariant)",
    },
    "roberts": {
        "Kx":  np.array([[ 1,  0],
                          [ 0, -1]], dtype=np.float64),
        "Ky":  np.array([[ 0,  1],
                          [-1,  0]], dtype=np.float64),
        "desc": "Roberts Cross (2×2, diagonal — fine details)",
    },
}


def compute_gradients(blurred: np.ndarray,
                      filter_name: str = "sobel") -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the chosen gradient filter and return:
      magnitude : gradient strength (0–255 normalised)
      angle     : gradient direction in degrees (0–180)
    """
    f   = FILTERS[filter_name]
    Gx  = convolve2d(blurred, f["Kx"])
    Gy  = convolve2d(blurred, f["Ky"])

    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255.0    # normalise to 0-255
    angle     = np.rad2deg(np.arctan2(Gy, Gx)) % 180  # fold to 0-180°
    return magnitude, angle


# ─────────────────────────────────────────────────────────────
#  STEP 3 ─ Non-Maximum Suppression (NMS)
# ─────────────────────────────────────────────────────────────
def non_maximum_suppression(magnitude: np.ndarray,
                             angle:     np.ndarray) -> np.ndarray:
    """
    Thin edges by suppressing non-local-maxima along the gradient direction.

    The gradient angle is quantised into 4 bins:
      0°   → compare left / right          (horizontal edge)
      45°  → compare top-right / bot-left  (diagonal ↗ edge)
      90°  → compare top / bottom          (vertical edge)
      135° → compare top-left / bot-right  (diagonal ↘ edge)
    """
    H, W  = magnitude.shape
    nms   = np.zeros((H, W), dtype=np.float64)
    a     = angle

    # Vectorised NMS using shifted arrays (fast)
    def _check(r1, c1, r2, c2, mask):
        p = np.where(mask, magnitude[r1, c1], 0)
        q = np.where(mask, magnitude[r2, c2], 0)
        return (magnitude[1:H-1, 1:W-1] >= p) & (magnitude[1:H-1, 1:W-1] >= q)

    inner = magnitude[1:H-1, 1:W-1]
    th    = a[1:H-1, 1:W-1]
    keep  = np.zeros_like(inner, dtype=bool)

    # 0°  horizontal
    m0  = (th < 22.5) | (th >= 157.5)
    keep |= m0 & (inner >= magnitude[1:H-1, 2:W  ]) & (inner >= magnitude[1:H-1, 0:W-2])

    # 45° diagonal ↗
    m45  = (th >= 22.5) & (th < 67.5)
    keep |= m45 & (inner >= magnitude[0:H-2, 2:W  ]) & (inner >= magnitude[2:H,   0:W-2])

    # 90° vertical
    m90  = (th >= 67.5) & (th < 112.5)
    keep |= m90 & (inner >= magnitude[0:H-2, 1:W-1]) & (inner >= magnitude[2:H,   1:W-1])

    # 135° diagonal ↘
    m135 = (th >= 112.5) & (th < 157.5)
    keep |= m135 & (inner >= magnitude[2:H,   2:W  ]) & (inner >= magnitude[0:H-2, 0:W-2])

    nms[1:H-1, 1:W-1] = np.where(keep, inner, 0)
    return nms


# ─────────────────────────────────────────────────────────────
#  STEP 4 ─ Double threshold
# ─────────────────────────────────────────────────────────────
STRONG = 255
WEAK   = 75


def double_threshold(nms:        np.ndarray,
                     low_ratio:  float = 0.05,
                     high_ratio: float = 0.15) -> tuple[np.ndarray, float, float]:
    """
    Classify pixels:
      ≥ high_thresh  → STRONG (definite edge)
      ≥ low_thresh   → WEAK   (candidate edge)
      < low_thresh   → 0      (suppressed)
    """
    high_t  = nms.max() * high_ratio
    low_t   = high_t * low_ratio

    result  = np.zeros_like(nms, dtype=np.float64)
    result[nms >= high_t] = STRONG
    result[(nms >= low_t) & (nms < high_t)] = WEAK
    return result, high_t, low_t


# ─────────────────────────────────────────────────────────────
#  STEP 5 ─ Hysteresis edge tracking
# ─────────────────────────────────────────────────────────────
def hysteresis(img: np.ndarray) -> np.ndarray:
    """
    Keep a WEAK pixel only if it is 8-connected to at least one STRONG pixel.
    Uses an iterative flood-fill approach for correctness.
    """
    H, W   = img.shape
    strong = (img == STRONG)
    weak   = (img == WEAK)
    out    = strong.astype(np.float64) * STRONG

    # Iterative propagation: expand STRONG into 8-connected WEAK pixels
    changed = True
    while changed:
        # Dilate current strong mask by 1 pixel (8-connectivity)
        
        padded   = np.pad(out > 0, 1, mode="constant").astype(bool)
        dilated  = (
            padded[0:H, 0:W] | padded[0:H, 1:W+1] | padded[0:H, 2:W+2] |
            padded[1:H+1, 0:W] |                      padded[1:H+1, 2:W+2] |
            padded[2:H+2, 0:W] | padded[2:H+2, 1:W+1] | padded[2:H+2, 2:W+2]
        ).astype(bool)
        new_strong = weak & dilated & ~(out == STRONG)
        if not new_strong.any():
            changed = False
        else:
            out[new_strong] = STRONG

    # Suppress remaining weak pixels not connected to strong
    out[out != STRONG] = 0
    return out


# ─────────────────────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────────────────────
def canny_edge_detection(
    image:        np.ndarray,
    filter_name:  str   = "sobel",
    gauss_size:   int   = 5,
    gauss_sigma:  float = 1.4,
    low_ratio:    float = 0.05,
    high_ratio:   float = 0.15,
    verbose:      bool  = True,
) -> dict:
    """
    Run the complete 5-stage Canny pipeline and return all intermediate images.

    Parameters
    ----------
    image       : Grayscale input (H × W, float64)
    filter_name : One of 'sobel', 'prewitt', 'scharr', 'roberts'
    gauss_size  : Gaussian kernel size (odd integer, default 5)
    gauss_sigma : Gaussian std-dev (default 1.4)
    low_ratio   : Low-threshold = high_threshold × low_ratio  (default 0.05)
    high_ratio  : High-threshold = max_gradient × high_ratio  (default 0.15)
    verbose     : Print progress (default True)

    Returns
    -------
    dict with keys: original, blurred, gradient, nms, thresholded, edges
    """
    stages = {"original": image}

    if verbose:
        print(f"\n{'─'*56}")
        print(f"  Canny Edge Detection  |  filter = {filter_name.upper()}")
        print(f"  Image shape : {image.shape}")
        print(f"  Gaussian    : kernel={gauss_size}×{gauss_size}, σ={gauss_sigma}")
        print(f"  Thresholds  : low_ratio={low_ratio}, high_ratio={high_ratio}")
        print(f"{'─'*56}")

    # Stage 1 — Gaussian blur
    blurred = gaussian_blur(image, gauss_size, gauss_sigma)
    stages["blurred"] = blurred
    if verbose: print("[1/5] ✓ Gaussian blur applied")

    # Stage 2 — Gradient computation
    magnitude, angle = compute_gradients(blurred, filter_name)
    stages["gradient"] = magnitude
    if verbose: print(f"[2/5] ✓ Gradients computed  (max={magnitude.max():.1f})")

    # Stage 3 — Non-maximum suppression
    nms = non_maximum_suppression(magnitude, angle)
    stages["nms"] = nms
    if verbose: print("[3/5] ✓ Non-maximum suppression done")

    # Stage 4 — Double threshold
    thresholded, high_t, low_t = double_threshold(nms, low_ratio, high_ratio)
    stages["thresholded"] = thresholded
    if verbose: print(f"[4/5] ✓ Double threshold  (high={high_t:.1f}, low={low_t:.1f})")

    # Stage 5 — Hysteresis
    edges = hysteresis(thresholded)
    stages["edges"] = edges
    if verbose: print("[5/5] ✓ Hysteresis edge tracking complete")

    return stages


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────
def visualise(stages: dict, filter_name: str,
              save_path: str = "canny_result.png") -> None:

    panel_info = [
        ("original",    "① Original (Grayscale)",        "gray"),
        ("blurred",     "② Gaussian Blur (noise↓)",      "gray"),
        ("gradient",    "③ Gradient Magnitude",          "hot"),
        ("nms",         "④ Non-Max Suppression",         "gray"),
        ("thresholded", "⑤ Double Threshold",            "gray"),
        ("edges",       "⑥  Final Canny Edges",          "gray"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"Canny Edge Detection  —  {filter_name.upper()} Filter",
        fontsize=17, color="#f0f0f0", fontweight="bold", y=1.01,
    )

    for ax, (key, title, cmap) in zip(axes.flat, panel_info):
        ax.imshow(stages[key], cmap=cmap, interpolation="nearest")
        ax.set_title(title, color="#d0d0d0", fontsize=11, pad=7)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Canny Edge Detection from scratch (pure NumPy, no OpenCV).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    filter_help = "Gradient filter (default: sobel):\n" + "\n".join(
        f"  {k:8s}: {v['desc']}" for k, v in FILTERS.items()
    )
    p.add_argument("--filter", "-f",
                   choices=list(FILTERS.keys()), default="sobel",
                   metavar="FILTER", help=filter_help)
    p.add_argument("--url",         default=None,
                   help="Custom image URL (uses random sample if omitted)")
    p.add_argument("--gauss-size",  type=int,   default=5,
                   help="Gaussian kernel size, must be odd (default: 5)")
    p.add_argument("--gauss-sigma", type=float, default=1.4,
                   help="Gaussian sigma (default: 1.4)")
    p.add_argument("--low-ratio",   type=float, default=0.05,
                   help="Low threshold ratio  (default: 0.05)")
    p.add_argument("--high-ratio",  type=float, default=0.15,
                   help="High threshold ratio (default: 0.15)")
    p.add_argument("--no-plot",     action="store_true",
                   help="Skip saving the matplotlib figure")
    p.add_argument("--output", "-o", default="canny_result.png",
                   help="Output figure filename (default: canny_result.png)")
    return p


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    print("\n" + "═"*56)
    print("  CANNY EDGE DETECTOR  |  Pure NumPy  |  No OpenCV")
    print("  Available filters: " + " | ".join(FILTERS.keys()))
    print("═"*56)

    image  = download_image(url=args.url)
    stages = canny_edge_detection(
        image,
        filter_name=args.filter,
        gauss_size=args.gauss_size,
        gauss_sigma=args.gauss_sigma,
        low_ratio=args.low_ratio,
        high_ratio=args.high_ratio,
    )

    if not args.no_plot:
        visualise(stages, filter_name=args.filter, save_path=args.output)

    edges     = stages["edges"]
    total     = edges.size
    edge_px   = int((edges == STRONG).sum())
    pct       = edge_px / total * 100
    print(f"\n[Stats] Edge pixels: {edge_px:,} / {total:,}  ({pct:.2f}%)")
    print(f"[Done]  Saved → {args.output}\n")


if __name__ == "__main__":
    main()
