# preprocessing_simple.py
# SPDX-License-Identifier: MIT

import os
import cv2
import glob
import json
import pathlib
import numpy as np
from typing import Tuple, List, Dict

# ---------------------------
# Utilities
# ---------------------------
def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert any image to uint8 grayscale.
    """
    if img is None:
        raise ValueError("Input image is None")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        g -= g.min()
        gray = np.clip(255.0 * g / ((g.max() - g.min()) or 1.0), 0, 255).astype(np.uint8)
    return gray  # HxW array (shape=H, shape[21]=W). [22]

def ensure_dir(p: str) -> None:
    """
    Create directory if missing.
    """
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)  # Creates output folders safely. [22]

# ---------------------------
# Skew via Probabilistic Hough
# ---------------------------
def estimate_skew_hough(gray_u8: np.ndarray,
                        canny1: int = 80, canny2: int = 200,
                        min_line_len: int = 40, max_line_gap: int = 10,
                        angle_clip_deg: float = 30.0) -> float:
    """
    Estimate global skew using Probabilistic Hough Lines on Canny edges.

    Steps:
      - Edge map via Canny.
      - HoughLinesP to get line segments.
      - Keep near-horizontal segments (|angle| <= angle_clip_deg around 0 or 180).
      - Return length-weighted median angle (deg), where positive is clockwise.

    Returns:
        float: estimated skew angle in degrees.
    """
    edges = cv2.Canny(gray_u8, canny1, canny2)  # Edge detection before Hough improves line votes. [2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=60,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)  # PHT variant. [2]
    if lines is None or len(lines) == 0:
        return 0.0  # No lines found; assume no skew. [2]
    angles = []
    weights = []
    for L in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, L)
        dx, dy = (x2 - x1), (y2 - y1)
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))  # Segment angle in degrees. [2]
        # Normalize around 0 degrees for text lines (map to [-90, 90])
        ang = ((ang + 90.0) % 180.0) - 90.0
        if abs(ang) <= angle_clip_deg:
            length = np.hypot(dx, dy)
            angles.append(ang)
            weights.append(length)
    if not angles:
        return 0.0
    # Weighted median for robustness
    order = np.argsort(angles)
    angles = np.array(angles)[order]
    weights = np.array(weights)[order]
    csum = np.cumsum(weights)
    idx = int(np.searchsorted(csum, csum[-1] * 0.5))
    return float(angles[min(idx, len(angles) - 1)])  # Robust skew estimate. [2][18]

def deskew(img: np.ndarray, angle_deg: float, expand: bool = True) -> np.ndarray:
    """
    Rotate image by angle_deg using getRotationMatrix2D; expand canvas to keep content.
    """
    gray = to_gray_uint8(img)
    h, w = gray.shape  # HxW indexing. [22]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)  # (cx, cy) center. [23]
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (nW / 2) - w / 2
        M[1, 2] += (nH / 2) - h / 2
        return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # (W,H). [24]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # (W,H). [25]

# ---------------------------
# Binarize and line segmentation
# ---------------------------
def sauvola(gray_u8: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """
    Sauvola thresholding (0=foreground, 255=background).
    """
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be odd and >=3")
    g = gray_u8.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (window, window), normalize=True)
    mean2 = cv2.boxFilter(g * g, -1, (window, window), normalize=True)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    T = mean * (1.0 + k * (std / (R + 1e-6) - 1.0))
    return np.where(g <= T, 0, 255).astype(np.uint8)  # Standard adaptive binarization behavior. [26]

def lines_by_projection(bin_u8: np.ndarray, min_line_h: int = 12,
                        valley_ratio: float = 0.2, smooth_sigma: int = 5) -> List[Tuple[int, int]]:
    """
    Find line bands with horizontal projection valleys on binary text (0=ink).
    """
    text = (bin_u8 == 0).astype(np.uint8)
    proj = text.sum(axis=1).astype(np.float32)
    if smooth_sigma and smooth_sigma > 1:
        k = int(smooth_sigma) | 1
        proj = cv2.GaussianBlur(proj.reshape(-1, 1), (1, k), 0).ravel()
    thresh = float(proj.mean() * valley_ratio)
    bands, start = [], None
    for r, v in enumerate(proj):
        if v > thresh and start is None:
            start = r
        if ((v <= thresh) or (r == len(proj) - 1)) and start is not None:
            end = r if v <= thresh else r
            if end - start + 1 >= min_line_h:
                bands.append((start, end))
            start = None
    return bands  # Lightweight, fast, and common in OCR pipelines. [6]

# ---------------------------
# Baseline & x-height from CCs
# ---------------------------
def baseline_xheight_from_cc(line_bin: np.ndarray,
                             min_area: int = 8, max_aspect: float = 20.0,
                             keep_frac: float = 0.90, xh_percentile: float = 0.60) -> Tuple[np.ndarray, float]:
    """
    Use connected components stats to derive baseline and x-height quickly.

    Steps:
      - CC stats (areas, bbox).
      - Keep plausible text blobs (area and aspect filters).
      - For each x-bin across the width, collect bottoms (y+h-1) of blobs overlapping the bin.
      - Baseline per bin = 90th percentile, then Gaussian-smoothed and interpolated across width.
      - x-height = robust percentile of blob heights (filtering extreme tails).
    """
    h, w = line_bin.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(255 - line_bin, connectivity=8, ltype=cv2.CV_32S)  # background=0. [10]
    # Filter CCs
    bottoms = []
    heights = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        aspect = ww / max(1.0, float(hh))
        if aspect > max_aspect:
            continue
        bottoms.append((x, x + ww - 1, y + hh - 1))
        heights.append(hh)
    if not bottoms:
        # Fallback: straight baseline near bottom and nominal x-height
        return np.full(w, float(h - 2), dtype=np.float32), float(max(8, h * 0.3))

    # Bin the width to stabilize baseline
    n_bins = max(16, min(128, w // 8))
    edges = np.linspace(0, w, n_bins + 1, dtype=np.int32)
    base_bins = []
    for bi in range(n_bins):
        L, R = edges[bi], edges[bi + 1] - 1
        ys = [bt for (xl, xr, bt) in bottoms if not (xr < L or xl > R)]
        if ys:
            k = max(1, int(len(ys) * keep_frac))
            ys_sorted = np.sort(ys)
            base_bins.append(float(np.mean(ys_sorted[-k:])))  # lower envelope via top-k bottoms
        else:
            base_bins.append(np.nan)

    # Interpolate missing bins
    base_bins = np.array(base_bins, dtype=np.float32)
    idx = np.arange(n_bins)
    good = ~np.isnan(base_bins)
    if not np.any(good):
        baseline = np.full(w, float(h - 2), dtype=np.float32)
    else:
        base_interp = np.interp(idx, idx[good], base_bins[good])
        # Smooth and upsample to per-pixel baseline
        base_sm = cv2.GaussianBlur(base_interp.reshape(1, -1).astype(np.float32), (1, 9), 0).ravel()
        xgrid = np.linspace(0, n_bins - 1, w).astype(np.float32)
        baseline = np.interp(xgrid, idx, base_sm).astype(np.float32)
        baseline = np.clip(baseline, 0, h - 1)

    # Robust x-height from blob heights (trim extremes)
    H = np.array(heights, dtype=np.float32)
    q1, q3 = np.percentile(H, [25, 75])
    mask = (H >= max(3, q1 * 0.5)) & (H <= q3 * 1.75)
    Hf = H[mask] if np.any(mask) else H
    xh = float(np.percentile(Hf, xh_percentile * 100.0)) if Hf.size else float(max(8, h * 0.3))
    return baseline, xh  # Fast and leverages CC stats for robustness. [7][10]

# ---------------------------
# Normalization
# ---------------------------
def normalize_line(line_gray: np.ndarray, baseline_y: np.ndarray, xheight_px: float,
                   target_xh: int = 32, asc: float = 0.5, desc: float = 0.5,
                   interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Flatten baseline and scale x-height to target, returning H_out x W image.
    """
    h, w = line_gray.shape[:2]
    s = float(target_xh) / max(1e-3, xheight_px)
    out_h = int(round(target_xh * (1.0 + asc + desc)))
    out_w = w
    y0 = int(round(target_xh * desc))
    map_x = np.tile(np.arange(out_w, dtype=np.float32), (out_h, 1))
    rows = np.arange(out_h, dtype=np.float32).reshape(-1, 1)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)
    for c in range(out_w):
        map_y[:, c] = (rows[:, 0] - y0) / s + baseline_y[c]
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    return cv2.remap(line_gray, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)  # Standard remap. [25]

# ---------------------------
# Page pipeline
# ---------------------------
def process_page(img: np.ndarray,
                 bin_window: int = 25, bin_k: float = 0.2,
                 min_line_h: int = 12) -> Dict:
    """
    Deskew → Sauvola → line bands.
    """
    gray = to_gray_uint8(img)
    angle = estimate_skew_hough(gray)  # Hough-based skew is simple and robust. [2]
    desk = deskew(gray, angle, expand=True)  # Rotate with (W,H) size ordering. [24]
    bin_img = sauvola(desk, window=bin_window, k=bin_k, R=128.0)  # Adaptive thresholding. [26]
    bands = lines_by_projection(bin_img, min_line_h, valley_ratio=0.2, smooth_sigma=5)  # Fast segmentation. [6]
    lines = [{"bbox": (int(t), int(b)), "gray": desk[t:b+1, :], "bin": bin_img[t:b+1, :]} for (t, b) in bands]
    return {"angle": float(angle), "deskewed": desk, "bin": bin_img, "lines": lines}

def normalize_lines(page: Dict, target_xh: int = 32, asc: float = 0.5, desc: float = 0.5) -> List[Dict]:
    """
    Per line: CC-based baseline+x-height → normalized line image.
    """
    out = []
    for i, L in enumerate(page["lines"]):
        base, xh = baseline_xheight_from_cc(L["bin"])  # CC stats → baseline + x-height. [10]
        norm = normalize_line(L["gray"], base, xh, target_xh, asc, desc, interpolation=cv2.INTER_LINEAR)  # Warp. [25]
        out.append({"index": i, "bbox": L["bbox"], "xheight": float(xh), "normalized": norm})
    return out

# ---------------------------
# Hardcoded batch runner
# ---------------------------
def run(input_glob: str, out_dir: str,
        target_xh: int = 32, bin_window: int = 25, bin_k: float = 0.2,
        min_line_h: int = 12, asc: float = 0.5, desc: float = 0.5) -> None:
    """
    Run the pipeline for all files matched by input_glob and write outputs.
    """
    ensure_dir(out_dir)  # Prepare output directory. [22]
    for path in glob.glob(input_glob):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Could not read: {path}")
            continue
        page = process_page(img, bin_window, bin_k, min_line_h)  # Build page artifacts. [26]
        stem = pathlib.Path(path).stem
        cv2.imwrite(os.path.join(out_dir, f"{stem}_deskew.png"), page["deskewed"])  # Save deskewed page. [25]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_bin.png"), page["bin"])  # Save binarized page. [26]
        lines = normalize_lines(page, target_xh, asc, desc)  # Normalize lines. [10]
        meta = []
        for L in lines:
            out_line = os.path.join(out_dir, f"{stem}_line{L['index']:02d}_norm.png")
            cv2.imwrite(out_line, L["normalized"])
            meta.append({"index": L["index"], "bbox": L["bbox"], "xheight": L["xheight"]})
        with open(os.path.join(out_dir, f"{stem}_lines.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)  # Write metadata for inspection. [22]

if __name__ == "__main__":
    # Hardcoded paths (edit as needed)
    input_glob = r"C:\Users\adity\OneDrive\Desktop\Python\OCR_from_scratch\data\Screenshot 2025-09-11 125131.png"
    out_dir = r"C:\Users\adity\OneDrive\Desktop\Python\OCR_from_scratch\data\skew_baseline"
    run(input_glob, out_dir, target_xh=32, bin_window=25, bin_k=0.2, min_line_h=12, asc=0.5, desc=0.5)
