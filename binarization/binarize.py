# ocr_restore_binarize.py
import os
import cv2
import pathlib
import numpy as np
from typing import Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# Core utilities
# ---------------------------
def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to 8-bit grayscale for OpenCV ops.
    - BGR -> grayscale conversion via cv2.cvtColor.
    - Non-uint8 inputs are normalized to [0,255] then cast.
    Returns:
        uint8 HxW image in [0,255].
    """
    if img is None:
        raise ValueError("Input image is None")  # [18]
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [18]
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        g = gray.astype(np.float32)
        g -= g.min()
        denom = (g.max() - g.min()) or 1.0
        gray = np.clip(255.0 * g / denom, 0, 255).astype(np.uint8)
    return gray  # [18]


def upscale_lanczos(gray_u8: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """
    Upscale with Lanczos to place strokes above pixel grid before deconvolution.
    Args:
        gray_u8: uint8 grayscale.
        scale: resize factor (e.g., 2.0–3.0).
    Returns:
        uint8 upscaled image.
    """
    h, w = gray_u8.shape
    out = cv2.resize(gray_u8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
    return out  # [18]


# ---------------------------
# Richardson–Lucy deconvolution
# ---------------------------
def gaussian_psf(ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    """
    Create a normalized 2D Gaussian PSF.
    Args:
        ksize: odd kernel size.
        sigma: Gaussian standard deviation.
    Returns:
        float32 PSF normalized to sum=1.
    """
    if ksize % 2 == 0:
        ksize += 1
    k = cv2.getGaussianKernel(ksize, sigma)
    psf = (k @ k.T).astype(np.float32)
    psf /= psf.sum() + 1e-8
    return psf  # [19]


def rl_deconvolve(gray_u8: np.ndarray, psf: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Richardson–Lucy deconvolution with a known PSF.
    Algorithm:
        est_{t+1} = est_t * ( (img / (est_t (*) psf)) (*) psf^flip )
    where (*) is convolution and psf^flip is PSF rotated 180°.
    Args:
        gray_u8: uint8 grayscale input.
        psf: float32 PSF, sum=1.
        iterations: number of iterations (e.g., 5–15).
    Returns:
        uint8 deconvolved image (clipped).
    """
    img = gray_u8.astype(np.float32) + 1e-6
    est = img.copy()
    psf_flip = cv2.flip(psf, -1)
    for _ in range(iterations):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REPLICATE) + 1e-6
        ratio = img / conv
        corr = cv2.filter2D(ratio, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
        est = est * corr
        est = np.clip(est, 0, 255)  # prevent blow‑up
    return est.astype(np.uint8)  # [19][20]


# ---------------------------
# Edge‑preserving denoising (no sharpening)
# ---------------------------
def anisotropic_diffusion(gray_u8: np.ndarray, n_iter: int = 8, k: float = 15.0, lam: float = 0.15) -> np.ndarray:
    """
    Perona–Malik anisotropic diffusion to smooth noise while preserving edges.
    Args:
        gray_u8: uint8 grayscale input.
        n_iter: iterations (6–12 typical).
        k: conductance; lower preserves edges more.
        lam: time step (<=0.25 for 4‑neighborhood stability).
    Returns:
        uint8 denoised image.
    """
    I = gray_u8.astype(np.float32)
    for _ in range(n_iter):
        n = np.roll(I, -1, 0) - I
        s = np.roll(I,  1, 0) - I
        e = np.roll(I, -1, 1) - I
        w = np.roll(I,  1, 1) - I
        cN = np.exp(-(n/k)**2)
        cS = np.exp(-(s/k)**2)
        cE = np.exp(-(e/k)**2)
        cW = np.exp(-(w/k)**2)
        I += lam * (cN*n + cS*s + cE*e + cW*w)
    return np.clip(I, 0, 255).astype(np.uint8)  # [21][11]


# ---------------------------
# Illumination normalization
# ---------------------------
def illumination_correct(gray_u8: np.ndarray, bg_kernel: int = 151, mode: str = "divide") -> np.ndarray:
    """
    Remove slow‑varying background by Gaussian background division/subtraction.
    Args:
        gray_u8: uint8 grayscale.
        bg_kernel: odd Gaussian kernel size (e.g., 101–301).
        mode: 'divide' (recommended) or 'subtract'.
    Returns:
        uint8 normalized image.
    """
    if bg_kernel % 2 == 0:
        bg_kernel += 1
    bg = cv2.GaussianBlur(gray_u8, (bg_kernel, bg_kernel), 0)
    g = gray_u8.astype(np.float32)
    b = np.maximum(bg.astype(np.float32), 1.0)
    if mode == "divide":
        n = g / b
        out = cv2.normalize(n, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        n = g - b
        out = cv2.normalize(n, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out  # [22]


# ---------------------------
# Thresholding (Sauvola and Otsu)
# ---------------------------
def sauvola_single(gray_u8: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0, preblur_ksize: int = 0) -> np.ndarray:
    """
    Single‑scale Sauvola binarization on a prepared grayscale.
    Args:
        gray_u8: uint8 grayscale input.
        window: odd local window (≈ x‑height).
        k: Sauvola parameter (0.15–0.35).
        R: dynamic range (128 for 8‑bit).
        preblur_ksize: optional small Gaussian pre‑blur (odd).
    Returns:
        uint8 binary (0=foreground, 255=background).
    """
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be odd and >=3")  # [13]
    g = gray_u8.astype(np.float32)
    if preblur_ksize and preblur_ksize > 0:
        g = cv2.GaussianBlur(g, (preblur_ksize|1, preblur_ksize|1), 0).astype(np.float32)
    mean = cv2.boxFilter(g, -1, (window, window), normalize=True)
    mean2 = cv2.boxFilter(g*g, -1, (window, window), normalize=True)
    var = np.maximum(mean2 - mean*mean, 0.0)
    std = np.sqrt(var)
    T = mean * (1.0 + k * (std / (R + 1e-6) - 1.0))
    bw = np.where(g <= T, 0, 255).astype(np.uint8)
    return bw  # [10][13]


def sauvola_multiscale(gray_u8: np.ndarray, windows: Tuple[int, int] = (25, 35), k: float = 0.2, R: float = 128.0, preblur_ksize: int = 0) -> np.ndarray:
    """
    Multi‑scale Sauvola by OR‑fusing two window sizes to preserve thin and thick strokes.
    Args:
        windows: two odd window sizes (e.g., 25 and 35).
    Returns:
        uint8 fused binary (0=foreground, 255=background).
    """
    w1, w2 = windows
    bw1 = sauvola_single(gray_u8, w1, k, R, preblur_ksize)
    bw2 = sauvola_single(gray_u8, w2, k, R, preblur_ksize)
    fused = np.minimum(bw1, bw2).astype(np.uint8)
    return fused  # [13]


def otsu_threshold(gray_u8: np.ndarray, denoise_ksize: int = 3) -> np.ndarray:
    """
    Global Otsu threshold after optional light denoising.
    Args:
        gray_u8: uint8 grayscale input.
        denoise_ksize: small odd kernel for Gaussian blur.
    Returns:
        uint8 binary (0=foreground, 255=background).
    """
    g = gray_u8
    if denoise_ksize and denoise_ksize > 0:
        g = cv2.GaussianBlur(g, (denoise_ksize|1, denoise_ksize|1), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw  # [19]


# ---------------------------
# Light morphology
# ---------------------------
def post_morphology(bw: np.ndarray, open_ksize: int = 1, close_ksize: int = 1) -> np.ndarray:
    """
    Clean up salt noise and small gaps with tiny opening/closing.
    Args:
        bw: uint8 binary, 0=foreground, 255=background.
        open_ksize: opening kernel size (use 0/1 to keep very small).
        close_ksize: closing kernel size (use 0/1 to keep very small).
    Returns:
        uint8 binary after minimal morphology.
    """
    out = bw.copy()
    if open_ksize and open_ksize > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, se)
    if close_ksize and close_ksize > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, se)
    return out  # [23]


# ---------------------------
# Full single‑image pipeline
# ---------------------------
def restore_then_binarize(gray_u8: np.ndarray,
                          scale: float = 3.0,
                          psf_size: int = 5,
                          psf_sigma: float = 1.2,
                          rl_iters: int = 10,
                          diff_iters: int = 8,
                          diff_k: float = 15.0,
                          diff_lambda: float = 0.15,
                          illum_kernel: int = 151,
                          sauvola_windows: Tuple[int, int] = (25, 35),
                          sauvola_k: float = 0.2,
                          sauvola_R: float = 128.0,
                          preblur_ksize: int = 0,
                          morph_open: int = 1,
                          morph_close: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete restore‑then‑binarize pipeline producing both Sauvola (multi‑scale) and Otsu outputs.
    Steps:
      1) Upscale with Lanczos.
      2) Richardson–Lucy deconvolution with small Gaussian PSF.
      3) Perona–Malik anisotropic diffusion.
      4) Illumination correction via background division.
      5) Multi‑scale Sauvola and Otsu thresholds.
      6) Minimal morphology.
    Returns:
        (bw_sauvola, bw_otsu): two uint8 binaries.
    """
    up = upscale_lanczos(gray_u8, scale=scale)  # [22]
    psf = gaussian_psf(psf_size, psf_sigma)     # [2]
    deconv = rl_deconvolve(up, psf, iterations=rl_iters)  # [2]
    den = anisotropic_diffusion(deconv, n_iter=diff_iters, k=diff_k, lam=diff_lambda)  # [8]
    flat = illumination_correct(den, bg_kernel=illum_kernel, mode="divide")  # [22]
    bw_s = sauvola_multiscale(flat, windows=sauvola_windows, k=sauvola_k, R=sauvola_R, preblur_ksize=preblur_ksize)  # [13]
    bw_o = otsu_threshold(flat, denoise_ksize=3)  # [19]
    bw_s = post_morphology(bw_s, morph_open, morph_close)  # [23]
    bw_o = post_morphology(bw_o, morph_open, morph_close)  # [23]
    return bw_s, bw_o  # [22]


# ---------------------------
# Batch processing
# ---------------------------
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pbm", ".pgm"}


def is_image(path: str) -> bool:
    """
    Return True if file extension looks like an image.
    """
    return pathlib.Path(path).suffix.lower() in SUPPORTED_EXT  # [24]


def iter_images(input_dir: str, recursive: bool = True) -> Iterable[str]:
    """
    Yield image paths from a directory, optionally recursively with os.walk.
    """
    input_dir = os.path.abspath(input_dir)
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                p = os.path.join(root, f)
                if is_image(p):
                    yield p  # [24]
    else:
        for f in os.listdir(input_dir):
            p = os.path.join(input_dir, f)
            if os.path.isfile(p) and is_image(p):
                yield p  # [24]


def batch_pipeline(input_dir: str,
                   output_dir: str,
                   recursive: bool = True,
                   n_workers: int = 0,
                   suffix_s: str = "sauvola_ms",
                   suffix_o: str = "otsu",
                   # pipeline params
                   scale: float = 3.0,
                   psf_size: int = 5,
                   psf_sigma: float = 1.2,
                   rl_iters: int = 10,
                   diff_iters: int = 8,
                   diff_k: float = 15.0,
                   diff_lambda: float = 0.15,
                   illum_kernel: int = 151,
                   sauvola_windows: Tuple[int, int] = (25, 35),
                   sauvola_k: float = 0.2,
                   sauvola_R: float = 128.0,
                   preblur_ksize: int = 0,
                   morph_open: int = 1,
                   morph_close: int = 1) -> None:
    """
    Run the full pipeline over a folder tree and write Sauvola and Otsu images side by side.
    Output filenames:
        <stem>_<suffix_s>.ext and <stem>_<suffix_o>.ext in a mirrored directory structure.
    Parallelism:
        Uses ThreadPoolExecutor for per‑file parallel I/O when n_workers > 1.
    """
    base_in = pathlib.Path(os.path.abspath(input_dir))
    base_out = os.path.abspath(output_dir)
    os.makedirs(base_out, exist_ok=True)

    def _process_one(path: str):
        rel = pathlib.Path(path).relative_to(base_in)
        stem, ext = rel.stem, rel.suffix
        out_dir = os.path.join(base_out, str(rel.parent))
        os.makedirs(out_dir, exist_ok=True)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return (path, False, "imread None")  # [24]
        gray = to_gray_uint8(img)  # [22]
        bw_s, bw_o = restore_then_binarize(
            gray,
            scale, psf_size, psf_sigma, rl_iters,
            diff_iters, diff_k, diff_lambda,
            illum_kernel, sauvola_windows, sauvola_k, sauvola_R,
            preblur_ksize, morph_open, morph_close
        )  # [2]
        ok1 = cv2.imwrite(os.path.join(out_dir, f"{stem}_{suffix_s}{ext}"), bw_s)  # [24]
        ok2 = cv2.imwrite(os.path.join(out_dir, f"{stem}_{suffix_o}{ext}"), bw_o)  # [24]
        ok = bool(ok1 and ok2)
        return (path, ok, "ok" if ok else "imwrite failed")  # [24]

    files = list(iter_images(str(base_in), recursive=recursive))  # [24]
    if not files:
        print("No images found.")  # [24]
        return  # [24]
    if n_workers and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for fut in as_completed([ex.submit(_process_one, p) for p in files]):
                p, ok, msg = fut.result()
                if not ok:
                    print(f"[FAIL] {p}: {msg}")  # [24]
    else:
        for p in files:
            pth, ok, msg = _process_one(p)
            if not ok:
                print(f"[FAIL] {pth}: {msg}")  # [24]


# ---------------------------
# Example main
# ---------------------------
if __name__ == "__main__":
    # Edit paths and params as needed
    batch_pipeline(
        input_dir=r"C:\Users\adity\OneDrive\Desktop\Python\OCR_from_scratch\data",
        output_dir=r"C:\Users\adity\OneDrive\Desktop\Python\OCR_from_scratch\data",
        recursive=True,
        n_workers=6,
        suffix_s="sauvola_ms_rl",
        suffix_o="otsu_rl",
        scale=3.0,
        psf_size=5, psf_sigma=1.2, rl_iters=10,
        diff_iters=8, diff_k=15.0, diff_lambda=0.15,
        illum_kernel=151,
        sauvola_windows=(25, 35), sauvola_k=0.20, sauvola_R=128.0,
        preblur_ksize=0,
        morph_open=1, morph_close=1
    )
