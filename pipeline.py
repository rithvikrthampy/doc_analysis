import argparse, base64, io, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from rembg import remove, new_session

EDGE_TIGHTEN_PX = 20
PAD_RATIO = 0.015


def imread_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def encode_base64_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf).decode("utf-8")

def order_quad_pts(quad: np.ndarray) -> np.ndarray:
    s = quad.sum(axis=1); d = np.diff(quad, axis=1).ravel()
    o = np.zeros((4,2), np.float32)
    o[0] = quad[np.argmin(s)]   # tl
    o[2] = quad[np.argmax(s)]   # br
    o[1] = quad[np.argmin(d)]   # tr
    o[3] = quad[np.argmax(d)]   # bl
    return o

def rotation_ccw_deg_from_quad(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    dx = float(tr[0] - tl[0])
    dy = float(tr[1] - tl[1])
    angle_top = np.degrees(np.arctan2(dy, dx))
    while angle_top <= -90.0: angle_top += 180.0
    while angle_top >   90.0: angle_top -= 180.0
    return -angle_top


class DocumentProcessor:
    def __init__(self, rembg_model: str = "isnet-general-use"):
        self.rembg_model = rembg_model
        self._session = new_session(model_name=rembg_model, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    
    def rembg_alpha(self, bgr: np.ndarray, tag: str) -> np.ndarray:
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".png", bgr)
        rgba_bytes = remove(buf.tobytes(), session=self._session)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"[rembg] {tag}: {ms:.1f} ms")

        rgba = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
        a = np.array(rgba)[:, :, 3]

        mask = (a > 0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), 1
        )
        mask = cv2.medianBlur(mask, 3)
        return mask
    
    def quad_from_alpha(self, mask: np.ndarray) -> np.ndarray | None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) <= 1: return None
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.float32)
        return order_quad_pts(quad)
    
    def process_image(self, bgr: np.ndarray, name: str) -> dict:
        start_time = time.perf_counter()
        
        mask = self.rembg_alpha(bgr, name)
        quad = self.quad_from_alpha(mask)
        if quad is None:
            H, W = bgr.shape[:2]
            quad = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
            print(f"[{name}] NOTE: rembg mask had no usable contour; using full image bounds.")
            angle_ccw = 0.0
        else:
            angle_ccw = rotation_ccw_deg_from_quad(quad)

        warped = warp_no_pad(bgr, quad)
        snapped = snap_to_edges(warped, max_shift=EDGE_TIGHTEN_PX)
        padded = add_white_padding(snapped, pad_ratio=PAD_RATIO)
        
        b64 = encode_base64_png(padded)
        execution_time = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "angle_deg": round(angle_ccw, 2),
            "cropped_image": padded,
            "base64": b64,
            "execution_time_ms": round(execution_time, 1)
        }


def warp_no_pad(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = quad
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    W = int(round(max(wA, wB))); H = int(round(max(hA, hB)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(
        bgr, M, (W, H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
    )

def snap_to_edges(warped: np.ndarray, max_shift: int = EDGE_TIGHTEN_PX) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)).astype(np.int32)
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)).astype(np.int32)
    lim = max(1, min(max_shift, H//4, W//4))
    top    = int(np.argmax(sobel_y[0:lim, :].sum(axis=1)))
    bottom = H - lim + int(np.argmax(sobel_y[H-lim:H, :].sum(axis=1)))
    left   = int(np.argmax(sobel_x[:, 0:lim].sum(axis=0)))
    right  = W - lim + int(np.argmax(sobel_x[:, W-lim:W].sum(axis=0)))
    top = np.clip(top, 0, H-2); bottom = np.clip(bottom, top+1, H-1)
    left = np.clip(left, 0, W-2); right = np.clip(right, left+1, W-1)
    return warped[top:bottom+1, left:right+1]

def add_white_padding(bgr: np.ndarray, pad_ratio: float = PAD_RATIO) -> np.ndarray:
    H, W = bgr.shape[:2]
    pad = int(round(pad_ratio * max(W, H)))
    return cv2.copyMakeBorder(
        bgr, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT, value=(255,255,255)
    )

def process_one(path: Path, out_dir: Path, processor: DocumentProcessor = None):
    if processor is None:
        processor = DocumentProcessor()
    
    name = path.stem
    bgr = imread_bgr(str(path))
    
    result = processor.process_image(bgr, name)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    cv2.imwrite(str(out_path), result["cropped_image"])
    
    print(f"angle_deg={result['angle_deg']}")
    print(result["base64"])
    print(f"[saved] {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--rembg-model", default="isnet-general-use",
                    help="rembg model name (default: isnet-general-use) or u2net")
    args = ap.parse_args()

    print(f"[rembg] using model: {args.rembg_model}")
    processor = DocumentProcessor(args.rembg_model)

    in_path = Path(args.input)
    out_dir = Path("out")

    files = []
    if in_path.is_dir():
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            files.extend(sorted(in_path.glob(ext)))
    else:
        files = [in_path]

    for ip in files:
        print(f"\n=== {ip} ===")
        try:
            process_one(ip, out_dir, processor)
        except Exception as e:
            print(f"[ERROR] {ip}: {e}")

if __name__ == "__main__":
    main()
