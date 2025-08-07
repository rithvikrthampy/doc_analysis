# Document‐Crop API  (rembg + FastAPI)

This service accepts a JPEG / PNG, segments the document with **rembg** (`isnet-general-use` provides more accuracy than `U2net`), warps it to a straight rectangle, trims the border, adds a white pad, and returns:

Response
```jsonc
{
  "angle_deg": 3.7,
  "image_base64": "iVBORw0K...",
  "filename": "India.png",
  "elapsed_ms": 912.4
}
```

---

## 1 · Install (Windows, macOS, Linux)

> Requires Python 3.12+

```bash
# clone
git clone https://github.com/YOUR_USERNAME/doc_analysis.git
cd doc_analysis

# create venv + install project deps
poetry install

## 2 · Run the API

```bash
poetry run python api.py
```

* Swagger UI: **<http://127.0.0.1:8000/docs>**

Results are also written to `./out/<name>.png`.

---

## 3 · Project layout

```
doc_analysis/
├─ api.py           # FastAPI wrapper
├─ pipeline.py      # core image logic (rembg → warp → pad)
├─ pyproject.toml   # Poetry dependencies
├─ sample_images/
└─ temp_out/        # output (auto-created)
```

---

## 4 · Key dependencies

| Package | Purpose |
|---------|---------|
| **rembg** | Foreground segmentation |
| **onnxruntime-directml** \| **onnxruntime** | GPU (DirectML) or CPU inference |
| **opencv-python-headless** | Perspective warp & edge cropping |
| **numpy** | Array math |
| **Pillow** | RGBA ↔ NumPy |
| **fastapi** | HTTP API |
| **uvicorn[standard]** | ASGI server |
| **python-multipart** | Multipart file uploads |

---

## Approach & Design Rationale  

- **End-to-end flow**  
  - *Segmentation* — `rembg` with the **IS-Net (“isnet-general-use”)** model isolates the document from any background.  
  - *Largest foreground contour* — the biggest connected component is convex-hulled, and `cv2.minAreaRect` fits a rotated box around it.  
  - *Perspective warp* — those four corners are mapped to an upright rectangle via `cv2.getPerspectiveTransform` + `warpPerspective`, removing the dominant skew.  
  - *Edge snap* — a light Sobel projection trims 1–20 px slivers so the crop hugs the real edges.  
  - *White padding* — a 1.5 % border is added with `cv2.copyBorder` for a clean margin.  
  - *Output* — the final PNG is saved to `temp_output/`, base-64 encoded, and the geometric rotation angle (`angle_deg`, CCW-positive) is returned.

- **Pre-processing specifics**  
  - Alpha from `rembg` is thresholded (`alpha > 0`), closed, and median-blurred to remove pinholes and edge fuzz.  
  - A single **DirectML GPU session** is created on startup (`["DmlExecutionProvider", "CPUExecutionProvider"]`) for speed; CPU fallback gives identical results.  
  - Warp destination size equals the longer side of the rotated box, preserving resolution.  
  - `angle_deg` is taken from the top edge of the quad, normalised to (-90°, 90°], then negated to reflect the correction applied.

- **Why a single `rembg` stage and no OCR deskew**  
  - One strong segmentation model copes with hands, shadows, and coloured tables consistently.  
  - Geometric warp removes skew without reading any text, so language and font are irrelevant.  
  - Latency stays below one second on consumer GPUs for 3-4 MP photos, meeting the assessment budget.

- **Libraries used**  
  - **rembg** — foreground segmentation.  
  - **onnxruntime-directml** (Windows) or **onnxruntime** (CPU) — runs the IS-Net ONNX model.  
  - **opencv-python-headless** — geometry, warp, edge trim, PNG encode.  
  - **numpy** — vector maths and angle calculations.  
  - **Pillow** — converts the RGBA output of `rembg` into NumPy.  
  - **fastapi** — REST endpoints (`/process/single`, `/process/multiple`).  
  - **uvicorn[standard]** — production ASGI server.  
  - **python-multipart** — handles image uploads for FastAPI.
