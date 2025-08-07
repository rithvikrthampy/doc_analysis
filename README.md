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