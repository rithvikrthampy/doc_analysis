from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import cv2
import numpy as np
import io
import time
from pathlib import Path
import tempfile
import os

from pipeline import DocumentProcessor, imread_bgr

app = FastAPI(title="Document Processing API", version="1.0.0")

processor = DocumentProcessor()

def process_uploaded_file(file_content: bytes, filename: str) -> Dict[str, Any]:
    try:
        nparr = np.frombuffer(file_content, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if bgr is None:
            raise ValueError(f"Could not decode image: {filename}")
        
        result = processor.process_image(bgr, filename)
        
        temp_dir = Path("temp_output")
        temp_dir.mkdir(exist_ok=True)
        
        import os
        name_without_ext = os.path.splitext(filename)[0]
        temp_path = temp_dir / f"processed_{name_without_ext}.png"
        cv2.imwrite(str(temp_path), result["cropped_image"])
        
        return {
            "filename": filename,
            "angle_deg": result["angle_deg"],
            "base64": result["base64"],
            "execution_time_ms": result["execution_time_ms"],
            "cropped_image_path": str(temp_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing {filename}: {str(e)}")

@app.post("/process/single")
async def process_single_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_content = await file.read()
    result = process_uploaded_file(file_content, file.filename)
    
    return JSONResponse(content=result)

@app.post("/process/multiple")
async def process_multiple_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    total_start_time = time.perf_counter()
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image",
                "execution_time_ms": 0
            })
            continue
        
        try:
            file_content = await file.read()
            result = process_uploaded_file(file_content, file.filename)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "execution_time_ms": 0
            })
    
    total_execution_time = (time.perf_counter() - total_start_time) * 1000.0
    
    return JSONResponse(content={
        "results": results,
        "total_execution_time_ms": round(total_execution_time, 1),
        "total_files": len(files),
        "successful_files": len([r for r in results if "error" not in r])
    })

@app.get("/")
async def root():
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/process/single": "Process a single image file",
            "/process/multiple": "Process multiple image files",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": processor.rembg_model}

@app.on_event("startup")
async def startup_event():
    print("Document Processing API starting up...")
    print(f"Using rembg model: {processor.rembg_model}")

@app.on_event("shutdown")
async def shutdown_event():
    print("Document Processing API shutting down...")
    temp_dir = Path("temp_output")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)