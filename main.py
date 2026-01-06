import os
import uuid
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------- IMPORTS --------------------
try:
    from document_processor import DocumentProcessor
    from translation_engine import TranslationEngine
except ImportError as e:
    raise ImportError(
        f"Missing supporting modules: {e}. "
        "Ensure 'document_processor.py' and 'translation_engine.py' exist."
    )

load_dotenv()

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- APP --------------------
app = FastAPI(
    title="AI Multi-Language Translation Studio",
    description="Professional eBook translation service",
    version="2.6.9"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "inputs"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- CONFIG --------------------
SUPPORTED_LANGS = {
    "spanish", "german", "hindi", "french", "italian",
    "portuguese", "russian", "japanese", "chinese", "arabic"
}

SUPPORTED_INPUTS = {".docx", ".pdf", ".epub", ".txt"}
SUPPORTED_OUTPUTS = {"docx", "pdf", "epub"}

MIME_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "epub": "application/epub+zip",
}

jobs_db: Dict[str, dict] = {}

processor = DocumentProcessor(output_dir=str(OUTPUT_DIR))
translator = TranslationEngine()

# -------------------- MODELS --------------------
class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    complete: bool = False
    error: bool = False
    message: Optional[str] = ""
    languages: List[str] = []

# -------------------- WORKFLOW --------------------
def run_translation_workflow(job_id: str, file_path: str, languages: List[str], formats: List[str]):
    try:
        jobs_db[job_id].update(status="Extracting text", progress=5)
        paragraphs = processor.extract_paragraphs(file_path)

        if not paragraphs:
            raise ValueError("Empty or unreadable document")

        total_langs = len(languages)

        for lang_idx, lang in enumerate(languages):
            translated = []
            jobs_db[job_id]["status"] = f"Translating {lang.title()}"

            for idx, para in enumerate(paragraphs):
                translated.append({
                    "text": translator.translate(para["text"], lang),
                    "style": para.get("style", "Normal")
                })

                progress = 10 + int(((lang_idx + idx / len(paragraphs)) / total_langs) * 80)
                jobs_db[job_id]["progress"] = progress

            jobs_db[job_id]["status"] = f"Exporting {lang.title()}"
            for fmt in formats:
                processor.save_by_format(translated, fmt, lang, job_id)

        jobs_db[job_id].update(
            status="Completed",
            progress=100,
            complete=True,
            message="All translations generated successfully."
        )

    except Exception as e:
        logger.exception("Translation failed")
        jobs_db[job_id].update(
            status="Failed",
            error=True,
            complete=True,
            message=str(e)
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# -------------------- ENDPOINTS --------------------
@app.post("/api/translate", status_code=202)
async def translate_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    languages: str = Query("spanish"),
    formats: str = Query("docx,pdf,epub")
):
    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()

    if ext not in SUPPORTED_INPUTS:
        raise HTTPException(400, "Unsupported input format")

    langs = [l.strip().lower() for l in languages.split(",") if l.strip().lower() in SUPPORTED_LANGS]
    fmts = [f.strip().lower() for f in formats.split(",") if f.strip().lower() in SUPPORTED_OUTPUTS]

    if not langs or not fmts:
        raise HTTPException(400, "Invalid language or format")

    input_path = INPUT_DIR / f"{job_id}{ext}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs_db[job_id] = {
        "status": "Queued",
        "progress": 0,
        "complete": False,
        "error": False,
        "filename": file.filename,
        "languages": [l.title() for l in langs],
    }

    background_tasks.add_task(run_translation_workflow, job_id, str(input_path), langs, fmts)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **jobs_db[job_id]}

@app.get("/api/download/{job_id}/{language}")
async def download_translation(
    job_id: str,
    language: str,
    file_format: str = Query("docx")
):
    job = jobs_db.get(job_id)
    if not job or not job["complete"]:
        raise HTTPException(404, "File not ready")

    fmt = file_format.lower().strip()
    lang = language.lower().strip()

    if fmt not in SUPPORTED_OUTPUTS:
        raise HTTPException(400, "Invalid format")

    file_path = OUTPUT_DIR / f"translated_{lang}_{job_id}.{fmt}"
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    original = Path(job["filename"]).stem
    download_name = f"{lang.title()}_{original}.{fmt}"

    return FileResponse(
        path=str(file_path),
        filename=download_name,
        media_type=MIME_TYPES[fmt],
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
