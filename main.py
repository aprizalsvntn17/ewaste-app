"""
Backend FastAPI — Aplikasi Klasifikasi Sampah Elektronik (E-Waste)
Endpoint utama: POST /predict
"""

import os
import json
import uuid
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import aiofiles

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ─── Konfigurasi Path ─────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
IS_VERCEL   = bool(os.environ.get("VERCEL"))

if IS_VERCEL:
    UPLOAD_DIR = Path("/tmp/uploads")
    DB_PATH    = Path("/tmp/ewaste.db")
else:
    UPLOAD_DIR = BASE_DIR / "uploads"
    DB_PATH    = BASE_DIR / "ewaste.db"

STATIC_DIR  = BASE_DIR / "static"
MODEL_PATH  = BASE_DIR / "model" / "ewaste_model.pth"
META_PATH   = BASE_DIR / "model" / "model_meta.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 10 Kategori Dataset & Info Penanganan ─────────────────────────────────────
CLASS_INFO = {
    "Battery": {
        "nama": "Baterai",
        "ikon": "🔋",
        "penanganan": [
            "Peringatan keselamatan: jangan ditekan, dibakar, atau dibasahi.",
            "Pisahkan dari sampah rumah tangga biasa dan simpan di wadah kering.",
            "Bawa ke titik pengumpulan baterai bekas atau drop-box e-waste terdekat."
        ],
        "bahaya": True,
    },
    "Keyboard": {
        "nama": "Aksesori (Keyboard)",
        "ikon": "⌨️",
        "penanganan": [
            "Lepaskan kabel atau baterai sebelum diserahkan.",
            "Serahkan ke bank sampah atau tempat pengumpulan e-waste.",
            "Komponen plastik dan logam di dalamnya bisa didaur ulang."
        ],
        "bahaya": False,
    },
    "Microwave": {
        "nama": "Peralatan Elektronik (Microwave)",
        "ikon": "📡",
        "penanganan": [
            "Cabut kabel listrik dan bersihkan bagian dalam sebelum diserahkan.",
            "Jangan dibongkar sendiri karena ada komponen kapasitor bertegangan tinggi.",
            "Serahkan ke fasilitas pengolahan sampah elektronik bersertifikat."
        ],
        "bahaya": False,
    },
    "Mobile": {
        "nama": "Ponsel",
        "ikon": "📱",
        "penanganan": [
            "Lakukan reset pabrik untuk menghapus seluruh data pribadi.",
            "Lepaskan kartu SIM dan memori sebelum diserahkan.",
            "Bawa ke konter daur ulang ponsel atau drop-box resmi e-waste."
        ],
        "bahaya": False,
    },
    "Mouse": {
        "nama": "Aksesori (Mouse)",
        "ikon": "🖱️",
        "penanganan": [
            "Keluarkan baterai jika menggunakan tipe nirkabel (wireless).",
            "Gulung kabel dengan rapi jika menggunakan kabel.",
            "Bawa ke tempat pengumpulan aksesori elektronik."
        ],
        "bahaya": False,
    },
    "PCB": {
        "nama": "Papan Sirkuit",
        "ikon": "🧩",
        "penanganan": [
            "Jangan dihancurkan atau dibakar karena mengandung logam berat.",
            "Simpan di wadah kering dan tidak mudah pecah.",
            "Serahkan ke pengepul atau fasilitas khusus daur ulang logam e-waste."
        ],
        "bahaya": False,
    },
    "Player": {
        "nama": "Aksesori (Pemutar Media)",
        "ikon": "🎧",
        "penanganan": [
            "Keluarkan kaset, CD, atau kartu memori dari dalam perangkat.",
            "Lepaskan baterai atau kabel daya pendukung.",
            "Bawa ke tempat pengumpulan sampah elektronik."
        ],
        "bahaya": False,
    },
    "Printer": {
        "nama": "Peralatan Elektronik (Printer)",
        "ikon": "🖨️",
        "penanganan": [
            "Lepaskan cartridge tinta atau toner sebelum dibuang.",
            "Keluarkan sisa kertas dari tray pengumpan.",
            "Hubungi layanan pengolahan limbah elektronik atau gerai daur ulang."
        ],
        "bahaya": False,
    },
    "Television": {
        "nama": "Peralatan Elektronik (Televisi)",
        "ikon": "📺",
        "penanganan": [
            "Hindari meretakkan atau memecahkan panel layar kaca.",
            "Jangan membongkar bagian belakang sendiri demi keamanan listrik.",
            "Serahkan ke dinas lingkungan hidup atau bank e-waste."
        ],
        "bahaya": False,
    },
    "Washing Machine": {
        "nama": "Peralatan Elektronik (Mesin Cuci)",
        "ikon": "🫧",
        "penanganan": [
            "Kuras sisa air dari selang dan drum sepenuhnya.",
            "Lepaskan kabel daya dan selang saluran masuk/keluar.",
            "Gunakan layanan angkut khusus barang elektronik rumah tangga besar."
        ],
        "bahaya": False,
    },
}

# ─── 6 Pilihan Kategori Koreksi (Feedback) Sesuai Spesifikasi ──────────────────
CORRECTION_CATEGORIES = [
    {"key": "Laptop", "nama": "Laptop", "ikon": "💻"},
    {"key": "Ponsel", "nama": "Ponsel", "ikon": "📱"},
    {"key": "Kabel", "nama": "Kabel", "ikon": "🔌"},
    {"key": "Baterai", "nama": "Baterai", "ikon": "🔋"},
    {"key": "Papan Sirkuit", "nama": "Papan Sirkuit", "ikon": "🧩"},
    {"key": "Aksesori", "nama": "Aksesori", "ikon": "🎧"},
]

# ─── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            category    TEXT NOT NULL,
            confidence  REAL NOT NULL,
            top3        TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            classification_id   INTEGER,
            original_category   TEXT NOT NULL,
            corrected_category  TEXT,
            is_correct          INTEGER NOT NULL,
            created_at          TEXT NOT NULL,
            FOREIGN KEY (classification_id) REFERENCES classifications(id)
        )
    """)
    conn.commit()
    conn.close()

def save_classification(filename: str, category: str, confidence: float, top3: list) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO classifications (filename, category, confidence, top3, created_at) VALUES (?,?,?,?,?)",
        (filename, category, confidence, json.dumps(top3), datetime.now().isoformat())
    )
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

def save_feedback(classification_id: int, original: str, corrected: Optional[str], is_correct: bool):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Jika original kosong, ambil otomatis dari tabel classifications
    if not original and classification_id:
        c.execute("SELECT category FROM classifications WHERE id = ?", (classification_id,))
        row = c.fetchone()
        if row:
            original = row[0]

    c.execute(
        "INSERT INTO feedback (classification_id, original_category, corrected_category, is_correct, created_at) VALUES (?,?,?,?,?)",
        (classification_id, original or "", corrected, int(is_correct), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_recent_classifications(limit: int = 5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, filename, category, confidence, created_at FROM classifications ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    return rows

# ─── Model Loading ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_instance = None
class_names = []
img_size = 224

def load_model():
    global model_instance, class_names, img_size
    if not MODEL_PATH.exists():
        logger.warning(f"Model belum ada di {MODEL_PATH}.")
        return False

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        class_names = checkpoint["class_names"]
        num_classes = checkpoint["num_classes"]
        img_size    = checkpoint.get("img_size", 224)

        net = models.mobilenet_v3_small(weights=None)
        in_features = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_features, num_classes)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        net.to(device)

        model_instance = net
        logger.info(f"Model berhasil dimuat dari {MODEL_PATH}. Kelas: {class_names}")
        return True
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        return False

def preprocess_image(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def predict(image_path: Path):
    if model_instance is None:
        # Coba muat ulang jika model baru saja selesai dilatih
        if not load_model():
            raise RuntimeError("Model belum dimuat.")

    tensor = preprocess_image(image_path)
    with torch.no_grad():
        logits = model_instance(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top3_idx   = torch.topk(probs, 3).indices.tolist()
    top3_probs = torch.topk(probs, 3).values.tolist()

    top_class      = class_names[top3_idx[0]]
    top_confidence = top3_probs[0]

    top3 = []
    for i, p in zip(top3_idx, top3_probs):
        cls_key = class_names[i]
        info = CLASS_INFO.get(cls_key, {"nama": cls_key, "ikon": "♻️"})
        top3.append({
            "kategori_en": cls_key,
            "kategori": info["nama"],
            "ikon": info["ikon"],
            "confidence": round(p * 100, 1),
        })

    return top_class, round(top_confidence * 100, 1), top3

def build_message(kategori_nama: str, confidence: float) -> str:
    if confidence >= 70:
        return f"Kami cukup yakin ini adalah {kategori_nama}."
    else:
        return "Kami kurang yakin, coba foto lebih dekat dan terang."

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kenali Sampah Elektronikmu",
    description="Aplikasi web sederhana pengenal jenis e-waste dengan model AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")

# ─── Pydantic Models ──────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    classification_id: int
    is_correct: bool
    corrected_category: Optional[str] = None

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
            return await f.read()
    return HTMLResponse("<h1>File index.html tidak ditemukan</h1>", status_code=500)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Validasi tipe file
    content_type = file.content_type or ""
    filename_lower = (file.filename or "").lower()
    is_valid_type = (
        content_type in ALLOWED_TYPES or
        filename_lower.endswith(".jpg") or
        filename_lower.endswith(".jpeg") or
        filename_lower.endswith(".png")
    )

    if not is_valid_type:
        raise HTTPException(
            status_code=400,
            detail="Format file tidak didukung. Gunakan file JPG atau PNG ya."
        )

    # Baca & validasi ukuran (maks 5 MB)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Ukuran gambar terlalu besar (maksimal 5 MB). Coba kompres dulu ya."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="File gambar kosong. Pilih foto yang valid ya."
        )

    # Simpan file
    ext = Path(file.filename or "image.jpg").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / fname

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    # Validasi keabsahan gambar
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Gambarnya kurang jelas atau rusak, coba foto lagi ya."
        )

    # Prediksi
    try:
        category, confidence, top3 = predict(file_path)
    except Exception as e:
        logger.error(f"Error inferensi: {e}")
        raise HTTPException(
            status_code=500,
            detail="Gambarnya kurang jelas, coba lagi ya."
        )

    # Ambil metadata kategori
    info = CLASS_INFO.get(category, {
        "nama": category,
        "ikon": "♻️",
        "penanganan": ["Bawa ke pusat daur ulang e-waste terdekat."],
        "bahaya": False,
    })

    # Simpan riwayat ke database
    cls_id = save_classification(fname, info["nama"], confidence, top3)

    return JSONResponse({
        "id": cls_id,
        "kategori_en": category,
        "kategori": info["nama"],
        "ikon": info["ikon"],
        "confidence": confidence,
        "top3": top3,
        "penanganan": info["penanganan"],
        "bahaya": info["bahaya"],
        "gambar_url": f"/uploads/{fname}",
        "pesan": build_message(info["nama"], confidence),
    })


@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    try:
        save_feedback(
            classification_id=req.classification_id,
            original="",
            corrected=req.corrected_category,
            is_correct=req.is_correct,
        )
        return {"status": "ok", "pesan": "Terima kasih atas koreksinya!"}
    except Exception as e:
        logger.error(f"Error feedback: {e}")
        return {"status": "error", "pesan": "Gagal menyimpan masukan."}


@app.get("/history")
async def history_endpoint():
    rows = get_recent_classifications(limit=5)
    result = []
    for row in rows:
        cls_id, filename, category, confidence, created_at = row
        # Cari icon yang cocok
        ikon = "♻️"
        for v in CLASS_INFO.values():
            if v["nama"] == category:
                ikon = v["ikon"]
                break
        for c in CORRECTION_CATEGORIES:
            if c["nama"] == category:
                ikon = c["ikon"]
                break

        result.append({
            "id": cls_id,
            "gambar_url": f"/uploads/{filename}",
            "kategori": category,
            "ikon": ikon,
            "confidence": confidence,
            "waktu": created_at,
        })
    return result


@app.get("/categories")
async def categories_endpoint():
    # Mengembalikan 6 kategori pilihan untuk koreksi (feedback)
    return CORRECTION_CATEGORIES


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_instance is not None,
        "device": str(device),
    }
