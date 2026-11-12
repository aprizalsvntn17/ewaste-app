import sys
from pathlib import Path

# Tambahkan root directory ke sys.path agar modul main dapat diimpor di Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app

# Vercel entrypoint
