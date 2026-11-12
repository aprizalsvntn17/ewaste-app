# ♻️ Kenali Sampah Elektronikmu (E-Waste Classifier)

Aplikasi web **SIMPEL**, responsif (*mobile-first*), dan mudah digunakan untuk mengenali jenis sampah elektronik (e-waste) dari foto. Dilengkapi dengan panduan penanganan yang aman serta mekanisme masukan/koreksi pengguna.

---

## 🎨 Desain & Antarmuka
- **Tema Warna**: HANYA putih dan biru (utama `#2563EB`, aksen biru muda `#DBEAFE`/`#EFF6FF`). Teks abu gelap/hitam.
- **Peringatan Khusus Baterai**: Menggunakan aksen merah lembut (`#FEF2F2`, border `#FECACA`, teks `#DC2626`) untuk keselamatan.
- **Tipografi**: Font Inter yang bersih, tebal, dan jelas.
- **Mobile-First**: Tombol besar, ruang sentuh lega, sudut membulat (*rounded corners*), dan bayangan tipis (*soft shadows*).
- **Ikon Kategori**: Laptop (💻), Ponsel (📱), Kabel (🔌), Baterai (🔋), Papan Sirkuit (🧩), Aksesori (🎧), dll.

---

## 🛠️ Stack Teknologi

| Komponen | Teknologi |
|---|---|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **Model AI** | [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/) (`MobileNetV3-Small`) |
| **Database** | SQLite3 (`ewaste.db` — tabel `classifications` & `feedback`) |
| **Frontend** | HTML5, CSS3 murni, dan Vanilla JavaScript (tanpa framework berat) |

---

## 🧠 Performa Model

Model dilatih menggunakan dataset yang ada pada folder `modified-dataset/` (10 kelas, 2.400 gambar latih):
- **Arsitektur**: MobileNetV3-Small (Transfer Learning + Fine-tuning)
- **Akurasi Validasi**: **96.7%**
- **Akurasi Pengujian (Test Accuracy)**: **92.33%**
- **Waktu Inferensi Rata-rata**: ~15–30 ms per gambar

---

## 🚀 Cara Menjalankan Lokal

### 1. Masuk ke Folder Proyek

```bash
cd "d:\PROJEK GREEN\ewaste-app"
```

### 2. Pasang Dependensi

```bash
pip install -r requirements.txt
```

### 3. Melatih Model (Opsional — Model Sudah Siap Pakai di `model/ewaste_model.pth`)

Jika ingin melatih ulang dari dataset:
```bash
python train_model.py
```

### 4. Menjalankan Server Aplikasi

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Buka peramban (browser) di:
```
http://127.0.0.1:8000
```
atau
```
http://localhost:8000
```

---

## 📱 Alur Penggunaan Satu Halaman

1. **Judul & Panduan**: *"Kenali Sampah Elektronikmu"* + *"Foto barangnya, kami bantu tentukan jenisnya."*
2. **Area Unggah Putus-Putus**: *"Ketuk untuk foto atau pilih gambar"* — mendukung kamera langsung HP, pemilihan file galeri, maupun *drag-and-drop*.
3. **Pratinjau & Prediksi**: Foto yang dipilih langsung tampil dengan opsi *"Ganti Foto"* dan tombol biru besar *"Kenali Sekarang"*. Dilengkapi indikator putaran *loading*.
4. **Kartu Hasil Klasifikasi**:
   - Ikon dan nama kategori huruf besar (contoh: **BATERAI**).
   - Bar tingkat keyakinan biru dan persentase (contoh: 96%).
   - Teks ramah: *"Kami cukup yakin ini adalah Baterai."* (Jika di bawah 70%: *"Kami kurang yakin, coba foto lebih dekat dan terang."*).
   - Kotak *"Cara menangani"*: 2–3 poin ringkas. Untuk baterai otomatis tampil dalam kotak merah lembut dengan peringatan bahaya: jangan ditekan, dibakar, atau dibasahi.
   - Pertanyaan *"Hasilnya sudah benar?"* dengan tombol *"✓ Ya, benar"* dan *"✕ Bukan, koreksi"*.
   - Jika memilih koreksi, muncul 6 pilihan kategori: **Laptop (💻)**, **Ponsel (📱)**, **Kabel (🔌)**, **Baterai (🔋)**, **Papan Sirkuit (🧩)**, dan **Aksesori (🎧)** yang tersimpan ke tabel SQLite `feedback`.
5. **Tombol "Foto Lagi"**: Mengulang proses foto dari awal secara instan.
6. **Riwayat Terakhir**: Menampilkan 5 klasifikasi terakhir secara otomatis di bagian bawah halaman.

---

## 🔌 API Endpoints

- `GET /` — Halaman utama frontend.
- `POST /predict` — Unggah file gambar (`file=@gambar.jpg`, validasi JPG/PNG, maks 5 MB). Mengembalikan JSON kategori, confidence, top 3, penanganan, dan pesan ramah.
- `POST /feedback` — Menyimpan umpan balik pengguna (`classification_id`, `is_correct`, `corrected_category`).
- `GET /history` — Mengambil 5 riwayat klasifikasi terakhir.
- `GET /categories` — Mengambil daftar 6 kategori pilihan untuk koreksi.
- `GET /health` — Memeriksa status kesiapan server dan model AI.

---

## 📁 Struktur Direktori

```
d:\PROJEK GREEN\ewaste-app\
├── main.py                     # Backend FastAPI & logika inferensi
├── train_model.py              # Skrip pelatihan PyTorch MobileNetV3-Small
├── requirements.txt            # Dependensi Python
├── README.md                   # Dokumentasi lengkap
├── ewaste.db                   # Database SQLite
├── model\
│   ├── ewaste_model.pth        # Bobot model terlatih (Test Acc: 92.33%)
│   └── model_meta.json         # Metadata akurasi & label
├── static\
│   └── index.html              # Antarmuka web satu halaman (HTML, CSS, JS)
├── uploads\                    # Penyimpanan foto hasil unggahan
└── screenshots\                # Tangkapan layar tampilan aplikasi
    ├── desktop_home.png        # Tampilan desktop awal
    ├── desktop_battery.png     # Tampilan desktop hasil klasifikasi baterai
    ├── mobile_home.png         # Tampilan HP (mobile-first) awal
    ├── mobile_battery.png      # Tampilan HP hasil klasifikasi baterai
    └── mobile_correction.png   # Tampilan HP panel koreksi 6 kategori
```
