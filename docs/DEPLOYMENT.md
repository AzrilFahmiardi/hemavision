# Deployment: Model Konjungtiva

## Status Saat Ini

Model konjungtiva sudah dapat dipakai untuk citra baru lewat `app/predict.py` (fungsi `predict()`) dan diuji lewat API lokal `app/main.py` (FastAPI, endpoint `POST /predict`). Keduanya sudah diverifikasi menghasilkan prediksi yang konsisten pada foto mentah nyata. Dokumen ini membandingkan opsi hosting cloud gratis untuk tahap selanjutnya, bila software perangkat lunak final butuh model ini dipanggil dari backend. Belum ada deploy eksternal yang dijalankan, keputusan menjalankannya menunggu konfirmasi terpisah.

## Menjalankan API Lokal

```
uvicorn app.main:app --reload --port 8000
```

Uji cepat:

```
curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@path/ke/foto.jpg" \
  -F "age_years=30" \
  -F "gender=F" \
  -F "site=ghana"
```

## Perbandingan Opsi Hosting Gratis

| Platform | Tier gratis | Kecocokan untuk model ini | Catatan |
|---|---|---|---|
| Hugging Face Spaces | CPU basic gratis permanen (2 vCPU, 16 GB RAM), tidur otomatis setelah idle lalu bangun saat diakses | Direkomendasikan. Model konjungtiva ringan (ResNet18 + fully connected, inference CPU dalam hitungan detik), tidak butuh GPU | Cukup unggah Dockerfile atau `app.py` (Gradio/FastAPI), repo Git-based, cocok untuk demo publik saat presentasi |
| Railway | Kredit percobaan terbatas (habis dalam beberapa hari pemakaian aktif), setelah itu berbayar | Kurang cocok untuk dipakai jangka panjang tanpa kartu kredit, tapi cukup untuk uji coba singkat | Setup mirip Heroku, deploy dari Git langsung |
| GCP Cloud Run | Selalu gratis untuk traffic rendah (2 juta request per bulan, tapi mensyaratkan billing account aktif) | Bisa dipakai, tapi setup lebih rumit (build image, push ke Artifact Registry, konfigurasi IAM) dan mensyaratkan kartu kredit untuk aktivasi billing meski tidak tertagih pada traffic rendah | Cocok bila software perangkat lunak sudah dideploy di GCP juga, untuk konsistensi infrastruktur |
| AWS (Lambda / App Runner) | Free tier 12 bulan pertama, request gratis terbatas | Ukuran model dan dependency (PyTorch, torchvision, OpenCV) mendekati batas ukuran paket Lambda, butuh container image, lebih rumit dibanding opsi lain | Dipertimbangkan hanya bila tim sudah punya akun AWS aktif |

**Rekomendasi**: Hugging Face Spaces untuk demo dan pengujian backend, karena gratis permanen, tanpa kartu kredit, dan setup paling sederhana untuk model PyTorch berukuran sedang seperti ini.

## Langkah Deploy ke Hugging Face Spaces (bila dipilih)

Spaces mendukung Docker custom, jadi struktur `app/` (FastAPI) bisa dipakai langsung tanpa restrukturisasi besar.

1. Buat Space baru dengan SDK "Docker".
2. Sertakan `Dockerfile` yang menyalin `src/`, `app/`, `configs/`, `artifacts/conjunctiva/` (checkpoint model, bukan dataset mentah), dan `requirements.txt`, lalu menjalankan `uvicorn app.main:app --host 0.0.0.0 --port 7860`.
3. Checkpoint model (`artifacts/conjunctiva/*.pt`) disertakan dalam image atau diunggah lewat Git LFS bila melewati batas ukuran repo biasa.
4. Set `PORT=7860` sesuai konvensi Spaces.

Dockerfile dan langkah push belum dibuat, menunggu keputusan eksplisit untuk benar-benar deploy.

## Batasan yang Perlu Diketahui Backend

- Endpoint hanya menerima satu foto per permintaan, tidak ada batching.
- Waktu inference didominasi oleh segmentasi U-Net dan ekstraksi embedding ResNet18 pada CPU, sekitar 1-3 detik per foto pada perangkat tanpa GPU.
- Model menolak citra yang gagal quality control (blur, glare, terlalu gelap atau terang, region of interest terlalu kecil) lewat `passed_qc: false` beserta alasan penolakan, backend perlu menampilkan pesan ini ke pengguna, bukan memaksa prediksi pada citra buruk.
- Prediksi severity (`severity_caveat`) memiliki keandalan rendah (Cohen kappa 0.147) dan harus ditampilkan dengan peringatan eksplisit ke pengguna akhir.
