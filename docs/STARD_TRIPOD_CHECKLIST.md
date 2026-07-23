# STARD 2015 and TRIPOD Reporting Checklist: Conjunctiva Model

Dokumen ini memetakan pekerjaan pipeline konjungtiva ke item checklist STARD 2015 (Standards for Reporting Diagnostic Accuracy Studies) dan TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis). Tujuannya adalah kejujuran pelaporan untuk proposal, bukan sekadar mencentang kotak. Item yang belum terpenuhi dicatat sebagai keterbatasan, bukan disembunyikan.

## STARD 2015

**Judul dan abstrak.** Studi ini adalah pengembangan model diagnostik skrining anemia non-invasif berbasis citra konjungtiva, memakai dua dataset publik retrospektif.

**Tujuan dan hipotesis.** Menguji apakah fusi fitur warna hand-crafted dan deep embedding memberi estimasi hemoglobin dan klasifikasi anemia yang lebih baik dibanding jalur tunggal, dituangkan lengkap di `docs/MODEL_ARCHITECTURE_CONJUNCTIVA.md`.

**Desain studi.** Retrospektif, memakai dua dataset publik cross-sectional (CP-AnemiC dan Eyes-Defy), bukan studi prospektif dengan perekrutan langsung.

**Partisipan.** Kriteria kelayakan mengikuti masing-masing dataset sumber. CP-AnemiC: anak 6-59 bulan dari 10 fasilitas kesehatan Ghana. Eyes-Defy: dewasa dari Italia dan India. Karakteristik demografi dirangkum pada Stage 0 (`notebooks/conjunctiva/01_data_harmonization.ipynb`).

**Uji indeks (index test).** Model Hemavision konjungtiva, dijelaskan komponen demi komponen pada `docs/MODEL_ARCHITECTURE_CONJUNCTIVA.md`.

**Standar rujukan (reference standard).** Kadar hemoglobin dari pemeriksaan laboratorium darah, dengan ambang anemia mengikuti kriteria WHO (anak Hb kurang dari 11 g/dL, dewasa pria kurang dari 13 g/dL, wanita kurang dari 12 g/dL), diimplementasikan pada `src/common/manifest.py` dan `src/sites/conjunctiva/data.py`.

**Analisis.** Validasi silang lima-fold berbasis pasien (`src/common/manifest.py::assign_kfold`), pencarian hyperparameter (Optuna, Stage 5), metrik regresi dan klasifikasi lengkap dengan interval melalui simpangan baku antar fold (Stage 5 dan Stage 6).

**Hasil dan estimasi akurasi diagnostik.** Dilaporkan pada Stage 6: AUC-ROC, sensitivitas dan spesifisitas pada threshold default maupun Youden, PPV, NPV, F1, beserta Bland-Altman untuk estimasi Hb.

**Keterbatasan yang dicatat jujur:**
- Bukan studi prospektif; tidak ada kejadian buruk yang perlu dilaporkan karena non-invasif, tetapi juga tidak ada data keamanan penggunaan lapangan.
- Estimasi interval kepercayaan memakai simpangan baku antar fold, bukan interval kepercayaan formal dari uji statistik tersendiri.
- Belum ada validasi pada populasi baru di luar CP-AnemiC dan Eyes-Defy.

## TRIPOD

**Judul dan abstrak.** Mengidentifikasi diri sebagai pengembangan model prediksi multivariabel (regresi Hb, klasifikasi anemia, severity ordinal).

**Sumber data, partisipan, outcome, prediktor.** Sama seperti pada bagian STARD di atas. Prediktor adalah representasi fusi (fitur warna hand-crafted, deep embedding, demografi, site token) yang dijelaskan lengkap di `docs/MODEL_ARCHITECTURE_CONJUNCTIVA.md`.

**Ukuran sampel.** 925 sampel gabungan (710 CP-AnemiC, 215 Eyes-Defy), dilaporkan di Stage 0.

**Penanganan data hilang.** Baris tanpa metadata lengkap (folder Eyes-Defy tanpa foto atau mask, 3 dari 218) dikeluarkan secara eksplisit dan tercatat log-nya di `src/sites/conjunctiva/data.py`. Baris tanpa label severity (semua baris Eyes-Defy) di-mask keluar dari loss dan evaluasi severity, bukan diberi nilai isian.

**Metode statistik.** Model multi-task neural (`src/common/models.py`), validasi silang lima-fold, pencarian hyperparameter Optuna dengan objective sadar keadilan antar dataset (Stage 5).

**Tipe validasi (poin paling penting untuk dicatat jujur).** Ini adalah **validasi internal (TRIPOD tipe 1b/2b)**: seluruh validasi silang dan pencarian hyperparameter dilakukan pada gabungan dua dataset publik yang sama, bukan pada kohort eksternal independen yang benar-benar terpisah. Bagian C Stage 6 (generalisasi lintas dataset, latih pada satu populasi dan uji pada populasi lain) adalah pendekatan paling mendekati validasi eksternal yang tersedia dalam proyek ini, tetapi kedua populasi tetap berasal dari dataset publik yang sama yang dipakai untuk pengembangan, sehingga bukan validasi eksternal penuh dalam pengertian TRIPOD.

**Performa model.** Dilaporkan lengkap pada Stage 5 (perbandingan lima konfigurasi, hasil tuning) dan Stage 6 (evaluasi klinis mendalam, ablation, generalisasi lintas dataset).

**Presentasi model.** Arsitektur dan kode sumber terbuka pada repositori ini (`src/common/models.py`, `src/common/train.py`), bukan disajikan sebagai persamaan koefisien tunggal karena model berbasis neural network.

**Keterbatasan yang dicatat jujur:**
- Tidak ada kohort validasi eksternal yang benar-benar independen dari proses pengembangan.
- Hyperparameter dituning pada dataset yang sama dengan yang dipakai melaporkan performa akhir (meski lewat validasi silang, bukan pada test set yang benar-benar terpisah dari seluruh proses tuning).
- Ukuran sampel tergolong kecil untuk standar model prediksi klinis (925 sampel gabungan).
