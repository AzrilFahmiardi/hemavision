# Blueprint Pipeline Konjungtiva — Hematrix (GEMASTIK 2026 PPL)

> **Status:** Rencana FINAL & disetujui. Implementasi **ditunda** sampai server tersedia.
> Semua kode akan ditaruh di folder khusus `hematrix/` (self-contained, mudah dipindah).
> Dokumen ini adalah sumber kebenaran untuk melanjutkan nanti.

---

## Context

Fokus fase ini: **modalitas konjungtiva** untuk skrining anemia non-invasif. Tujuan: pipeline yang (a) menggabungkan yang terbaik dari 4 paper, (b) valid & teliti secara metodologis, (c) "proses kompleks" yang memukau juri saat pitching TANPA menjadi teater.

**Temuan inspeksi dataset (fakta nyata, bukan asumsi):**

- **CP-AnemiC** (710 citra, `dataset/CP-AnemiC dataset/`): strip konjungtiva **sudah di-crop & tersegmentasi** (mask di alpha channel RGBA; 60–82% transparan). Label `Anemia_Data_Collection_Sheet.xlsx` → Hb 3.1–15 g/dL, **severity: Non-anemik 286 / Mild 144 / Moderate 232 / Severe 48**, umur (bulan), gender. Anak Ghana 6–59 bln, ambang Hb<11. Ukuran strip median ~228×88 px, mode RGBA.
- **Eyes-Defy** (`dataset/EYES-DEFY-ANEMIA/`): **Italy 124 + India 96** folder. Tiap folder = **foto mata utuh `.jpg` 3984×2988** + **mask RGBA** (`*_palpebral.png`, `*_forniceal.png`, `*_forniceal_palpebral.png`; alpha = mask). Metadata `Italy.xlsx`/`India.xlsx`: `Number, Hgb, Gender, Age` (tahun), Note. Dewasa, ambang M<13/F<12. **Catatan teknis:** xlsx India pakai **koma desimal** (mis. `15,1`) & banyak baris kosong → perlu parser toleran koma + dropna. Join key: folder# = `Number`; CP stem = `IMAGE_ID`.

**Konsekuensi penting:** foto mata utuh HANYA ada di Eyes-Defy → segmentasi U-Net & white-balance-sklera/MediaPipe hanya bisa dilatih/didemokan di sana. Pada CP-AnemiC pakai mask bawaan. **Gray-World + referensi sklera + MediaPipe wajib DIKOREKSI** (tak ada sklera di strip CP-AnemiC).

**Keputusan terkunci:** Multi-task penuh (biner + regresi Hb + severity ordinal) · pakai KEDUA dataset · strategi **Joint + domain-token** · deliverable = blueprint implementasi · demo = animasi transformasi 1 foto tahap-demi-tahap.

**Prinsip validitas (pagar anti-teater):** tiap modul kompleks WAJIB dibuktikan lewat **ablation** (harus menaikkan metrik), atau dipangkas.

---

## Ringkasan posisi 4 paper (ambil / tinggalkan)

| Sumber | Ambil | Tinggalkan |
|---|---|---|
| **Paper 1** (QPAIN'26 — evaluasi KEDUA dataset) | CLAHE-on-V; 24 fitur hand-crafted; **CORAL domain adaptation**; protokol 50-split + standardisasi train-only | ROI "non-black masking" (kita punya mask asli) |
| **Paper A** (CP-AnemiC descriptor) | **Joint klasifikasi+regresi Hb**; multi-backbone benchmark; Bland-Altman | LAB a* hanya karakterisasi |
| **Paper B** (Eyes-Defy descriptor) | **HHR, entropy, brightness, tekstur g1–g5, +demografi**; **RUSBoost utamakan sensitivitas**; **palpebral = ROI terbaik** | Deep learning (gagal, data kecil) |
| **Paper 4** (BPANet) | **CBAM (CSA) attention**; **Fusion Attention + fusi demografi**; **Dual Loss** (MSE+Focal via bin distribusi); **domain/site token** | Penolakan segmentasi (kita justru punya mask) |

**Novelty milik kita** (tak ada di 4 paper): **Dual-Path fusion** (hand-crafted + deep), **severity ordinal**, **joint lintas-dataset dengan site-token**.

Baseline yang harus direplikasi/ditembus: CP-AnemiC SVM tuned **Acc ~0.849** (Paper 1) & ViT MAE **1.50** (Paper A); Eyes-Defy BPANet MAE **1.212**; cross-dataset naif hanya **~0.56** (Paper 1 — target kita perbaiki).

---

## Arsitektur Pipeline (7 stage)

### Stage 0 — Ingest & Standardization Layer  `src/data/`
- **ROI seragam → RGBA cutout konjungtiva**: Eyes-Defy → mask `*_palpebral.png` (palpebral terbaik per Paper B) ke `.jpg` utuh; CP-AnemiC → sudah cutout. Ukuran tetap (pad-to-square → 224×224, aspect-preserving).
- **Harmonisasi metadata** → skema: `hb_gdl`, `age_years` (CP: bulan/12), `gender`, `site∈{ghana,italy,india}`, `severity` (CP saja), `roi_precropped`.
- **Ambang biner per populasi** (bukan global): anak <11; dewasa M<13/F<12. Biner diturunkan dari Hb + ambang sesuai age/gender.
- **Split by-patient**, stratifikasi per site+kelas.
- *(Skema manifest & parser koma sudah dirancang; tinggal ditulis ulang saat lanjut.)*

### Stage 1 — Quality Control gate  `src/qc/`
Tolak citra buruk (anti garbage-in): **variance-of-Laplacian** (blur), histogram exposure, fraksi area ROI, deteksi glare → pass/reject + alasan.

### Stage 2 — Illumination Normalization  `src/preprocess/`
- Baseline kedua dataset: **CLAHE pada channel V** (HSV) — terbukti Paper 1.
- Ablation/mode-mentah: Gray-World + referensi-sklera **hanya saat foto mata utuh** (Eyes-Defy/app). Jujur: tak berlaku strip CP-AnemiC.
- Filter pixel valid (<20 / >240, Paper B) sebelum fitur hand-crafted.

### Stage 3 — Segmentation front-end (disupervisi Eyes-Defy)  `src/segmentation/`
- **U-Net ringan**: Eyes-Defy `.jpg` utuh → mask palpebral. Loss Dice+BCE.
- Peran: app capture mentah; bintang visual demo; jembatan raw→ROI (tak dipakai di CP-AnemiC).
- Opsional MediaPipe FaceMesh pra-lokalisasi mata (mode app).

### Stage 4 — Dual-Path Feature Extraction (NOVELTY)  `src/features/`
- **Path A (hand-crafted, interpretable)** pada ROI valid: μ/σ RGB·HSV·CIELAB; `mean(R−G)`, `HHR=μR/(μG+ε)`, `HHR_hue=n(H>0.95)/N`, `R/(R+G+B)`; **Erythema Index** `EI∝log10(1/R_green)`; tekstur `g1..g5` (3×3) + entropy + brightness → ~30-D, standardisasi train-only.
- **Path B (deep)**: **MobileNetV3** (edge) / ResNet50 + **CSA/CBAM** (disisipkan awal, fokus pembuluh darah).
- **Fusion Attention** (BPANet) gabung [Path A ⊕ Path B ⊕ demografi(age,gender) ⊕ **site-token one-hot**].

### Stage 5 — Multi-Task Heads  `src/models/`
- **Regresi Hb** (common ground): **Dual Loss** = MSE(expected) + α·Focal(bin distribusi), [0,24] g/dL, ~48 bin. Alternatif Huber.
- **Biner**: head sigmoid + **Focal Loss**, cross-check biner-dari-Hb via ambang populasi.
- **Severity ordinal** (CP saja): **CORN/ordinal loss**, **di-mask** saat label absen (Eyes-Defy) → multi-task tetap jalan.
- Bobot loss dicari via **Optuna (TPE)**.

### Domain handling (inti "Joint + domain-token")
- Utama: **site-token conditioning** di fusion.
- Ablation: **CORAL** / **DANN** (gradient-reversal).
- **Lapor jujur:** matriks generalisasi lintas-dataset (train Ghana→test Italia/India dst), with/without token & adaptasi.

---

## Protokol Training  `src/train/`
1. Fase 1: latih U-Net segmentasi (Eyes-Defy).
2. Fase 2: latih multi-task joint pada kedua dataset (ROI terstandardisasi).
- k-fold **group-by-patient**; standardisasi & class-weight train-only; AdamW lr 1e-4; augmentasi spasial ringan (rotasi ±15°, flip), **hindari augmentasi ubah-warna** (BPANet: rusak sinyal Hb).
- Utamakan **sensitivitas** (skrining) — threshold via Youden/target sens≥85%.

## Evaluasi  `src/eval/` — isi celah paper lain
- Regresi: MAE, **RMSE, R²**, **Bland-Altman** (bias+LoA).
- Klasifikasi: AUC-ROC, **Sensitivitas (prioritas)**, Spesifisitas, F1, PPV/NPV.
- Severity: akurasi, **Cohen's κ**, confusion matrix.
- Lintas-dataset: tabel generalisasi (with/without token & CORAL).
- Pelaporan: **STARD 2015 + TRIPOD**.

## Ablation (PAGAR VALIDITAS)  `experiments/`
Tiap modul on/off → Δmetrik: QC, CLAHE, segmentasi, Path-A/Path-B/fusi, site-token, demografi, Dual-Loss vs MSE. Bukti tiap kompleksitas "membayar sewa".

---

## Pemetaan ke Video Pitching (hero animation)
Satu foto mata → transformasi tahap-demi-tahap, tiap stage = satu beat:
1. Foto mentah → **QC scan** (contoh reject blur).
2. **MediaPipe → U-Net**: overlay mask konjungtiva menyala (paling memukau).
3. **Normalisasi iluminasi**: split before/after CLAHE.
4. **Dual-path split-screen**: kiri biomarker interpretable (R−G, a*, EI, HHR) + **heatmap kemerahan**; kanan **attention heatmap** — "mengapa".
5. **Fusion**: dua jalur + demografi + bendera situs menyatu.
6. **Output**: **Hb gauge** + severity + **confidence/interval**.
7. **Montage robustness Ghana↔Italia↔India** — peta dunia.
→ Menggabungkan 4 sorotan: segmentasi live, XAI, robustness, kalibrasi.

---

## Struktur repo (folder khusus, self-contained)
```
hematrix/
  configs/paths.py          # resolusi path dataset (env HEMATRIX_DATASET / ../dataset)
  src/{data,qc,preprocess,segmentation,features,models,train,eval}/
  experiments/  notebooks/  outputs/  artifacts/
```
Dataset mentah tetap di root `dataset/`; `hematrix/` mengaksesnya via config path relatif/env agar tetap jalan setelah dipindah.

## Urutan eksekusi (saat server siap)
1. Stage 0 loader + harmonisasi metadata + EDA (verifikasi distribusi Hb/severity/site).
2. Stage 1–2 QC + CLAHE (fungsi murni, mudah diuji).
3. Stage 3 U-Net segmentasi (Eyes-Defy) → checkpoint + contoh overlay.
4. Stage 4 dual-path (mulai Path A → baseline SVM/RF cepat, replikasi Acc~0.85).
5. Stage 5 multi-task + site-token; latih joint.
6. Evaluasi + ablation + tabel lintas-dataset.
7. Ekspor TFLite + rakit aset demo.

## Verifikasi end-to-end
- Sanity: Path-A + SVM tuned CP-AnemiC ≈ **Acc 0.849** → fitur benar.
- Segmentasi: IoU/Dice U-Net holdout Eyes-Defy + overlay.
- Multi-task: MAE Hb ≈/menembus baseline; Bland-Altman bias≈0.
- Generalisasi: matriks lintas-dataset; site-token/CORAL menaikkan balanced-acc di atas ~0.56.
- Ablation: tiap modul Δmetrik positif; jika tidak → pangkas.

## Koreksi terhadap rancangan awal (fase konjungtiva)
- ❌ Buang: white-balance referensi-sklera & MediaPipe sebagai wajib di CP-AnemiC; ITA melanin (relevan kulit/kuku, bukan mukosa); DCGAN; situs kuku/bibir; edge-quant → sisihkan ke fase lanjut.
- ✅ Pertahankan & perkuat: Dual-Path fusion, multi-task, U-Net (disupervisi Eyes-Defy), Focal/Huber/Ordinal, domain adaptation, STARD/TRIPOD + Bland-Altman.

---

## Environment (tercatat saat sesi rancang)
Python 3.13.5 · tersedia: numpy 1.26, pandas 2.3, opencv 4.11, scikit-learn 1.8, torch 2.11 (cu130), Pillow 11.3, matplotlib 3.10, openpyxl 3.1.
**Perlu diinstal saat lanjut:** torchvision, scikit-image, albumentations, mediapipe, optuna.
