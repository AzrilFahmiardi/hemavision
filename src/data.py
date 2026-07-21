"""Stage 0. Ingest dan standardisasi dua dataset (CP-AnemiC dan Eyes-Defy)
menjadi satu manifest terunifikasi.

Modul ini bertanggung jawab membaca metadata kedua dataset, menyamakan skema
kolom, menurunkan label biner anemia memakai ambang sesuai populasi, serta
menyiapkan pemisahan data berbasis pasien. Implementasi menyusul pada Stage 0.
"""
from __future__ import annotations
