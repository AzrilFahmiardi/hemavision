"""Stage 2. Illumination normalization.

Menstabilkan variasi pencahayaan memakai CLAHE pada channel V ruang warna HSV,
dengan opsi koreksi tambahan saat foto mata utuh tersedia, dan penyaringan
pixel valid sebelum ekstraksi fitur. Implementasi menyusul pada Stage 2.
"""
from __future__ import annotations
