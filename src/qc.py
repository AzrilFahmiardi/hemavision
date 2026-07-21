"""Stage 1. Quality control gate.

Menolak citra berkualitas buruk sebelum masuk pipeline, memakai deteksi blur
(variance of Laplacian), pemeriksaan eksposur, fraksi area region of interest,
dan deteksi glare. Implementasi menyusul pada Stage 1.
"""
from __future__ import annotations
