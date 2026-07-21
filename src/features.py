"""Stage 4. Dual-path feature extraction.

Menyediakan jalur fitur warna hand-crafted yang interpretable (statistik kanal
RGB, HSV, CIELAB, rasio terkait hemoglobin, erythema index, entropy, dan fitur
tekstur) serta jalur deep embedding berbasis backbone ringan dengan modul
attention. Kedua jalur difusikan bersama variabel demografi dan site token.
Implementasi menyusul pada Stage 4.
"""
from __future__ import annotations
