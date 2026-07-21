"""Training utilities.

Menyediakan rutinitas pelatihan dua fase, yakni pelatihan U-Net segmentasi lalu
pelatihan model multi-task gabungan, dengan validasi silang berbasis pasien,
pembobotan kelas, dan pemilihan threshold operasi yang mengutamakan
sensitivitas. Implementasi menyusul saat stage terkait dikerjakan.
"""
from __future__ import annotations
