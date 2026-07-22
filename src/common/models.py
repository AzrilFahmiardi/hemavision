"""Multi-task model and heads.

Mendefinisikan arsitektur multi-task dengan head regresi hemoglobin, head
klasifikasi biner, dan head severity ordinal, beserta modul fusion attention
dan mekanisme site token untuk pelatihan gabungan lintas dataset. Implementasi
menyusul saat model multi-task dikerjakan.
"""
from __future__ import annotations
