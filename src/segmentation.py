"""Stage 3. Segmentation front-end.

Melatih U-Net ringan pada foto mata utuh Eyes-Defy untuk memprediksi mask
konjungtiva palpebral. Dipakai pada mode capture foto mentah dan sebagai
jembatan dari citra mentah menuju region of interest. Implementasi menyusul
pada Stage 3.
"""
from __future__ import annotations
