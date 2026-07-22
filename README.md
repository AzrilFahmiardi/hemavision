# Hemavision

Sistem skrining anemia non-invasif berbasis citra konjungtiva. Berfokus pada modalitas konjungtiva memakai dataset CP-AnemiC dan Eyes-Defy. 

## Project Layout

Proyek disusun mengikuti arsitektur multi-situs anatomis, sehingga pengembangan satu situs (konjungtiva, telapak tangan, kuku) terpisah dari situs lain namun tetap berada dalam repo yang sama.

Paket `src/common/` memuat fungsi generik yang dipakai lintas situs, seperti quality control, normalisasi iluminasi, ekstraksi fitur dual-path, serta model dan metrik evaluasi multi-task. Paket `src/sites/<situs>/` memuat kode yang spesifik untuk satu situs, seperti loader dataset dan model segmentasi. Folder `notebooks/<situs>/` memuat notebook driver per stage untuk situs tersebut. Folder `configs/` memuat resolusi path yang bernamespace per situs. Folder `dataset/<situs>/`, `outputs/<situs>/`, dan `artifacts/<situs>/` masing-masing menampung data mentah, hasil (manifest, tabel metrik, gambar), dan checkpoint model milik satu situs, sehingga tidak saling menimpa.

