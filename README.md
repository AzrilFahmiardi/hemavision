# Hemavision

Sistem skrining anemia non-invasif berbasis citra konjungtiva untuk GEMASTIK 2026 divisi PPL. Fase saat ini berfokus pada modalitas konjungtiva memakai dataset CP-AnemiC dan Eyes-Defy. Rencana lengkap pipeline berada pada `docs/RENCANA_PIPELINE_KONJUNGTIVA.md`.

## Project Layout

Paket `src/` memuat fungsi reusable per stage. Folder `notebooks/` memuat notebook driver per stage. Folder `configs/` memuat resolusi path dan pengaturan. Folder `outputs/` menampung manifest, tabel metrik, dan gambar. Folder `artifacts/` menampung checkpoint model. Folder `dataset/` hanya ada di server sebagai target sinkronisasi.

## Conventions

Aturan penulisan kode wajib berada pada `CONVENTIONS.md`. Patuhi sebelum menulis atau mengedit kode apa pun.

## Environment

Buat environment memakai `environment.yml` atau `requirements.txt`. Di server dipakai environment conda bernama `hemavision` dengan torch build cu128 yang telah terverifikasi berjalan pada GPU RTX 5080.

## Workflow

Penulisan kode dilakukan di lokal, lalu disinkronkan ke server memakai rsync, dan pelatihan dijalankan di server GPU. Perintah sinkronisasi dan eksekusi terdokumentasi pada file rencana proyek.
