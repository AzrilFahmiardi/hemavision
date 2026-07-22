# Code Conventions

Dokumen ini adalah acuan wajib penulisan kode untuk proyek Hemavision. Berlaku untuk notebook (.ipynb) maupun modul Python (.py).

## Commenting and Documentation

Dilarang menggunakan tanda `#` untuk komentar di dalam sel kode. Penjelasan ditulis memakai docstring Python (`"""penjelasan"""`) atau di sel Markdown. Jangan menulis komentar yang membandingkan dengan versi sebelumnya atau menjelaskan perubahan historis. Penjelasan hanya mendeskripsikan logika saat ini.

## Naming and Refactoring

Gunakan nama yang stabil, profesional, dan konsisten untuk variabel, kelas, serta instance. Jangan mengganti nama hanya karena metode membaik. Sebagai contoh, tetap gunakan `model`, bukan `better_model`, `new_model`, atau `improved_model`. Hindari nama yang terlalu kasual.

## Formatting and Tone

Dilarang memakai emoji pada output, pernyataan print, maupun sel Markdown. Dilarang memakai em dash atau double dash pada teks apa pun. Gunakan koma, tanda kurung, atau titik dua sebagai gantinya.

## Notebook Structure and Hierarchy

Susun notebook dengan header Markdown yang jelas, memakai satu tanda pagar untuk seksi utama dan dua tanda pagar untuk subseksi, mencerminkan fitur yang dibangun. Dilarang memakai daftar bernomor untuk menyusun seksi atau penjelasan. Judul dan sub-judul ditulis dalam Bahasa Inggris, sedangkan penjelasan detail di sel Markdown ditulis dalam Bahasa Indonesia.

## Deliverable Layout

Setiap stage pipeline ditulis sebagai satu notebook driver di folder `notebooks/<situs>/`. Fungsi generik yang dipakai lintas situs diletakkan pada paket `src/common/`, sedangkan fungsi yang spesifik untuk satu situs anatomis diletakkan pada paket `src/sites/<situs>/`. Notebook mengimpor keduanya sesuai kebutuhan.
