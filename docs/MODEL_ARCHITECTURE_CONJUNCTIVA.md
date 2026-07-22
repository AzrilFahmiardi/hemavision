# Model Architecture: Conjunctiva Site

Dokumen ini merangkum komposisi teknis model konjungtiva untuk keperluan penulisan metodologi proposal. Model ini adalah kombinasi eksplisit dari beberapa komponen (hybrid), bukan satu model tunggal siap pakai, sehingga setiap komponen perlu disebutkan beserta alasan pemilihannya, mengikuti gaya penulisan metodologi pada empat paper rujukan (Paper 1 QPAIN, Paper A CP-AnemiC, Paper B Eyes-Defy, Paper 4 BPANet).

## Komponen dan Justifikasi

| Komponen | Apa sebenarnya | Kenapa dipilih |
|---|---|---|
| Path B, backbone visual | ResNet18 pretrained ImageNet, dengan modul CSA (Channel-Spatial Attention gaya CBAM) disisipkan setelah blok pertama | ResNet18 dipilih ketimbang ResNet50 (Paper A) atau MobileNetV2 (Paper B) karena ukuran dataset kecil (925 sampel gabungan), sehingga backbone lebih ringan mengurangi risiko overfitting. CSA mengikuti Paper 4 (BPANet) yang membuktikan lewat ablation bahwa atensi ini menaikkan performa dengan menonjolkan pembuluh darah dan tekstur halus, komponen yang tidak dimiliki Paper A maupun Paper B. |
| Path A, fitur warna | Bukan neural network, murni 27 fitur hasil formula matematis: statistik RGB, HSV, CIELAB, rasio HHR, erythema index, entropy, brightness, tekstur gray level lokal | Mengikuti Paper 1 (24 fitur serupa) dan Paper B (14 fitur serupa termasuk HHR dan tekstur). Memberi interpretability tinggi (dapat dijelaskan secara klinis kenapa model memutuskan anemia). Paper 1 membuktikan fitur semacam ini dengan SVM sederhana dapat menyamai performa ResNet50 (akurasi 0.849 berbanding 0.848). |
| Fusion | Modul kecil berisi dua linear layer, disebut Fusion Attention | Terinspirasi Paper 4 (BPANet), dipakai untuk menggabungkan Path A, Path B, demografi (umur, gender), dan site token menjadi satu representasi. |
| Kepala keluaran (bukan model terpisah, hanya layer linear di ujung jaringan) | Tiga head: regresi hemoglobin (Dual Loss), klasifikasi biner anemia (Focal Loss), severity ordinal empat kelas (CORN loss) | Dual Loss mengikuti Paper 4 (BPANet) yang terbukti lewat ablation menaikkan MAE. Focal Loss adalah versi deep learning dari penanganan class imbalance yang ditekankan Paper B (RUSBoost). Head severity ordinal adalah kontribusi baru kami karena CP-AnemiC memiliki label severity namun tidak ada satupun dari empat paper yang memanfaatkannya. |

## Kalimat Ringkasan untuk Proposal

Kami mengusulkan arsitektur dual-path yang menggabungkan backbone ResNet18 dengan modul channel-spatial attention (Path B) dan 27 fitur biomarker warna hand-crafted (Path A), difusikan melalui modul fusion attention bersama variabel demografi dan site token, lalu dilatih secara multi-task dengan tiga head: regresi hemoglobin, klasifikasi biner anemia, dan estimasi severity ordinal.

## Novelty vs Empat Paper Rujukan

Tidak ada satupun dari empat paper yang memfusikan jalur hand-crafted dengan jalur deep embedding (dual-path), memakai label severity ordinal, atau melatih model gabungan lintas dataset dengan site token. Ketiga hal ini menjadi kontribusi orisinal proyek ini.

## Perbandingan Model (Eksperimen Stage 5)

Untuk membuktikan setiap pilihan desain secara empiris (bukan klaim), lima konfigurasi dilatih dengan protokol lima-fold cross-validation berbasis pasien yang identik, agar perbandingan adil:

1. Path A saja (fitur hand-crafted, ditambah demografi dan site token) sebagai baseline hand-crafted.
2. Path B saja dengan backbone ResNet18-CSA.
3. Path B saja dengan backbone MobileNetV3-CSA (opsi lebih ringan, relevan untuk deployment perangkat mobile atau edge).
4. Full Fusion dengan backbone ResNet18-CSA, kandidat model utama.
5. Full Fusion dengan backbone MobileNetV3-CSA.

Sanity check tambahan sebelum kelima konfigurasi di atas: Path A saja dipasangkan dengan SVM klasik (bukan neural network) diharapkan mendekati akurasi 0.849 pada CP-AnemiC (Paper 1), sebagai bukti bahwa implementasi fitur hand-crafted kami sudah benar sebelum dipakai pada arsitektur yang lebih kompleks.

Hasil perbandingan lima konfigurasi ini (MAE hemoglobin, akurasi anemia, akurasi severity) menjadi bukti kuantitatif untuk menjawab pertanyaan seperti kenapa memilih ResNet18 dibanding MobileNetV3, dan kenapa pendekatan dual-path dipilih dibanding jalur tunggal.

## Baseline Target dari Literatur

Target metrik yang perlu didekati atau dilampaui sebagai tolok ukur validitas:
- MAE hemoglobin pada Eyes-Defy sekitar 1.212 g/dL (BPANet).
- MAE hemoglobin pada CP-AnemiC sekitar 1.50 g/dL (ViT, Paper A).
- Akurasi klasifikasi anemia pada CP-AnemiC sekitar 0.849 (SVM tuned, Paper 1).
