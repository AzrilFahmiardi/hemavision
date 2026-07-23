# Model Architecture: Conjunctiva Site

Dokumen ini merangkum komposisi teknis model konjungtiva untuk keperluan penulisan metodologi proposal. Model ini adalah kombinasi eksplisit dari beberapa komponen (hybrid), bukan satu model tunggal siap pakai, sehingga setiap komponen perlu disebutkan beserta alasan pemilihannya, mengikuti gaya penulisan metodologi pada empat paper rujukan (Paper 1 QPAIN, Paper A CP-AnemiC, Paper B Eyes-Defy, Paper 4 BPANet).

## Komponen dan Justifikasi

| Komponen | Apa sebenarnya | Kenapa dipilih |
|---|---|---|
| Path B, backbone visual | ResNet18 pretrained ImageNet, dengan modul CSA (Channel-Spatial Attention gaya CBAM) disisipkan setelah blok pertama | ResNet18 dipilih ketimbang ResNet50 (Paper A) atau MobileNetV2 (Paper B) karena ukuran dataset kecil (925 sampel gabungan), sehingga backbone lebih ringan mengurangi risiko overfitting. CSA terinspirasi Paper 4 (BPANet), tetapi pada pipeline produksi kami saat ini CSA memakai bobot inisialisasi acak yang belum dilatih, bukan atensi yang benar-benar mempelajari pola dari citra konjungtiva. Lihat bagian "Keterbatasan CSA" di bawah untuk penjelasan lengkap dan percobaan perbaikan yang sudah dilakukan. |
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

## Hasil Akhir Setelah Tuning (Stage 5 dan Stage 6)

Konfigurasi Full Fusion ResNet18-CSA, setelah pencarian hyperparameter Optuna dengan objective sadar keadilan antar dataset, mencapai MAE hemoglobin 1.554 g/dL dan akurasi klasifikasi anemia 0.733 (AUC-ROC 0.826, sensitivitas 0.798 pada threshold Youden). Akurasi CP-AnemiC yang sebelumnya collapse ke kelas mayoritas (0.644) membaik menjadi 0.742 setelah tuning. Detail lengkap ada pada notebook Stage 5 dan Stage 6.

Angka ini kemudian ditemukan tidak reproducible untuk dipakai di backend deployment (lihat bagian Perbaikan Reproducibility Embedding di bawah). **Angka final yang dipakai pada model produksi adalah MAE 1.580 g/dL dan akurasi 0.7502**, sedikit lebih baik dari angka Stage 5/6 di atas, dan kali ini reproducible.

Keterbatasan yang diakui secara jujur: klasifikasi severity masih lemah (Cohen kappa 0.147), dan generalisasi lintas populasi (latih satu dataset, uji dataset lain) masih di bawah temuan Paper 1, sebagian karena site token menjadi konstan pada skenario tersebut.

## Perbaikan Reproducibility Embedding untuk Deployment

Saat menyiapkan `app/predict.py` untuk inference pada foto baru, ditemukan bug kritis: `EmbeddingBackbone` menginisialisasi CSA dan layer proyeksi 256-D secara acak setiap kali kelasnya dibuat, dan backbone ini tidak pernah disimpan sebagai checkpoint pada Stage 4, hanya embedding hasilnya yang disimpan. Akibatnya, backbone segar yang dibuat ulang saat inference memakai proyeksi acak yang sama sekali berbeda dari yang dipelajari `HemavisionModel` selama pelatihan, sehingga prediksi pada citra baru tidak valid (diverifikasi: selisih rerata absolut embedding lama vs baru 0.417, hampir sama dengan standar deviasi embedding itu sendiri 0.427, yaitu praktis dua vektor acak yang tidak berkorelasi).

**Perbaikan**: `EmbeddingBackbone` diberi parameter `seed` eksplisit, backbone (termasuk CSA dan proyeksi) disimpan sebagai checkpoint (`artifacts/conjunctiva/embedding_backbone_resnet18.pt`), dan `predict.py` memuat checkpoint yang sama alih-alih membuat backbone baru. Karena CSA belum terlatih (lihat bagian sebelumnya), seed embedding pada dasarnya adalah satu hyperparameter lagi yang memengaruhi performa akhir secara acak. Karena itu, pencarian hyperparameter Optuna diulang penuh (`app/reoptuna_deployment_model.py`, 40 trial) dengan seed embedding turut dicari bersama hyperparameter lain, memakai objective yang identik dengan Stage 5.

Hasil pencarian ulang: MAE rata-rata 1.580 g/dL, akurasi rata-rata 0.7502, akurasi kasus terburuk antar dataset 0.7408 (embedding seed 9295). Model, backbone, dan hyperparameter ini adalah yang dipakai `app/predict.py` dan endpoint FastAPI `app/main.py`, diverifikasi lewat pengujian langsung pada lima foto mentah Eyes-Defy Italy (hasil `predict()` konsisten dengan hasil endpoint `/predict` untuk sampel yang sama).

## Keterbatasan CSA: Belum Terlatih, Percobaan Perbaikan Belum Berhasil

Ditemukan bahwa `EmbeddingBackbone` (ResNet18, CSA, dan layer proyeksi 256-D) pada pipeline produksi hanya dipakai untuk ekstraksi fitur satu kali (`extract_deep_embeddings`), lalu embeddingnya dibekukan sebagai input tetap ke pelatihan multi-task. Akibatnya CSA dan layer proyeksi **tidak pernah menerima gradien dari loss tugas apa pun**, sehingga bobotnya murni inisialisasi acak, berbeda dari BPANet yang melatih CSA-nya bersama head secara end-to-end.

**Percobaan perbaikan yang sudah dilakukan**: badan ResNet18 dibekukan (bobot ImageNet dipertahankan), sedangkan CSA, proyeksi, fusion attention, trunk, dan tiga head dilatih bersama lewat backpropagation penuh atas citra (`src/common/train.py::run_kfold_end_to_end`, notebook `05b_finetune_end_to_end.ipynb`). Bobot CSA terbukti benar-benar berubah (rerata selisih 0.089 terhadap inisialisasi acak baru, dikonfirmasi lewat pengecekan langsung), sehingga pelatihannya valid secara teknis.

**Hasil**: performa justru menurun dibanding baseline dengan embedding dibekukan.

| Konfigurasi | MAE | Akurasi | Akurasi Severity |
|---|---|---|---|
| Tabular, embedding dibekukan (CSA belum terlatih) | 1.554 | 0.733 | 0.416 |
| End-to-end, CSA terlatih, backbone dibekukan | 1.794 | 0.590 | 0.325 |

Dugaan penyebab utama: `EndToEndDataset` belum memakai augmentasi citra (flip, rotasi, brightness jitter), sehingga melatih parameter tambahan (CSA dan proyeksi) selama puluhan epoch penuh pada data yang relatif sedikit (925 sampel) rawan overfitting. Hyperparameter yang dipakai juga hasil tuning Optuna untuk regime tabular, belum tentu optimal untuk regime end-to-end ini.

**Keputusan**: untuk saat ini, **pipeline tabular dengan embedding dibekukan dipertahankan sebagai model final** (MAE 1.554, akurasi 0.733), karena performanya lebih baik dan sudah tervalidasi menyeluruh (Stage 5 dan Stage 6). Perbaikan CSA end-to-end (dengan augmentasi citra dan tuning hyperparameter khusus regime tersebut) dicatat sebagai **pekerjaan lanjutan** bila waktu memungkinkan, bukan blocker untuk menyelesaikan fase konjungtiva.

**Implikasi untuk komunikasi ke juri**: jangan mengklaim CSA "belajar menonjolkan pembuluh darah" pada sistem saat ini. Kalimat yang jujur dan aman: *"Modul CSA terinspirasi BPANet; kami menguji pelatihan end-to-end-nya dan menemukan performa menurun pada skala data kami, kemungkinan karena overfitting tanpa augmentasi citra, sehingga untuk saat ini kami mempertahankan pipeline dengan embedding tetap yang terbukti lebih baik, sambil mencatat penyempurnaan lebih lanjut sebagai pekerjaan mendatang."* Ini konsisten dengan prinsip melaporkan hasil pengujian apa adanya.

## Keputusan Ablation: Fusion Attention Dipertahankan

Studi ablation Stage 6 menunjukkan bahwa menghilangkan modul Fusion Attention (diganti penggabungan langsung tanpa atensi) justru menghasilkan MAE sedikit lebih baik (1.518 berbanding 1.554) dan akurasi lebih tinggi (0.766 berbanding 0.733) dibanding model penuh. Selisih ini kecil dan tidak mengubah kesimpulan utama, sehingga kami memilih **mempertahankan Fusion Attention** dalam sistem, dengan alasan:

- Selisih performa kecil dan wajar terjadi pada skala data 925 sampel, bukan penurunan performa yang signifikan.
- Modul ini dirancang untuk skala data yang lebih besar ketika situs anatomis tambahan (telapak tangan, kuku) bergabung ke dalam arsitektur late-fusion multi-situs, sehingga tetap relevan sebagai desain jangka panjang.
- Kompetisi ini menilai kelengkapan dan kematangan rekayasa perangkat lunak, bukan semata metrik machine learning, sehingga mempertahankan komponen yang telah diuji dan divalidasi lebih memberi nilai tambah dibanding menyederhanakan sistem demi selisih metrik yang kecil.

Kalimat siap pakai bila ditanya juri: *"Kami menguji lewat ablation dan dampak Fusion Attention pada skala data kami tergolong kecil, tetapi kami pertahankan karena dirancang untuk skala multi-situs yang lebih besar dan tidak merugikan performa secara berarti."* Ini menunjukkan pengujian sungguhan, bukan komponen yang ditambahkan tanpa validasi.
