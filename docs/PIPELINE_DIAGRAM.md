# Pipeline Diagram: Conjunctiva Site

Diagram mermaid berikut merangkum pipeline konjungtiva secara menyeluruh, dari data mentah hingga evaluasi. Siap ditempel ke proposal atau dirender menjadi gambar lewat editor mermaid.

```mermaid
flowchart TD
    subgraph Stage0["Stage 0. Data Harmonization"]
        A1[CP-AnemiC 710 sampel] --> A3[Manifest Terunifikasi]
        A2[Eyes-Defy 215 sampel] --> A3
        A3 --> A4[Ambang Biner per Populasi]
        A4 --> A5[Split K-Fold Berbasis Pasien]
    end

    subgraph Stage1_2["Stage 1-2. Quality Control and Illumination"]
        B1[Foto Konjungtiva] --> B2{Quality Control}
        B2 -->|Ditolak| B2X[Blur / Glare / Eksposur]
        B2 -->|Lolos| B3[CLAHE Normalization]
        B3 --> B4[Valid Pixel Filter]
    end

    subgraph Stage3["Stage 3. Segmentation"]
        C1[Foto Mata Utuh, Eyes-Defy] --> C2[U-Net ResNet34]
        C2 --> C3[Mask Konjungtiva Palpebral]
    end

    subgraph Stage4["Stage 4. Dual-Path Feature Extraction"]
        D1[ROI Ternormalisasi] --> D2["Path A: 27 Fitur Hand-crafted<br/>RGB, HSV, CIELAB, HHR,<br/>Erythema Index, Tekstur"]
        D1 --> D3["Path B: ResNet18-CSA<br/>Embedding 256-D"]
        D4[Demografi: Umur, Gender] --> D5[Fusion Attention]
        D6[Site Token] --> D5
        D2 --> D5
        D3 --> D5
    end

    subgraph Stage5["Stage 5. Multi-Task Training"]
        E1[Fusion Vector] --> E2[Trunk Bersama]
        E2 --> E3[Head Regresi Hb<br/>Dual Loss]
        E2 --> E4[Head Klasifikasi Anemia<br/>Focal Loss]
        E2 --> E5[Head Severity Ordinal<br/>CORN Loss]
        E6[Optuna Tuning] -.-> E2
    end

    subgraph Stage6["Stage 6. Evaluation and Ablation"]
        F1[Bland-Altman, ROC, Youden Threshold]
        F2[Ablation: CSA, Fusion Attention,<br/>Dual Loss, Site Token, Demografi]
        F3[Generalisasi Lintas Dataset]
        F4[Checklist STARD and TRIPOD]
    end

    A5 --> B1
    B4 --> D1
    C3 -.->|"Supervisi mask (Eyes-Defy)"| C1
    B4 --> C1
    D5 --> E1
    E3 --> F1
    E4 --> F1
    E5 --> F1
    E2 --> F2
    E2 --> F3
```

## Ringkasan Alur Utama

Data dari dua dataset publik disatukan dan dibagi berbasis pasien (Stage 0), lalu setiap citra melewati gerbang kualitas dan normalisasi pencahayaan (Stage 1-2). Segmentasi U-Net menjembatani foto mata utuh menuju region of interest (Stage 3). Dua jalur fitur komplementer, hand-crafted dan deep embedding, difusikan bersama demografi dan site token (Stage 4), lalu dilatih multi-task dengan tiga head yang dituning lewat Optuna (Stage 5). Model diuji klinis mendalam, diverifikasi lewat ablation, dan diukur generalisasinya lintas populasi (Stage 6).
