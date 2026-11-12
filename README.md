# UTS Pengantar Pembelajaran Mesin
### Customer Travel Churn Prediction Pipeline

Repository ini berisi implementasi pipeline preprocessing dan persiapan data untuk kasus klasifikasi *Customer Churn* menggunakan dataset Tour & Travels dari Kaggle (`Customertravel.csv`).

---

## 📋 Ringkasan Alur Pipeline

Pipeline di [main.py](main.py) mencakup tahapan-tahapan berikut:

1. **BAB I - Data Collection**  
   Memuat dataset CSV mentah dan memastikan target bertipe klasifikasi biner (0: Tetap/Loyal, 1: Churn).
2. **BAB II - Data Understanding**  
   Pemeriksaan nama kolom, tipe data, statistik deskriptif, dan sebaran nilai unik per kolom.
3. **BAB III - Data Preprocessing**  
   - Pemeriksaan & penanganan missing value.
   - Deteksi & pembersihan baris duplikat.
   - Pengecekan distribusi kelas target.
   - Encoding data kategorikal (Ordinal mapping untuk `AnnualIncomeClass`, Binary mapping untuk Yes/No, dan One-Hot Encoding).
   - Ekspor data bersih ke `output/data_bersih.csv`.
4. **BAB IV - Exploratory Data Analysis (EDA)**  
   - Visualisasi histogram distribusi semua variabel.
   - Correlation heatmap untuk melihat korelasi antar fitur.
   - Boxplot dan identifikasi outlier menggunakan metode Interquartile Range (IQR).
5. **BAB V - Pembagian Data (Data Splitting)**  
   Pemisahan fitur ($X$) dan target ($y$), lalu split menjadi data latih (80%) dan data uji (20%) secara `stratified` untuk menjaga rasio kelas.
6. **BAB VI - Standardisasi Data**  
   Normalisasi fitur menggunakan `StandardScaler` (fit dilakukan hanya pada data training untuk mencegah *data leakage*).
7. **BAB VII - Penanganan Class Imbalance dengan SMOTE**  
   Menyeimbangkan kelas target pada data training menggunakan teknik *Synthetic Minority Over-sampling Technique* (SMOTE).
8. **BAB VIII - Penyimpanan Hasil Pipeline**  
   Menyimpan dataset final yang sudah siap untuk pemodelan:
   - `output/train_siap_analisis.csv`
   - `output/test_siap_analisis.csv`

---

## 📁 Struktur Direktori

```text
utsPengantarMesin/
├── data/
│   └── Customertravel.csv         # Dataset mentah dari Kaggle
├── output/                        # Hasil eksekusi pipeline
│   ├── 4_1_histogram.png          # Grafik distribusi histogram
│   ├── 4_2_correlation.png        # Heatmap matriks korelasi
│   ├── 4_3_boxplot.png            # Boxplot deteksi outlier
│   ├── 7_smote.png                # Grafik komparasi sebelum vs sesudah SMOTE
│   ├── data_bersih.csv            # Data setelah preprocessing & encoding
│   ├── train_siap_analisis.csv    # Data training siap latih (setelah scaling & SMOTE)
│   └── test_siap_analisis.csv     # Data testing siap evaluasi (setelah scaling)
├── main.py                        # Script utama pipeline Machine Learning
├── .gitignore
└── README.md
```

---

## 🚀 Cara Menjalankan

1. **Install dependensi yang diperlukan:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

2. **Jalankan script:**
   ```bash
   python main.py
   ```
