import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


DATA_PATH = "data/Customertravel.csv"
OUTPUT_DIR = "output"          
TARGET_COL = "Target"          
DROP_DUPLICATES = True         
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 50)


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# BAB I - DATA COLLECTION
judul("BAB I - DATA COLLECTION")

df = pd.read_csv(DATA_PATH)
print("Jumlah data awal (baris, kolom):", df.shape)
print("\n5 baris pertama:")
print(df.head())

# Pastikan kolom target ada. Jika nama berbeda, pakai kolom terakhir.
if TARGET_COL not in df.columns:
    TARGET_COL = df.columns[-1]
    print(f"\n[INFO] Kolom 'Target' tidak ditemukan, memakai kolom terakhir: {TARGET_COL}")

# Validasi syarat soal: harus binary class
n_kelas = df[TARGET_COL].nunique()
print(f"\nJumlah kelas pada target '{TARGET_COL}':", n_kelas)
assert n_kelas == 2, "Target bukan binary class!"


# BAB II - DATA UNDERSTANDING
judul("BAB II - DATA UNDERSTANDING")

print("2.1 Nama kolom:")
for col in df.columns:
    print("-", col)

print("\n2.2 Informasi dataset:")
df.info()

print("\n2.3 Statistik deskriptif:")
print(df.describe(include="all"))

print("\n2.4 Nilai unik setiap kolom:")
for col in df.columns:
    print(f"\nKolom: {col}")
    print("Jumlah nilai unik:", df[col].nunique())
    print("Nilai unik:", df[col].unique()[:20])


# BAB III - DATA PREPROCESSING
judul("BAB III - DATA PREPROCESSING")

# ---- 3.1 Missing value ----
print("3.1 Pemeriksaan missing value")
missing = df.isnull().sum()
missing_table = pd.DataFrame({
    "Jumlah Missing": missing,
    "Persentase (%)": (missing / len(df)) * 100,
})
print(missing_table)

# Tangani missing value: numerik -> median, kategorikal -> modus
if missing.sum() > 0:
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    print("\nTotal missing value setelah imputasi:", df.isnull().sum().sum())
else:
    print("\nTidak ada missing value, tidak perlu imputasi.")

# ---- 3.2 Nilai tidak valid (contoh: nilai <= 0 pada kolom Age) ----
print("\n3.2 Pemeriksaan nilai tidak valid")
if "Age" in df.columns:
    print("Jumlah Age <= 0 :", (df["Age"] <= 0).sum())
    print("Rentang Age     :", df["Age"].min(), "-", df["Age"].max())

# ---- 3.3 Data duplikat ----
print("\n3.3 Pemeriksaan data duplikat")
n_dup = df.duplicated().sum()
print("Jumlah baris duplikat:", n_dup)
if DROP_DUPLICATES and n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplikat dihapus. Ukuran dataset sekarang:", df.shape)
    
# ---- 3.4 Distribusi target ----
print("\n3.4 Distribusi target")
print(df[TARGET_COL].value_counts())
print((df[TARGET_COL].value_counts(normalize=True) * 100).round(2))

# ---- 3.5 Encoding data kategorikal ----
# Model scikit-learn hanya menerima angka, jadi kolom teks harus diubah.
print("\n3.5 Encoding data kategorikal")
fitur_kategorikal = df.drop(columns=[TARGET_COL]).select_dtypes(include=["object", "category"]).columns.tolist()
print("Kolom kategorikal:", fitur_kategorikal)

# a) Kolom ordinal (ada urutan) -> mapping manual
urutan_income = {"Low Income": 0, "Middle Income": 1, "High Income": 2}
if "AnnualIncomeClass" in df.columns:
    df["AnnualIncomeClass"] = df["AnnualIncomeClass"].map(urutan_income)
    fitur_kategorikal.remove("AnnualIncomeClass")

# b) Kolom Yes/No -> 1/0
for col in fitur_kategorikal.copy():
    nilai = set(df[col].dropna().unique())
    if nilai <= {"Yes", "No"}:
        df[col] = df[col].map({"Yes": 1, "No": 0})
        fitur_kategorikal.remove(col)

# c) Sisanya (>2 kategori tanpa urutan, mis. FrequentFlyer) -> One-Hot Encoding
if fitur_kategorikal:
    df = pd.get_dummies(df, columns=fitur_kategorikal, drop_first=False, dtype=int)

print("Kolom setelah encoding:", df.columns.tolist())
print(df.dtypes)

# Target dipastikan bertipe integer 0/1
if not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
df[TARGET_COL] = df[TARGET_COL].astype(int)

# ---- 3.6 Simpan data bersih ----
df.to_csv(os.path.join(OUTPUT_DIR, "data_bersih.csv"), index=False)
print("\nData bersih disimpan ke:", os.path.join(OUTPUT_DIR, "data_bersih.csv"))


# BAB IV - EXPLORATORY DATA ANALYSIS (EDA)
judul("BAB IV - EXPLORATORY DATA ANALYSIS")

# 4.1 Histogram
df.hist(figsize=(15, 12), bins=20)
plt.suptitle("Distribusi Variabel Dataset Tour & Travels Churn", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_1_histogram.png"), dpi=120)
plt.show()

# 4.2 Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Dataset Tour & Travels Churn")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_2_correlation.png"), dpi=120)
plt.show()

# 4.3 Boxplot (outlier tidak langsung dihapus)
plt.figure(figsize=(14, 7))
sns.boxplot(data=df.drop(columns=[TARGET_COL]))
plt.title("Boxplot Variabel Dataset Tour & Travels Churn")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_3_boxplot.png"), dpi=120)
plt.show()

# 4.4 Deteksi outlier dengan IQR (hanya dilaporkan)
print("Deteksi outlier (metode IQR) pada kolom non-biner:")
for col in df.drop(columns=[TARGET_COL]).columns:
    if df[col].nunique() > 5:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        print(f"- {col}: {n_out} outlier")
print("Outlier tidak dihapus karena belum tentu merupakan kesalahan data.")

# 4.5 Temuan tambahan: korelasi fitur terhadap target
print("\nKorelasi fitur terhadap Target:")
print(df.corr()[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False).round(2))

# 4.6 Temuan tambahan: tingkat churn per kategori (data tanpa duplikat)
df_raw = pd.read_csv(DATA_PATH).drop_duplicates()
print("\nTingkat churn per kategori:")
for col in ["FrequentFlyer", "AnnualIncomeClass",
            "AccountSyncedToSocialMedia", "BookedHotelOrNot"]:
    print(df_raw.groupby(col)[TARGET_COL].agg(["count", "mean"]).round(3))
    print()

# BAB V - PEMBAGIAN DATA
judul("BAB V - PEMBAGIAN DATA")

# 5.1 Pemisahan fitur dan target
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]
print("Ukuran X:", X.shape, "| Ukuran y:", y.shape)

# 5.2 Train-test split (80:20, stratify menjaga proporsi kelas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)
print("Data training:", X_train.shape)
print("Data testing :", X_test.shape)
print("\nProporsi kelas data training (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\nProporsi kelas data testing (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))


# BAB VI - STANDARDISASI DATA
judul("BAB VI - STANDARDISASI DATA")

# z = (x - mean) / std. fit HANYA pada data training agar tidak terjadi data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("Rata-rata data training setelah scaling (mendekati 0):")
print(X_train_scaled.mean().round(3))
print("\nStandar deviasi data training setelah scaling (mendekati 1):")
print(X_train_scaled.std().round(3))


# BAB VII - PENANGANAN CLASS IMBALANCE DENGAN SMOTE
judul("BAB VII - PENANGANAN CLASS IMBALANCE DENGAN SMOTE")

# SMOTE hanya diterapkan pada data TRAINING, data testing tidak disentuh
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Distribusi sebelum SMOTE:")
print(y_train.value_counts())
print("\nDistribusi setelah SMOTE:")
print(y_train_smote.value_counts())

# Visualisasi sebelum vs sesudah
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
y_train.value_counts().sort_index().plot(kind="bar", ax=ax[0], color=["#4C72B0", "#DD8452"])
ax[0].set_title("Sebelum SMOTE")
y_train_smote.value_counts().sort_index().plot(kind="bar", ax=ax[1], color=["#4C72B0", "#DD8452"])
ax[1].set_title("Sesudah SMOTE")
for a in ax:
    a.set_xlabel("Kelas (Target)")
    a.set_ylabel("Jumlah")
    a.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_smote.png"), dpi=120)
plt.show()


# BAB VIII - PENYIMPANAN HASIL PIPELINE
judul("BAB VIII - PENYIMPANAN HASIL PIPELINE")

train_final = X_train_smote.copy()
train_final[TARGET_COL] = y_train_smote.values
test_final = X_test_scaled.copy()
test_final[TARGET_COL] = y_test.values

train_final.to_csv(os.path.join(OUTPUT_DIR, "train_siap_analisis.csv"), index=False)
test_final.to_csv(os.path.join(OUTPUT_DIR, "test_siap_analisis.csv"), index=False)
print("Tersimpan di folder:", OUTPUT_DIR)
print("- data_bersih.csv")
print("- train_siap_analisis.csv", train_final.shape)
print("- test_siap_analisis.csv ", test_final.shape)

judul("RINGKASAN ALUR PIPELINE")
print("1. Data Collection     : load CSV Kaggle & validasi binary class")
print("2. Data Understanding  : info, statistik, nilai unik")
print("3. Preprocessing       : missing value, duplikat, encoding")
print("4. EDA                 : histogram, korelasi, boxplot, outlier")
print("5. Split Data          : train 80% / test 20% (stratify)")
print("6. Standardisasi       : StandardScaler (fit di train)")
print("7. Class Imbalance     : SMOTE (hanya data train)")
print("8. Simpan hasil        : data siap untuk pemodelan")