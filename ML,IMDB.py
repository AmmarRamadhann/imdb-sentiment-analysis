# ============================================================
# PREDIKSI SENTIMEN ULASAN FILM IMDB
# Algoritma: TF-IDF + Logistic Regression
# Dataset: IMDB Movie Reviews
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. MEMBACA DATASET
# ============================================================

folder_program = Path(__file__).resolve().parent

file_train = folder_program / "train.csv"
file_test = folder_program / "test.csv"

if not file_train.exists():
    raise FileNotFoundError(
        "File train.csv tidak ditemukan di folder program."
    )

if not file_test.exists():
    raise FileNotFoundError(
        "File test.csv tidak ditemukan di folder program."
    )

df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)

print("=" * 60)
print("PREDIKSI SENTIMEN ULASAN FILM IMDB")
print("=" * 60)

print("\nUkuran dataset train:", df_train.shape)
print("Ukuran dataset test :", df_test.shape)

print("\nKolom dataset train:")
print(df_train.columns.tolist())

print("\nLima data pertama:")
print(df_train.head())


# ============================================================
# 2. MEMERIKSA KOLOM DATASET
# ============================================================

# Menyamakan nama kolom agar sesuai dengan kode
df_train.columns = df_train.columns.str.strip().str.lower()
df_test.columns = df_test.columns.str.strip().str.lower()

# Dataset kamu menggunakan kolom text dan sentiment
if "text" not in df_train.columns:
    raise ValueError("Kolom 'text' tidak ditemukan pada train.csv.")

if "sentiment" not in df_train.columns:
    raise ValueError("Kolom 'sentiment' tidak ditemukan pada train.csv.")

if "text" not in df_test.columns:
    raise ValueError("Kolom 'text' tidak ditemukan pada test.csv.")



# ============================================================
# 3. PREPROCESSING DATA
# ============================================================

# Menghapus data kosong
df_train = df_train.dropna(
    subset=["text", "sentiment"]
).copy()

df_test = df_test.dropna(
    subset=["text"]
).copy()

# Memastikan kolom ulasan berupa teks
df_train["text"] = df_train["text"].astype(str).str.strip()
df_test["text"] = df_test["text"].astype(str).str.strip()

# Menyeragamkan label sentimen
df_train["sentiment"] = (
    df_train["sentiment"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Menghapus ulasan kosong
df_train = df_train[df_train["text"] != ""].copy()
df_test = df_test[df_test["text"] != ""].copy()

print("\nJumlah data train setelah preprocessing:", len(df_train))
print("Jumlah data test setelah preprocessing:", len(df_test))

print("\nDistribusi sentimen:")
print(df_train["sentiment"].value_counts())



# ============================================================
# 4. VISUALISASI DISTRIBUSI SENTIMEN
# ============================================================

distribusi = df_train["sentiment"].value_counts()

plt.figure(figsize=(7, 5))
distribusi.plot(kind="bar")

plt.title("Distribusi Sentimen Ulasan Film IMDB")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Ulasan")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 5. MEMBAGI DATA TRAIN DAN VALIDASI
# ============================================================

X = df_train["text"]
y = df_train["sentiment"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah data pelatihan :", len(X_train))
print("Jumlah data validasi  :", len(X_valid))


# ============================================================
# 6. MEMBANGUN MODEL TF-IDF + LOGISTIC REGRESSION
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

print("\nMelatih model...")
model.fit(X_train, y_train)

print("Pelatihan model selesai.")


# ============================================================
# 7. EVALUASI MODEL
# ============================================================

y_pred = model.predict(X_valid)

akurasi = accuracy_score(y_valid, y_pred)

print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL")
print("=" * 60)

print(f"Akurasi: {akurasi:.4f}")
print(f"Akurasi dalam persen: {akurasi * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_valid, y_pred, zero_division=0))

print("Confusion Matrix:")
cm = confusion_matrix(y_valid, y_pred, labels=model.classes_)
print(cm)


# ============================================================
# 8. VISUALISASI CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Sentimen IMDB")
plt.tight_layout()
plt.show()


# ============================================================
# 9. MELATIH ULANG MODEL DENGAN SELURUH DATA TRAIN
# ============================================================

# Setelah evaluasi, model dilatih ulang menggunakan semua
# data train agar dapat digunakan untuk memprediksi test.csv.

print("\nMelatih ulang model menggunakan seluruh data train...")

model.fit(X, y)

print("Model akhir selesai dilatih.")


# ============================================================
# 10. PREDIKSI DATA TEST
# ============================================================

prediksi_test = model.predict(df_test["text"])

hasil_prediksi = pd.DataFrame({
    "text": df_test["text"],
    "prediksi_sentimen": prediksi_test
})

# Jika test.csv memiliki label sentiment, bandingkan prediksi
# dengan label aslinya.

if "sentiment" in df_test.columns:
    label_test = (
        df_test["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_label = label_test.isin(model.classes_)

    if valid_label.sum() > 0:
        akurasi_test = accuracy_score(
            label_test[valid_label],
            prediksi_test[valid_label]
        )

        print("\nAkurasi pada test.csv:")
        print(f"{akurasi_test:.4f} ({akurasi_test * 100:.2f}%)")

        print("\nLaporan evaluasi test.csv:")
        print(
            classification_report(
                label_test[valid_label],
                prediksi_test[valid_label],
                zero_division=0
            )
        )

        hasil_prediksi["sentiment_asli"] = label_test.values

    else:
        print(
            "\nLabel pada test.csv tidak cocok dengan "
            "kelas sentimen yang dipelajari model."
        )

        print("Prediksi dibuat, tetapi akurasi test tidak dihitung.")
else:
    print(
        "\nKolom sentiment tidak tersedia di test.csv."
    )
    print(
        "Prediksi dibuat tanpa evaluasi terhadap label asli."
    )


# ============================================================
# 11. MENYIMPAN HASIL PREDIKSI
# ============================================================

file_hasil = folder_program / "hasil_prediksi.csv"

hasil_prediksi.to_csv(
    file_hasil,
    index=False,
    encoding="utf-8-sig"
)

print("\nHasil prediksi disimpan ke:")
print(file_hasil)

print("\nContoh hasil prediksi:")
print(hasil_prediksi.head(10))


# ============================================================
# 12. MENYIMPAN MODEL
# ============================================================

import joblib

file_model = folder_program / "model_sentimen_imdb.pkl"

joblib.dump(model, file_model)

print("\nModel disimpan ke:")
print(file_model)


# ============================================================
# 13. CONTOH PREDIKSI ULASAN BARU
# ============================================================

ulasan_baru = [
    "This movie was amazing and the acting was excellent.",
    "This film was boring and a complete waste of time."
]

prediksi_baru = model.predict(ulasan_baru)

print("\n" + "=" * 60)
print("CONTOH PREDIKSI ULASAN BARU")
print("=" * 60)

for ulasan, sentimen in zip(ulasan_baru, prediksi_baru):
    print("\nUlasan:", ulasan)
    print("Prediksi sentimen:", sentimen)

print("\nProgram selesai.")
