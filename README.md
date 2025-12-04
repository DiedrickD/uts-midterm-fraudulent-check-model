# 💳 UTS Midterm: End-to-End Fraud Detection Model

## Daftar Isi
1.  [Pendahuluan](#pendahuluan)
2.  [Set Data dan Imbalance](#set-data-dan-imbalance)
3.  [Metodologi dan Pipeline](#metodologi-dan-pipeline)
4.  [Hasil dan Analisis Evaluasi](#hasil-dan-analisis-evaluasi)
5.  [Cara Menjalankan Proyek](#cara-menjalankan-proyek)

---

## 1. Pendahuluan

Proyek ini menyajikan solusi *end-to-end* untuk masalah **Deteksi Penipuan Transaksi** menggunakan set data yang sangat tidak seimbang. Tujuannya adalah membangun model **Deep Neural Network (DNN)** yang akurat dan seimbang untuk memprediksi probabilitas bahwa suatu transaksi adalah penipuan (`isFraud=1`).

Solusi yang dikembangkan secara eksplisit menangani:
1.  **Integrasi Data** Transaksi dan Identitas.
2.  **Penanganan Ketidakseimbangan Kelas** (Class Imbalance).
3.  **Optimalisasi Metrik** untuk meminimalkan kerugian finansial (Recall) sekaligus mengelola beban kerja peninjauan (Precision).

---

## 2. Set Data dan Imbalance

Data proyek terdiri dari dua set data yang harus digabungkan:

* **Transaksi (`*_transaction.csv`):** Fitur-fitur transaksi, jumlah, kategori, dan waktu (`TransactionDT`).
* **Identitas (`*_identity.csv`):** Informasi perangkat, *browser*, dan jaringan yang terkait dengan transaksi.

**Ketidakseimbangan Kelas:**
Data target memiliki rasio penipuan yang sangat kecil (sekitar $\mathbf{3.5\%}$), menjadikannya masalah yang didominasi oleh kelas minoritas.

---

## 3. Metodologi dan Pipeline

### 3.1. Pra-pemrosesan Data
* **Penggabungan Data:** Data transaksi dan identitas digabungkan menggunakan `TransactionID`.
* **Penanganan Nilai Hilang:**
    * Kolom dengan nilai hilang di atas $70\%$ dihapus.
    * Nilai yang hilang diimputasi menggunakan **median** (numerik) dan **mode** (kategorikal).
* **Rekayasa Fitur:** Fitur berbasis waktu (`day_of_week` dan `hour`) dibuat dari kolom `TransactionDT`.
* **Encoding:** Fitur kategorikal di-*encode* menggunakan One-Hot Encoding atau Label Encoding.
* **Pembagian Data:** Data pelatihan dibagi menjadi $80\%$ Training dan $20\%$ Validation, menggunakan `stratify` untuk menjaga rasio penipuan.

### 3.2. Pemodelan DNN dan Penanganan Imbalance
1.  **Class Weighting:** **Bobot kelas** dihitung berdasarkan rasio ketidakseimbangan dan diterapkan pada fungsi kerugian (`loss function`) model Keras. Ini memastikan model memberikan penalti yang jauh lebih besar untuk kasus penipuan yang terlewat (**False Negatives**), secara fundamental meningkatkan potensi **Recall**.
2.  **Model Training:** Model Deep Neural Network dilatih dengan *callbacks* **Early Stopping** dan *Learning Rate Scheduler* untuk optimasi dan menghindari *overfitting*.

### 3.3. Threshold Tuning
Untuk menyeimbangkan kinerja model, **ambang batas klasifikasi** disesuaikan dari nilai default ($0.5$) ke nilai optimal yang memaksimalkan **F1 Score** (berdasarkan analisis kurva Precision-Recall), menghasilkan kinerja akhir yang seimbang.

---

## 4. Hasil dan Analisis Evaluasi

Model dievaluasi pada set validasi dengan ambang batas yang telah disesuaikan. Hasil ini menunjukkan kinerja yang seimbang, menghindari alarm palsu yang berlebihan sambil tetap menangkap sebagian besar kasus penipuan.

| Metrik | Nilai | Penjelasan |
| :--- | :--- | :--- |
| **ROC AUC** | $\mathbf{0.9344}$ | Kemampuan diskriminatif model yang kuat. |
| **Accuracy** | $0.9691$ | Akurasi keseluruhan. |
| **Precision (Penipuan)** | $\mathbf{0.5470}$ | Sekitar **55%** dari semua alarm adalah penipuan sejati. |
| **Recall (Penipuan)** | $\mathbf{0.6784}$ | **68%** dari seluruh kasus penipuan berhasil ditangkap (mengurangi kerugian finansial). |
| **F1 Score** | $\mathbf{0.6057}$ | Metrik terbaik yang mencerminkan keseimbangan antara Precision dan Recall. |

### Matriks Kebingungan (Confusion Matrix)

| | Predicted Non-Fraud (0) | Predicted Fraud (1) |
| :--- | :--- | :--- |
| **Actual Non-Fraud (0)** | $111,653$ (TN) | $\mathbf{2,322}$ (**FP - False Positives**) |
| **Actual Fraud (1)** | $\mathbf{1,329}$ (**FN - False Negatives**) | $\mathbf{2,804}$ (TP) |

**Implikasi Operasional:**
* Hanya $\mathbf{2,322}$ transaksi sah yang salah diklasifikasikan sebagai penipuan, yang merupakan jumlah yang dapat dikelola untuk peninjauan manual, memastikan **customer experience** yang lebih baik.
* Dengan $\mathbf{1,329}$ kasus penipuan yang terlewat, kerugian finansial diminimalkan.

---

## 5. Cara Menjalankan Proyek

1.  **Kloning Repositori:**
    ```bash
    git clone [https://github.com/uts-midterm/](https://github.com/uts-midterm/)[NAMA_REPOS_ANDA]
    ```
2.  **Instal Dependensi:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow keras matplotlib
    ```
3.  **Siapkan Data:** Pastikan file `train_transaction.csv`, `test_transaction.csv`, `train_identity.csv`, dan `test_identity.csv` berada di lokasi yang dapat diakses oleh notebook (sesuai dengan variabel `BASE_PATH`).
4.  **Jalankan Notebook:** Buka dan jalankan file `diedrick-midterm-fraud-transaction-data-ipynb.ipynb` secara berurutan.
