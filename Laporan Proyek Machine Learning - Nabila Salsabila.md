# Laporan Proyek Machine Learning - Nabila Salsabila

## Domain Proyek

Pada industri telekomunikasi, customer churn adalah tantangan besar yang dapat berdampak signifikan pada pendapatan perusahaan. Churn terjadi ketika pelanggan memutuskan berhenti menggunakan layanan perusahaan, yang memengaruhi stabilitas pendapatan dan meningkatkan biaya akuisisi pelanggan baru. Dengan menggunakan model machine learning, perusahaan dapat mengidentifikasi pelanggan berisiko tinggi untuk churn lebih awal. Model ini memproses data historis perilaku pelanggan untuk mengungkap pola yang mengindikasikan risiko churn, sehingga memungkinkan perusahaan untuk mengambil langkah-langkah preventif melalui program retensi yang ditargetkan (Wagh & Wagh, 2022; Ahmad et al., 2019). Prediksi churn memungkinkan perusahaan mengambil langkah preventif terhadap pelanggan berisiko tinggi yang mungkin meninggalkan layanan. Strategi retensi ini lebih efektif dibandingkan akuisisi pelanggan baru, yang umumnya membutuhkan biaya lebih tinggi. Dengan demikian, kemampuan mengurangi churn dapat meningkatkan keuntungan perusahaan melalui pemanfaatan data pelanggan secara optimal (Dhangar & Anand, 2021).

## Business Understanding

### Problem Statements

- Bagaimana mengidentifikasi pelanggan yang cenderung berhenti berlangganan berdasarkan data historis?
- Variabel apa saja yang paling mempengaruhi kecenderungan pelanggan untuk churn?
- Berapa tingkat akurasi prediksi churn yang dapat dicapai dengan model yang dikembangkan?

### Goals

- Mengembangkan model yang mampu memprediksi churn pelanggan berdasarkan fitur-fitur yang tersedia.
- Mengidentifikasi variabel yang paling berpengaruh terhadap churn pelanggan untuk memberikan insight bagi perusahaan.
- Menghasilkan model prediktif dengan akurasi lebih baik agar dapat digunakan untuk strategi retensi pelanggan.

### Solution statements

- Menggunakan tiga model machine learning, yaitu Logistic Regression, Random Forest, dan XGBoost, untuk memprediksi churn pelanggan. Model dengan akurasi dan metrik evaluasi terbaik akan dipilih sebagai solusi akhir.
- Melakukan hyperparameter tuning pada model yang terpilih untuk mengoptimalkan performanya, dengan metrik evaluasi berupa akurasi, precision, recall, dan F1 score.

## Data Understanding

Dataset yang digunakan adalah Telco Customer Churn Dataset dari Kaggle, berisi informasi pelanggan yang telah berhenti berlangganan dan yang masih aktif, mencakup data demografis, informasi layanan, dan preferensi pembayaran.

Link Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Struktur Data
Jumlah Baris: 7,043
Jumlah Kolom: 21

### Variabel-variabel pada dataset ini sebagai berikut:
| Variabel         | Deskripsi                                               | Tipe Data  |
|------------------|---------------------------------------------------------|------------|
| customerID       | ID unik pelanggan                                       | object     |
| gender           | Jenis kelamin pelanggan                                 | object     |
| SeniorCitizen    | Indikator apakah pelanggan adalah orang tua             | int64      |
| Partner          | Apakah pelanggan memiliki pasangan                      | object     |
| Dependents       | Apakah pelanggan memiliki tanggungan                    | object     |
| tenure           | Lama waktu pelanggan menggunakan layanan                | int64      |
| PhoneService     | Apakah pelanggan memiliki layanan telepon               | object     |
| MultipleLines    | Apakah pelanggan memiliki beberapa sambungan telepon    | object     |
| InternetService  | Jenis layanan internet yang digunakan                   | object     |
| OnlineSecurity   | Apakah pelanggan memiliki layanan keamanan online       | object     |
| OnlineBackup     | Apakah pelanggan memiliki layanan pencadangan online    | object     |
| DeviceProtection | Apakah pelanggan memiliki layanan proteksi perangkat    | object     |
| TechSupport      | Apakah pelanggan memiliki layanan dukungan teknis       | object     |
| StreamingTV      | Apakah pelanggan memiliki layanan streaming TV          | object     |
| StreamingMovies  | Apakah pelanggan memiliki layanan streaming film        | object     |
| Contract         | Tipe kontrak (bulanan, satu tahun, dua tahun)           | object     |
| PaperlessBilling | Apakah pelanggan menggunakan penagihan tanpa kertas     | object     |
| PaymentMethod    | Metode pembayaran yang digunakan pelanggan              | object     |
| MonthlyCharges   | Biaya bulanan yang dikenakan                            | float64    |
| TotalCharges     | Total biaya yang telah dibayarkan                       | object    |
| Churn            | Apakah pelanggan meninggalkan layanan (ya/tidak)        | object     |


### Kondisi Data

- Missing Value: Pada kolom TotalCharges, nilai yang hilang diisi dengan nilai median untuk menjaga keseragaman data.
- Duplikasi: Tidak ditemukan data duplikat.
- Outlier: Pengecekan outlier dilakukan pada MonthlyCharges.

## Data Preparation

- Handling Missing Values: Nilai yang hilang pada kolom TotalCharges digantikan dengan median untuk menghindari pengaruh nilai kosong pada model.

- Outlier Detection and Handling: Untuk memastikan data bebas dari nilai ekstrem yang dapat mengganggu performa model, dilakukan deteksi outlier pada kolom MonthlyCharges menggunakan metode Interquartile Range (IQR). Outlier diidentifikasi sebagai nilai yang berada di luar rentang [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR], di mana Q1 dan Q3 adalah kuartil pertama dan ketiga. Nilai yang terdeteksi sebagai outlier ditangani dengan [menghapus data atau membatasi nilai pada batas maksimum yang telah ditentukan]. Hal ini bertujuan untuk memastikan distribusi nilai MonthlyCharges lebih seimbang dan stabil.

- Pemisahan fitur dan target: Pada langkah ini, data dipecah menjadi fitur (X) dan target (y). Fitur adalah variabel yang digunakan untuk memprediksi apakah pelanggan akan churn atau tidak, sementara target adalah kolom Churn yang akan diprediksi. Kolom customerID dihapus karena hanya berfungsi sebagai pengenal unik pelanggan dan tidak relevan dalam analisis. Kolom Churn dikonversi ke format numerik, di mana Yes direpresentasikan sebagai 1 (churn) dan No sebagai 0.

- Scaling Numerical Variables: Kolom numerik tenure, MonthlyCharges, dan TotalCharges dinormalisasi menggunakan StandardScaler untuk meningkatkan performa model dan mencegah bias akibat skala variabel yang berbeda. Scaling ini memastikan variabel-variabel numerik memiliki distribusi nilai yang serupa, yang dapat mempercepat konvergensi model.

- Encoding Categorical Variables: Kolom kategori seperti gender, InternetService, Contract, dan lainnya diubah menjadi nilai numerik menggunakan teknik Label Encoding. Konversi ini dilakukan untuk memungkinkan model memproses data kategori.

- Pembagian Data menjadi training dan testing: Data selanjutnya dibagi menjadi data latih (training set) dan data uji (testing set) dengan perbandingan 80:20. Hal ini bertujuan agar model dapat belajar dari sebagian besar data (80%) dan kemudian diuji performanya pada sisa data (20%) untuk mengevaluasi akurasi dan kemampuannya memprediksi data baru. Parameter random_state=42 digunakan agar pembagian data konsisten setiap kali kode dijalankan.

Langkah-langkah di atas memastikan data berada dalam format numerik yang siap diproses oleh model, serta meminimalisir pengaruh outlier, bias skala variabel, dan data kategori pada model prediksi churn.

## Modeling
Pada proyek ini, tiga model digunakan untuk memprediksi churn: Logistic Regression, Random Forest, dan XGBoost. Setiap model memiliki keunggulan yang berbeda dan diuji dengan tujuan memilih model terbaik untuk prediksi churn. Berikut ini adalah penjelasan detail dari setiap model beserta hyperparameter tuning yang dilakukan pada model XGBoost.

- Logistic Regression: Logistic Regression dipilih sebagai baseline karena kesederhanaannya dan kemampuannya untuk memberikan interpretasi yang baik terhadap hubungan antara fitur dan target. Logistic Regression menggunakan pendekatan regresi linier, kemudian menerapkan fungsi logistik (sigmoid) untuk mengubah nilai menjadi probabilitas. Untuk model ini menggunakan parameter dasar max_iter=200 untuk memastikan model memiliki cukup iterasi untuk konvergen.

- Random Forest: Random Forest adalah metode ensemble yang menggabungkan banyak decision tree untuk meningkatkan akurasi. Setiap tree di dalam Random Forest dilatih pada subset data yang berbeda untuk menangani variabilitas dan mengurangi risiko overfitting, terutama pada data yang kompleks atau non-linear. Dalam proyek ini, Random Forest digunakan dengan parameter n_estimators=100 (jumlah tree dalam model) dan max_depth=10 (kedalaman maksimum tree), yang diatur untuk keseimbangan antara performa dan kecepatan.

- XGBoost: XGBoost adalah algoritma boosting yang efektif untuk meningkatkan akurasi prediksi dengan komputasi yang cepat. Algoritma ini bekerja dengan cara membangun model secara bertahap, di mana model berikutnya berfokus pada kesalahan dari model sebelumnya. XGBoost menggunakan learning_rate, max_depth, dan n_estimators sebagai hyperparameter utama. Untuk model ini, hyperparameter tuning dilakukan menggunakan GridSearchCV untuk mencari kombinasi terbaik dari parameter-parameter tersebut. Berdasarkan hasil tuning, kombinasi parameter terbaik yang diperoleh adalah:
  - learning_rate: 0.1
  - max_depth: 3
  - n_estimators: 50



## Evaluation

### Logistic Regression

- Accuracy: 82.1%
- Precision (Churn): 0.69
- Recall (Churn): 0.60
- F1-Score (Churn): 0.64

Model ini menunjukkan bahwa model Logistic Regression dapat mengklasifikasikan 82.1% sampel secara benar, baik churn maupun non-churn. Precision menunjukkan persentase prediksi churn yang benar-benar churn dari semua prediksi churn yang dibuat. Dengan nilai 0.69, ini berarti 69% dari prediksi churn adalah benar-benar churn. Semakin tinggi precision, semakin baik model dalam menghindari false positives (salah deteksi churn). Recall mengukur persentase pelanggan yang churn yang benar-benar terdeteksi oleh model dari keseluruhan pelanggan yang churn. Nilai recall 0.60 menunjukkan bahwa model berhasil mengidentifikasi 60% dari total pelanggan yang churn. Semakin tinggi recall, semakin baik model dalam menangkap churn yang sebenarnya. F1-Score adalah rata-rata harmonis dari precision dan recall. Dengan F1-score 0.64, model menunjukkan keseimbangan antara ketepatan dan cakupan dalam mendeteksi churn.

### Random Forest

- Accuracy: 80.4%
- Precision (Churn): 0.68
- Recall (Churn): 0.50
- F1-Score (Churn): 0.58

Akurasi Random Forest sedikit lebih rendah daripada Logistic Regression, yaitu 80.4%. Precision 0.68 menunjukkan bahwa model masih dapat memprediksi churn dengan ketepatan yang baik. Namun, precision ini sedikit lebih rendah dibandingkan Logistic Regression. Recall 0.50 untuk churn berarti bahwa model hanya mampu mendeteksi 50% pelanggan yang churn. Artinya, banyak pelanggan yang sebenarnya churn tetapi tidak terdeteksi oleh model ini, menunjukkan recall yang lebih rendah dibandingkan Logistic Regression. Dengan nilai F1-Score 0.58, model ini menunjukkan bahwa keseimbangan antara precision dan recall untuk mendeteksi churn masih lebih rendah dibandingkan Logistic Regression.

### XGBoost

- Accuracy: 80.4%
- Precision (Churn): 0.66
- Recall (Churn): 0.54
- F1-Score (Churn): 0.59

Akurasi XGBoost juga berada pada tingkat yang sama dengan Random Forest, yaitu 80.4%. Precision 0.66 untuk churn menunjukkan bahwa XGBoost memiliki tingkat ketepatan yang lebih rendah dibandingkan Logistic Regression tetapi sedikit lebih tinggi daripada Random Forest. Recall 0.54 menunjukkan bahwa XGBoost dapat menangkap 54% pelanggan yang churn. Meskipun recall lebih tinggi dari Random Forest, nilainya masih di bawah Logistic Regression. F1-score 0.59 menunjukkan bahwa model ini memiliki keseimbangan antara precision dan recall yang lebih baik dibandingkan Random Forest namun masih lebih rendah dibandingkan Logistic Regression.

### Metrik Evaluasi
Berikut adalah penjelasan dan rumus dari metrik evaluasi yang digunakan pada proyek ini:
##### 1. Akurasi
Akurasi mengukur proporsi prediksi yang benar dibandingkan dengan total jumlah prediksi. Rumusnya adalah:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

- **TP** (True Positive): jumlah kasus positif yang benar terdeteksi.
- **TN** (True Negative): jumlah kasus negatif yang benar terdeteksi.
- **FP** (False Positive): jumlah kasus negatif yang salah terdeteksi sebagai positif.
- **FN** (False Negative): jumlah kasus positif yang salah terdeteksi sebagai negatif.

##### 2. Presisi
Presisi mengukur akurasi dari prediksi positif yang dibuat oleh model. Presisi menunjukkan seberapa banyak dari semua prediksi positif yang benar-benar positif. Semakin tinggi nilai presisi, semakin sedikit false positives. Rumusnya adalah:

Precision = TP / (TP + FP)

##### 3. Recall
Recall mengukur seberapa banyak dari semua kasus positif yang benar-benar terdeteksi oleh model. Recall menunjukkan kemampuan model untuk menangkap semua kasus positif. Semakin tinggi nilai recall, semakin sedikit false negatives. Rumusnya adalah:

Recall = TP / (TP + FN)

##### 4. F1-Score
F1-Score adalah rata-rata harmonis dari presisi dan recall. F1-Score memberikan keseimbangan antara presisi dan recall. Nilai F1-Score yang tinggi menunjukkan bahwa model memiliki performa baik dalam kedua metrik tersebut. Rumusnya adalah:

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)


Metrik evaluasi tersebut membantu dalam menilai performa model dalam memprediksi churn. Memahami metrik-metrik ini penting untuk menganalisis seberapa efektif model dalam mencapai tujuan proyek.


**Kesimpulan**: 
Dari hasil di atas, Logistic Regression memiliki performa terbaik untuk prediksi churn dalam hal accuracy (82.1%), precision (0.69), recall (0.60), dan F1-score (0.64). Meskipun Random Forest dan XGBoost adalah algoritma yang lebih kompleks, keduanya tidak mencapai performa yang lebih baik dari Logistic Regression, terutama dalam hal recall dan F1-score. Dengan kata lain, Logistic Regression adalah model terbaik karena memiliki keseimbangan antara ketepatan dan cakupan yang lebih baik dalam mendeteksi pelanggan yang kemungkinan churn.

**Variabel Penting yang Mempengaruhi Churn**:
- Tipe Kontrak Bulanan: Pelanggan dengan kontrak bulanan cenderung lebih berisiko churn dibandingkan dengan pelanggan kontrak jangka panjang.
- Layanan Keamanan Online: Pelanggan tanpa layanan keamanan online lebih mungkin untuk churn.
- Tagihan Bulanan Tinggi: Tagihan bulanan yang tinggi merupakan faktor signifikan dalam risiko churn.

Dari hasil evaluasi model yang digunakan, dampaknya terhadap pemahaman bisnis dapat dijelaskan sebagai berikut:
1. Kemampuan Model dalam Menjawab Problem Statements
Model Logistic Regression berhasil menjawab pertanyaan utama dari problem statements:
- Mengidentifikasi Pelanggan yang Berpotensi Churn: Dengan akurasi 82.1%, Logistic Regression dapat mengidentifikasi pelanggan yang berpotensi churn dengan baik. Tingginya nilai precision (0.69) dan recall (0.60) menunjukkan bahwa model ini efektif dalam mengenali pelanggan berisiko tinggi untuk churn.
- Mengidentifikasi Variabel Penting yang Mempengaruhi Churn: Melalui evaluasi variabel yang berpengaruh, ditemukan bahwa pelanggan dengan kontrak bulanan, tanpa layanan keamanan online, dan tagihan bulanan yang tinggi memiliki risiko churn lebih besar. Temuan ini memberikan informasi berharga yang dapat digunakan perusahaan untuk memprioritaskan intervensi terhadap pelanggan dengan karakteristik serupa.
- Mencapai Tingkat Akurasi yang Diinginkan: Model Logistic Regression menunjukkan tingkat akurasi tertinggi di antara ketiga model yang diuji, yaitu 82.1%, yang mendekati target akurasi yang diharapkan. Ini membuktikan bahwa model ini dapat digunakan sebagai dasar dalam memprediksi churn pelanggan.

2. Pencapaian Goals yang Diharapkan

- Prediksi Churn Pelanggan: Model ini berhasil memprediksi churn dengan akurasi yang cukup tinggi, mencapai tujuan untuk membangun model prediktif yang andal dalam mengklasifikasikan pelanggan churn.
- Memberikan Insight bagi Perusahaan: Dengan mengidentifikasi variabel yang paling signifikan dalam memengaruhi churn, model ini memberikan wawasan yang berguna bagi perusahaan. Insight ini memungkinkan perusahaan untuk menyusun strategi retensi yang tepat sasaran, seperti menawarkan paket kontrak yang lebih panjang atau memberikan penawaran khusus pada pelanggan dengan tagihan bulanan yang tinggi.
- Pengembangan Model dengan Akurasi yang Lebih Baik: Model ini mencapai akurasi yang lebih tinggi dibandingkan model lain dalam proyek ini, memenuhi tujuan untuk menghasilkan model dengan performa terbaik untuk strategi retensi pelanggan.

3. Dampak Solusi Terhadap Business Understanding

- Logistic Regression sebagai Solusi Akhir: Berdasarkan evaluasi metrik, Logistic Regression memberikan keseimbangan terbaik antara akurasi, precision, dan recall untuk memprediksi churn. Model ini memungkinkan perusahaan untuk mengidentifikasi pelanggan churn dengan lebih baik, memberikan dampak signifikan terhadap upaya perusahaan dalam menjaga pelanggan setia.
- Optimalisasi Model Melalui Hyperparameter Tuning: Hyperparameter tuning yang dilakukan pada XGBoost belum berhasil meningkatkan performa yang lebih baik dari Logistic Regression. Namun, penggunaan hyperparameter tuning tetap memberikan informasi tambahan bagi perusahaan mengenai model alternatif dan kekuatannya, sebagai pertimbangan untuk evaluasi lebih lanjut jika data berubah atau diperbarui.
- Penggunaan Insight untuk pengambilan keputusan: Identifikasi faktor penting yang memengaruhi churn, seperti tipe kontrak bulanan dan layanan keamanan online, memungkinkan perusahaan untuk merancang program retensi yang lebih tepat sasaran. Dengan demikian, model dan insight ini memiliki dampak nyata dalam memungkinkan perusahaan untuk melakukan intervensi lebih awal pada pelanggan berisiko.

Secara keseluruhan, model Logistic Regression memberikan solusi yang mampu memenuhi setiap aspek dari Problem Statements, Goals, dan Solution Statements yang diajukan dalam proyek ini.

Referensi: 
- Sarang Narkhede. Understanding Confusion Matrix. May 2018. https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62.
- Ahmad, A.K., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning in big data platform. Journal of Big Data, 6(1), 28. https://doi.org/10.1186/s40537-019-0191-6
- Dhangar, K., & Anand, P. (2021). A Review on Customer Churn Prediction Using Machine Learning Approach. International Journal of Innovations in Engineering Research and Technology, 8(05), 193-201. https://doi.org/10.17605/OSF.IO/ACNKJ
- Wagh, S. K., & Wagh, K. S. (2022). Customer Churn Prediction in Telecom Sector Using Machine Learning Techniques. SSRN. Available at: https://ssrn.com/abstract=4158415