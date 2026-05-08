import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder


# =========================
# HEADER
# =========================
st.title("🤖 AutoML Web App")
st.markdown("### Build, Train, Evaluate, and Predict in One Place")
st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("## 🤖 AutoML App")
st.sidebar.caption("Regression & Classification")
st.sidebar.divider()
st.sidebar.caption("© AutoML App 2026")

# =========================
# SESSION
# =========================
for key in ["data", "model", "results", "target", "features", "task", "label_encoder"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# NAVBAR
# =========================
tab1, tab2, tab3 = st.tabs(["🏠 AutoML", "📘 Tentang", "📖 Cara Pakai"])

# =========================
# TAB 1
# =========================
with tab1:

    st.header("🚀 Workflow AutoML")
    st.write(
        "AutoML membantu Anda membangun model machine learning secara otomatis "
        "mulai dari upload dataset, preprocessing, training model, evaluasi, "
        "hingga prediksi tanpa coding yang rumit."
    )

    # =========================
    # UPLOAD
    # =========================
    with st.container():

        st.subheader("📂 Upload Dataset")

        file = st.file_uploader(
            "Upload Dataset CSV",
            type=["csv"]
        )

        if file:
            df = pd.read_csv(file)
            st.session_state.data = df
            st.success("✅ Dataset berhasil diupload!")

    # =========================
    # MAIN PROCESS
    # =========================
    if st.session_state.data is not None:

        df = st.session_state.data

        # =========================
        # OVERVIEW
        # =========================
        with st.container():

            st.subheader("📊 Data Overview")

            st.dataframe(
                df,
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("### Statistik Data")
                st.dataframe(
                    df.describe(),
                    use_container_width=True
                )

            with col2:
                st.write("### Missing Value")

                missing_df = pd.DataFrame({
                    "Kolom": df.columns,
                    "Missing": df.isnull().sum().values
                })

                st.dataframe(
                    missing_df,
                    use_container_width=True
                )

        st.divider()

        # =========================
        # VISUALIZATION
        # =========================
        with st.container():

            st.subheader("📈 Visualization")

            col = st.selectbox(
                "Pilih Kolom",
                df.columns
            )

            fig, ax = plt.subplots()
            sns.histplot(df[col], ax=ax)

            fig2, ax2 = plt.subplots()
            sns.heatmap(
                df.corr(numeric_only=True),
                annot=True,
                ax=ax2
            )

            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(fig)

            with col2:
                st.pyplot(fig2)

        st.divider()

        # =========================
        # TRAINING
        # =========================
        with st.expander("⚙️ Konfigurasi Training", expanded=True):

            st.subheader("🤖 Model Training")

            target = st.selectbox(
                "Pilih Target",
                df.columns
            )

            df = df.dropna()

            X = df.drop(target, axis=1)
            y = df[target]

            # =========================
            # DETEKSI TASK
            # =========================
            if y.dtype == 'object' or y.nunique() < 10:

                task = "classification"
                scoring = "accuracy"
                best_score = 0

                st.info("📌 Task Detected: Classification")

                le = LabelEncoder()
                y = le.fit_transform(y)

                st.session_state.label_encoder = le

                cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                )

            else:

                task = "regression"
                scoring = "r2"
                best_score = -999

                st.info("📌 Task Detected: Regression")

                st.session_state.label_encoder = None

                cv = KFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                )

            # =========================
            # DETEKSI KOLOM
            # =========================
            categorical_cols = X.select_dtypes(
                include='object'
            ).columns.tolist()

            numeric_cols = X.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()

            # =========================
            # PILIH ORDINAL MANUAL
            # =========================
            ordinal_cols = st.multiselect(
                "Pilih Kolom Ordinal",
                categorical_cols,
                help="Contoh ordinal: Rendah < Sedang < Tinggi"
            )

            categorical_cols = [
                c for c in categorical_cols
                if c not in ordinal_cols
            ]

            # =========================
            # PREPROCESSING
            # =========================
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ("ord", OrdinalEncoder(), ordinal_cols)
            ])

            # =========================
            # MODEL
            # =========================
            if task == "classification":

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier()
                }

            else:

                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor()
                }

            # =========================
            # TRAIN
            # =========================
            if st.button("🚀 Train AutoML"):

                results = []
                best_model = None
                best_score_local = best_score

                for name, m in models.items():

                    pipe = Pipeline([
                        ("prep", preprocessor),
                        ("model", m)
                    ])

                    scores = cross_val_score(
                        pipe,
                        X,
                        y,
                        cv=cv,
                        scoring=scoring
                    )

                    mean_score = scores.mean()

                    results.append({
                        "Model": name,
                        "Score": mean_score
                    })

                    if mean_score > best_score_local:
                        best_score_local = mean_score
                        best_model = pipe

                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values(
                    by="Score",
                    ascending=False
                )

                # =========================
                # EVALUASI
                # =========================
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                best_model.fit(X_train, y_train)

                y_pred = best_model.predict(X_test)

                st.subheader("📊 Evaluasi Model")

                # =========================
                # CLASSIFICATION
                # =========================
                if task == "classification":

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Accuracy",
                            round(
                                accuracy_score(y_test, y_pred),
                                4
                            )
                        )

                    with col2:
                        st.metric(
                            "Jumlah Data Test",
                            len(y_test)
                        )

                    cm = confusion_matrix(y_test, y_pred)

                    fig_cm, ax_cm = plt.subplots()

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        ax=ax_cm
                    )

                    with st.expander("📌 Confusion Matrix"):
                        st.pyplot(fig_cm)

                    report = classification_report(
                        y_test,
                        y_pred,
                        output_dict=True
                    )

                    with st.expander("📄 Classification Report"):
                        st.dataframe(
                            pd.DataFrame(report).transpose(),
                            use_container_width=True
                        )

                # =========================
                # REGRESSION
                # =========================
                else:

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "R2 Score",
                            round(
                                r2_score(y_test, y_pred),
                                4
                            )
                        )

                    with col2:
                        st.metric(
                            "MAE",
                            round(
                                mean_absolute_error(y_test, y_pred),
                                4
                            )
                        )

                    with col3:
                        st.metric(
                            "MSE",
                            round(
                                mean_squared_error(y_test, y_pred),
                                4
                            )
                        )

                # =========================
                # FINAL MODEL
                # =========================
                best_model.fit(X, y)

                st.session_state.model = best_model
                st.session_state.task = task
                st.session_state.features = X.columns.tolist()

                # =========================
                # LEADERBOARD
                # =========================
                st.subheader("🏆 Leaderboard Model")

                st.dataframe(
                    df_results,
                    use_container_width=True
                )

                fig, ax = plt.subplots()

                ax.barh(
                    df_results["Model"],
                    df_results["Score"]
                )

                st.pyplot(fig)

        st.divider()

        # =========================
        # PREDICTION
        # =========================
        if st.session_state.model is not None:

            st.subheader("🔮 Prediction")

            input_data = {}

            cols = st.columns(2)

            for i, col in enumerate(st.session_state.features):

                with cols[i % 2]:

                    if df[col].dtype == 'object':

                        val = st.selectbox(
                            col,
                            df[col].unique()
                        )

                    else:

                        val = st.number_input(
                            col,
                            value=float(df[col].mean())
                        )

                    input_data[col] = val

            if st.button("Prediksi"):

                input_df = pd.DataFrame([input_data])

                result = st.session_state.model.predict(input_df)

                if st.session_state.task == "classification":

                    if st.session_state.label_encoder:
                        result = st.session_state.label_encoder.inverse_transform(result)

                st.success(f"✅ Hasil Prediksi: {result[0]}")

        st.divider()

        # =========================
        # DOWNLOAD
        # =========================
        if st.session_state.model is not None:

            st.subheader("⬇️ Download Model")

            st.success("✅ Model siap didownload")

            buffer = io.BytesIO()

            joblib.dump(
                st.session_state.model,
                buffer
            )

            buffer.seek(0)

            st.download_button(
                "Download Model",
                buffer,
                "model.joblib"
            )
    # =========================
# TAB 2 - TENTANG
# =========================
with tab2:

    st.header("📘 Tentang AutoML")

    st.write("""
    AutoML (Automated Machine Learning) adalah teknologi yang membantu proses
    pembuatan model machine learning secara otomatis tanpa perlu melakukan
    coding yang rumit.

    Dengan AutoML, pengguna dapat melakukan:
    - Upload dataset
    - Preprocessing data otomatis
    - Training model machine learning
    - Evaluasi performa model
    - Prediksi data baru

    Semua proses tersebut dilakukan dalam satu workflow yang sederhana
    dan mudah digunakan.
    """)

    st.subheader("🤖 Apa Itu Machine Learning?")

    st.write("""
    Machine Learning adalah cabang dari Artificial Intelligence (AI)
    yang memungkinkan komputer belajar dari data untuk membuat prediksi
    atau keputusan secara otomatis.

    Contoh penggunaan machine learning:
    - Prediksi harga rumah
    - Klasifikasi email spam
    - Prediksi kelulusan siswa
    - Prediksi penjualan
    - Sistem rekomendasi
    """)

    st.subheader("🚀 Fungsi AutoML Web App")

    st.write("""
    Aplikasi ini membantu pengguna untuk:

    ✅ Upload dataset CSV  
    ✅ Analisis data otomatis  
    ✅ Visualisasi data  
    ✅ Training beberapa model machine learning  
    ✅ Evaluasi model otomatis  
    ✅ Membandingkan performa model  
    ✅ Prediksi data baru  
    ✅ Download model siap pakai  

    Aplikasi ini mendukung:
    - Classification
    - Regression
    """)

    st.subheader("⚙️ Algoritma yang Digunakan")

    st.write("""
    ### Classification
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier

    ### Regression
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    """)

    st.subheader("📊 Evaluasi Model")

    st.write("""
    Untuk Classification:
    - Accuracy
    - Confusion Matrix
    - Classification Report

    Untuk Regression:
    - R2 Score
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    """)

# =========================
# TAB 3 - CARA PAKAI
# =========================
with tab3:

    st.header("📖 Cara Menggunakan AutoML")

    st.subheader("1️⃣ Upload Dataset")

    st.write("""
    Upload dataset dalam format CSV.

    Tips dataset:
    - Pastikan memiliki header kolom
    - Hindari terlalu banyak missing value
    - Gunakan data yang relevan
    """)

    st.subheader("2️⃣ Data Overview")

    st.write("""
    Setelah dataset diupload, sistem akan menampilkan:
    - Tabel dataset
    - Statistik data
    - Missing value

    Tahap ini membantu memahami kondisi dataset sebelum training.
    """)

    st.subheader("3️⃣ Visualisasi Data")

    st.write("""
    Anda dapat:
    - Melihat distribusi data
    - Melihat heatmap korelasi

    Visualisasi membantu memahami pola data.
    """)

    st.subheader("4️⃣ Pilih Target")

    st.write("""
    Target adalah kolom yang ingin diprediksi.

    Contoh:
    - Classification → Status Kelulusan
    - Regression → Harga Rumah
    """)

    st.subheader("5️⃣ Pilih Kolom Ordinal")

    st.write("""
    Pilih kolom ordinal jika memiliki urutan.

    Contoh:
    - Rendah
    - Sedang
    - Tinggi
    """)

    st.subheader("6️⃣ Training Model")

    st.write("""
    Klik tombol Train AutoML.

    Sistem akan:
    - Melakukan preprocessing
    - Melatih beberapa model
    - Membandingkan performa model
    - Memilih model terbaik
    """)

    st.subheader("7️⃣ Evaluasi Model")

    st.write("""
    Sistem akan menampilkan evaluasi model secara otomatis.

    Classification:
    - Accuracy
    - Confusion Matrix
    - Classification Report

    Regression:
    - R2 Score
    - MAE
    - MSE
    """)

    st.subheader("8️⃣ Prediction")

    st.write("""
    Masukkan data baru untuk melakukan prediksi.

    Untuk classification:
    - Sistem akan menampilkan hasil prediksi
    - Menampilkan tingkat keyakinan model
    """)

    st.subheader("9️⃣ Download Model")

    st.write("""
    Model terbaik dapat didownload dalam format .joblib
    dan digunakan kembali tanpa training ulang.
    """)

    st.subheader("🎯 Tips Penggunaan")

    st.write("""
    - Gunakan dataset yang bersih
    - Hindari terlalu banyak missing value
    - Pilih target dengan benar
    - Gunakan fitur yang relevan
    - Semakin baik data, semakin baik model
    """)
