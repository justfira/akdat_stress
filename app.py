import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import sqlite3

# ==============================
# DATABASE SETUP
# ==============================
conn = sqlite3.connect('stres_mahasiswa.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS hasil_analisis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sumber_data TEXT,
    nama_file TEXT,
    jumlah_data INTEGER,
    cluster INTEGER,
    rata_rata_stres REAL,
    kategori TEXT,
    rekomendasi TEXT
)
''')
conn.commit()

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(page_title="Analisis Tingkat Stres Mahasiswa", page_icon="🎓", layout="wide")

menu = st.sidebar.radio("📋 Navigasi", ["Analisis", "Riwayat Analisis"])

st.sidebar.markdown("### Dibuat oleh:")
st.sidebar.markdown("""
- Vania Zhafira Zahra  
- Putri Diva Riyanti  
- Reva Hanum Salsabila  
""")
st.sidebar.markdown("© 2025 Kelompok MPSI - Akuisisi Data")

# ==============================
# HALAMAN 1 - ANALISIS
# ==============================
if menu == "Analisis":
    st.title("🎓 Analisis Tingkat Stres Mahasiswa Menggunakan Clustering")

    if "clustering_done" not in st.session_state:
        st.session_state.clustering_done = False

    disabled_state = st.session_state.clustering_done

    # --- PILIH SUMBER DATA ---
    st.header("📂 Pilih Sumber Data")
    source = st.radio("Pilih sumber data:", ["Upload File CSV", "Input Manual"], disabled=disabled_state)

    df = None

    # --- OPSI 1: UPLOAD FILE ---
    if source == "Upload File CSV":
        uploaded_file = st.file_uploader("Unggah file CSV dataset stres mahasiswa", type=["csv"], disabled=disabled_state)
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.source = "Upload File CSV"
            st.session_state.file_name = uploaded_file.name

            st.success(f"✅ File '{uploaded_file.name}' berhasil diunggah.")
            st.write("Jumlah data:", len(df), "baris")

            if st.checkbox("Lihat dataset mentah", disabled=False):
                st.dataframe(df.head(20))
        else:
            if not st.session_state.clustering_done:
                st.warning("⚠️ Silakan upload file CSV terlebih dahulu.")
                st.stop()

    # --- OPSI 2: INPUT MANUAL ---
    else:
        st.header("✏️ Input Data Mahasiswa Manual (Minimal 5 Data)")
        st.info("Masukkan minimal 5 data mahasiswa. Kamu bisa menambah atau menghapus data dengan tombol di bawah.")

        if "count" not in st.session_state:
            st.session_state.count = 5

        cgpa_options = [
            "0.00 - 0.99", "1.00 - 1.99", "2.00 - 2.49",
            "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"
        ]

        data_manual = []
        for i in range(st.session_state.count):
            st.subheader(f"Data Mahasiswa ke-{i+1}")
            col1, col2 = st.columns(2)
            age = col1.number_input(f"Umur (Age) - {i+1}", min_value=17, max_value=30, value=20, key=f"age_{i}", disabled=disabled_state)
            cgpa_range = col2.selectbox(f"Rentang CGPA - {i+1}", cgpa_options, key=f"cgpa_{i}", disabled=disabled_state)
            gender = st.selectbox(f"Jenis Kelamin - {i+1}", ["Male", "Female"], key=f"gender_{i}", disabled=disabled_state)
            year = st.selectbox(f"Tahun Studi - {i+1}", ["Year 1", "Year 2", "Year 3", "Year 4"], key=f"year_{i}", disabled=disabled_state)
            depression = st.selectbox(f"Apakah memiliki depresi? - {i+1}", ["No", "Yes"], key=f"dep_{i}", disabled=disabled_state)
            anxiety = st.selectbox(f"Apakah memiliki kecemasan? - {i+1}", ["No", "Yes"], key=f"anx_{i}", disabled=disabled_state)
            panic = st.selectbox(f"Apakah mengalami panic attack? - {i+1}", ["No", "Yes"], key=f"panic_{i}", disabled=disabled_state)
            specialist = st.selectbox(f"Apakah pernah konsultasi ke spesialis? - {i+1}", ["No", "Yes"], key=f"spec_{i}", disabled=disabled_state)

            data_manual.append({
                "Choose your gender": gender,
                "Your current year of Study": year,
                "Age": age,
                "What is your CGPA?": cgpa_range,
                "Do you have Depression?": depression,
                "Do you have Anxiety?": anxiety,
                "Do you have Panic attack?": panic,
                "Did you seek any specialist for a treatment?": specialist
            })

        colx1, colx2 = st.columns(2)
        if colx1.button("➕ Tambah Data", disabled=disabled_state):
            st.session_state.count += 1
            st.rerun()
        if st.session_state.count > 5:
            if colx2.button("🗑️ Hapus Data Terakhir", disabled=disabled_state):
                st.session_state.count -= 1
                st.rerun()

        df = pd.DataFrame(data_manual)
        st.session_state.df = df
        st.session_state.source = "Input Manual"
        st.session_state.file_name = "Manual Input"

        st.success(f"✅ {len(df)} data mahasiswa telah dimasukkan.")
        st.dataframe(df)

    # --- KONFIRMASI ---
    st.header("✅ Konfirmasi Analisis")
    if "ok_run" not in st.session_state:
        st.session_state.ok_run = False

    if not st.session_state.ok_run and not st.session_state.clustering_done:
        if st.button("OK, Jalankan Analisis", disabled=disabled_state):
            st.session_state.ok_run = True
            st.rerun()

    if not st.session_state.ok_run and not st.session_state.clustering_done:
        st.info("Klik tombol **OK, Jalankan Analisis** untuk memulai pemrosesan data.")
        st.stop()

    # =============================
    # PREPROCESSING
    # =============================
    # --- Safety check sebelum preprocessing ---
    if df is None:
        # Jika df lokal hilang, ambil dari session_state
        df = st.session_state.get("df", None)

    # Kalau setelah dicek tetap tidak ada, stop biar tidak error
    if df is None:
        st.warning("⚠️ Dataset belum dimuat. Silakan upload file CSV atau isi data manual terlebih dahulu.")
        st.stop()

# Sekarang aman untuk menyalin data
    df_proc = df.copy()
    df_proc = df_proc.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Standarkan kapitalisasi
    if "Your current year of Study" in df_proc.columns:
        df_proc["Your current year of Study"] = df_proc["Your current year of Study"].astype(str).str.strip().str.title()
    if "Choose your gender" in df_proc.columns:
        df_proc["Choose your gender"] = df_proc["Choose your gender"].astype(str).str.strip().str.title()

    def convert_cgpa_range(val):
        if isinstance(val, str) and "-" in val:
            try:
                low, high = val.split("-")
                return (float(low) + float(high)) / 2
            except:
                return None
        else:
            return pd.to_numeric(val, errors="coerce")

    binary_cols = [
        'Do you have Depression?',
        'Do you have Anxiety?',
        'Do you have Panic attack?',
        'Did you seek any specialist for a treatment?'
    ]
    for col in binary_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].replace({'Yes': 1, 'No': 0})

    df_proc['Age'] = pd.to_numeric(df_proc['Age'], errors='coerce')
    df_proc['What is your CGPA?'] = df_proc['What is your CGPA?'].apply(convert_cgpa_range)
    df_proc = df_proc.dropna(subset=['Age'])
    selected_cols = ['Age', 'What is your CGPA?'] + binary_cols

    st.dataframe(df_proc[selected_cols])
    st.success("✅ Data berhasil diproses")

    # --- NORMALISASI ---
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc[selected_cols])

    # --- CLUSTERING ---
    st.header("🧩 Clustering Mahasiswa")
    k = st.slider("Pilih jumlah cluster (K)", 2, 6, 3, disabled=disabled_state)
    run_cluster = st.button("🚀 Jalankan K-Means", disabled=disabled_state)

    if run_cluster and not st.session_state.clustering_done:
        st.session_state.clustering_done = True
        st.rerun()

    if st.session_state.clustering_done:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        df_proc["Cluster"] = labels

        st.subheader("📊 Hasil Clustering")
        st.dataframe(df_proc[selected_cols + ["Cluster"]])

        st.subheader("📈 Visualisasi Data")
        fig = px.scatter(df_proc, x="Age", y="What is your CGPA?",
                         color=df_proc["Cluster"].astype(str),
                         title="Visualisasi Cluster (Age vs CGPA)",
                         hover_data=binary_cols)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribusi Jumlah Mahasiswa per Cluster")
        st.bar_chart(df_proc["Cluster"].value_counts().sort_index())

        # --- INTERPRETASI ---
        st.subheader("🧠 Interpretasi Tingkat Stres")
        mean_cluster = df_proc.groupby("Cluster")[selected_cols].mean()
        mean_cluster["Stress_Score"] = mean_cluster[['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']].mean(axis=1)
        sorted_clusters = mean_cluster.sort_values("Stress_Score", ascending=False).reset_index()

        for _, row in sorted_clusters.iterrows():
            cluster = int(row["Cluster"])
            stress = row["Stress_Score"]
            cgpa = row["What is your CGPA?"]
            age = round(row["Age"])
            level = "Tinggi 🔴" if stress > 0.7 else "Sedang 🟠" if stress > 0.3 else "Rendah 🟢"
            st.markdown(f"**Cluster {cluster}:** Tingkat stres **{level}** (skor {stress:.2f}), umur {age}, IPK {cgpa:.2f}")

        # --- ANALISIS GENDER ---
        st.subheader("👩‍🎓 Analisis Berdasarkan Jenis Kelamin")
        gender_stress = df_proc.groupby("Choose your gender")[binary_cols].mean().mean(axis=1)
        st.bar_chart(gender_stress)
        for gender, score in gender_stress.items():
            rekom = (
                "Tekanan tinggi, perlu konseling." if score > 0.7 else
                "Stres sedang, butuh relaksasi." if score > 0.4 else
                "Stres rendah, pertahankan keseimbangan."
            )
            st.markdown(f"**{gender}** – Skor stres: {score:.2f} → {rekom}")

        # --- ANALISIS TAHUN STUDI ---
        st.subheader("🎓 Analisis Berdasarkan Tahun Studi")
        year_stress = df_proc.groupby("Your current year of Study")[binary_cols].mean().mean(axis=1)
        st.line_chart(year_stress)
        for year, score in year_stress.items():
            kondisi = (
                "Tekanan akademik tinggi (skripsi/tugas akhir)." if score > 0.7 else
                "Beban kuliah meningkat." if score > 0.4 else
                "Tingkat stres rendah."
            )
            st.markdown(f"**{year}** – Skor stres: {score:.2f} → {kondisi}")

        # --- SPK / REKOMENDASI ---
        top_gender = gender_stress.idxmax()
        top_year = year_stress.idxmax()
        rekom_utama = f"Fokuskan dukungan pada {top_gender} di {top_year}."
        st.success(f"💡 Kelompok paling berisiko: **{top_gender} - {top_year}**. {rekom_utama}")

        # Simpan hasil ke DB
        c.execute('''
            INSERT INTO hasil_analisis (sumber_data, nama_file, jumlah_data, cluster, rata_rata_stres, kategori, rekomendasi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            source,
            st.session_state.file_name,
            len(df_proc),
            k,
            float(sorted_clusters["Stress_Score"].mean()),
            f"{top_gender} - {top_year}",
            rekom_utama
        ))
        conn.commit()

        # Tombol ulang analisis
        st.markdown("---")
        if st.button("🔁 Ulangi Analisis"):
            for key in ["ok_run", "clustering_done", "count"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ==============================
# HALAMAN 2 - RIWAYAT ANALISIS
# ==============================
elif menu == "Riwayat Analisis":
    st.title("📜 Riwayat Analisis Tersimpan")
    hasil_df = pd.read_sql_query("SELECT * FROM hasil_analisis", conn)
    if hasil_df.empty:
        st.info("Belum ada hasil analisis yang disimpan.")
    else:
        st.dataframe(hasil_df)
        csv = hasil_df.to_csv(index=False).encode('utf-8')
        st.download_button("📤 Ekspor ke CSV", csv, "riwayat_analisis.csv", "text/csv")
