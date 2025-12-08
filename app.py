import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import numpy as np

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
c.execute('''
CREATE TABLE IF NOT EXISTS input_manual (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama_mahasiswa TEXT,
    gender TEXT,
    tahun_studi TEXT,
    umur INTEGER,
    ipk REAL,
    depresi TEXT,
    kecemasan TEXT,
    panic_attack TEXT,
    konsultasi_spesialis TEXT,
    cluster INTEGER,
    kategori_stres TEXT,
    stress_score REAL,
    tanggal_analisis TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()
st.set_page_config(page_title="Analisis", page_icon="🎓", layout="wide")

# CSS
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 260px !important; 
        }
        .main .block-container {
            padding-top: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-bottom: 2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("📋 Navigasi")

if "menu" not in st.session_state:
    st.session_state.menu = "Analisis"

if st.sidebar.button("🔎 Analisis"):
    st.session_state.menu = "Analisis"

if st.sidebar.button("📜 Riwayat Analisis"):
    st.session_state.menu = "Riwayat Analisis"

if st.sidebar.button("👤 Riwayat Input Manual"):
    st.session_state.menu = "Riwayat Input Manual"

menu = st.session_state.menu

st.sidebar.markdown("---")
st.sidebar.markdown("### Dibuat oleh:")
st.sidebar.markdown("""
- Vania Zhafira Zahra  
- Putri Diva Riyanti  
- Reva Hanum Salsabila  
""")
st.sidebar.markdown("© 2025 Kelompok 9 - Akuisisi Data")

# ==============================
# REFERENSI CLUSTER DARI SURVEY
# ==============================
CLUSTER_REFERENCE = {
    0: {
        'label': '🟢 Rendah',
        'stress_score': 0.16,
        'Age': 20.6,
        'CGPA': 3.33,
        'Depression': 0.22,
        'Anxiety': 0.00,
        'Panic': 0.27,
        'Specialist': 0.00,
        'count': 63
    },
    1: {
        'label': '🟠 Sedang',
        'stress_score': 0.62,
        'Age': 20.2,
        'CGPA': 3.47,
        'Depression': 0.48,
        'Anxiety': 1.00,
        'Panic': 0.39,
        'Specialist': 0.00,
        'count': 31
    },
    2: {
        'label': '🔴 Tinggi',
        'stress_score': 0.72,
        'Age': 21.0,
        'CGPA': 3.42,
        'Depression': 1.00,
        'Anxiety': 0.50,
        'Panic': 0.67,
        'Specialist': 1.00,
        'count': 6
    }
}

def classify_by_reference(row):
    """Klasifikasi berdasarkan jarak ke centroid cluster referensi"""
    features = np.array([
        row['Age'],
        row['What is your CGPA?'],
        row['Do you have Depression?'],
        row['Do you have Anxiety?'],
        row['Do you have Panic attack?'],
        row['Did you seek any specialist for a treatment?']
    ])
    
    min_distance = float('inf')
    assigned_cluster = 0
    
    for cluster_id, cluster_data in CLUSTER_REFERENCE.items():
        centroid = np.array([
            cluster_data['Age'],
            cluster_data['CGPA'],
            cluster_data['Depression'],
            cluster_data['Anxiety'],
            cluster_data['Panic'],
            cluster_data['Specialist']
        ])
        
        distance = np.linalg.norm(features - centroid)
        
        if distance < min_distance:
            min_distance = distance
            assigned_cluster = cluster_id
    
    return assigned_cluster, CLUSTER_REFERENCE[assigned_cluster]['label']

def create_gauge_chart(value, title):
    """Membuat gauge chart untuk skor stres"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#90EE90'},
                {'range': [33, 67], 'color': '#FFD700'},
                {'range': [67, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

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

    # --- OPSI 1: UPLOAD FILE ----
    if source == "Upload File CSV":
        # Panduan Format CSV
        with st.expander("📖 Panduan Format File CSV", expanded=False):
            st.markdown("""
            **Kolom yang Diperlukan:**
            - `Choose your gender` → Male/Female
            - `Age` → Umur (angka)
            - `Your current year of Study` → Year 1/Year 2/Year 3/Year 4
            - `What is your CGPA?` → Angka atau rentang (contoh: 3.50 atau 3.00 - 3.49)
            - `Do you have Depression?` → Yes/No
            - `Do you have Anxiety?` → Yes/No
            - `Do you have Panic attack?` → Yes/No
            - `Did you seek any specialist for a treatment?` → Yes/No
            
            **Contoh format header CSV:**
            ```
            Timestamp,Choose your gender,Age,What is your course?,Your current year of Study,What is your CGPA?,Marital status,Do you have Depression?,Do you have Anxiety?,Do you have Panic attack?,Did you seek any specialist for a treatment?
            ```
            
            ⚠️ **Catatan:** Header kolom harus sama persis (case-sensitive). Gunakan format CSV dengan encoding UTF-8.
            """)
        
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
        st.header("✏️ Input Data Mahasiswa Manual (Minimal 1 Data)")
        st.info("Masukkan minimal 1 data mahasiswa untuk mendapatkan hasil analisis cluster.")

        if "count" not in st.session_state:
            st.session_state.count = 1

        cgpa_options = [
            "0.00 - 0.99", "1.00 - 1.99", "2.00 - 2.49",
            "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"
        ]

        data_manual = []
        for i in range(st.session_state.count):
            st.subheader(f"Data Mahasiswa ke-{i+1}")
            
            # Input Nama Mahasiswa
            nama_mhs = st.text_input(f"Nama Mahasiswa - {i+1}", placeholder="Masukkan nama lengkap", key=f"nama_{i}", disabled=disabled_state)
            
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
                "Nama": nama_mhs if nama_mhs else f"Mahasiswa {i+1}",
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
        if st.session_state.count > 1:
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
    if df is None:
        df = st.session_state.get("df", None)

    if df is None:
        st.warning("⚠️ Dataset belum dimuat. Silakan upload file CSV atau isi data manual terlebih dahulu.")
        st.stop()

    df_proc = df.copy()
    df_proc = df_proc.applymap(lambda x: x.strip() if isinstance(x, str) else x)

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

    # --- CLUSTERING ---
    st.header("🧩 Analisis Tingkat Stres (3 Kategori: Rendah, Sedang, Tinggi)")
    run_cluster = st.button("🚀 Jalankan Analisis", disabled=disabled_state)

    if run_cluster and not st.session_state.clustering_done:
        st.session_state.clustering_done = True
        st.rerun()

    if st.session_state.clustering_done:
        n_data = len(df_proc)
        
        # === JIKA DATA < 3, PAKAI KLASIFIKASI BERDASARKAN REFERENSI ===
        if n_data < 3:
            st.info(f"📊 Menggunakan **klasifikasi berbasis referensi survey** (100 mahasiswa)")
            
            results = df_proc.apply(classify_by_reference, axis=1)
            df_proc["Cluster"] = results.apply(lambda x: x[0])
            df_proc["Kategori_Stres"] = results.apply(lambda x: x[1])
            
            # Penjelasan per mahasiswa
            for idx, row in df_proc.iterrows():
                cluster_id = row['Cluster']
                ref_data = CLUSTER_REFERENCE[cluster_id]
                nama_mhs = row.get('Nama', f'Mahasiswa {idx+1}')
                
                st.markdown("---")
                st.subheader(f"📊 Hasil Analisis: {nama_mhs}")
                
                # Gauge Chart untuk Stress Score
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    stress_score = (row['Do you have Depression?'] + row['Do you have Anxiety?'] + row['Do you have Panic attack?']) / 3
                    gauge_fig = create_gauge_chart(stress_score, "Tingkat Stres")
                    st.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_input_{idx}")
                
                with col2:
                    st.markdown(f"### {ref_data['label']}")
                    st.markdown(f"**Kategori Tingkat Stres**")
                    st.markdown(f"Skor: **{stress_score:.2f}** / 1.00")
                    st.markdown(f"---")
                    st.metric("Nama", nama_mhs)
                    st.metric("Jenis Kelamin", row['Choose your gender'])
                    st.metric("Tahun Studi", row['Your current year of Study'])
                    st.metric("Umur", f"{int(row['Age'])} tahun")
                    st.metric("IPK", f"{row['What is your CGPA?']:.2f}")
                
                # Bar Chart Gejala
                st.markdown("#### 📊 Profil Gejala Stres")
                symptoms_data = pd.DataFrame({
                    'Gejala': ['Depresi', 'Kecemasan', 'Panic Attack', 'Konsultasi Spesialis'],
                    'Status': [
                        row['Do you have Depression?'],
                        row['Do you have Anxiety?'],
                        row['Do you have Panic attack?'],
                        row['Did you seek any specialist for a treatment?']
                    ]
                })
                
                fig_symptoms = px.bar(
                    symptoms_data, 
                    x='Gejala', 
                    y='Status',
                    color='Status',
                    color_continuous_scale=['#90EE90', '#FF6B6B'],
                    title='Status Gejala (0 = Tidak, 1 = Ya)',
                    range_y=[0, 1]
                )
                fig_symptoms.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_symptoms, use_container_width=True, key=f"symptoms_{idx}")
                
                # Rekomendasi
                st.markdown("#### 💡 Rekomendasi")
                if cluster_id == 2:
                    st.error("⚠️ **Tingkat Stres Tinggi** - Segera konsultasi dengan psikolog/konselor kampus. Mahasiswa dengan profil serupa menunjukkan gejala serius yang memerlukan penanganan profesional.")
                elif cluster_id == 1:
                    st.warning("⚠️ **Tingkat Stres Sedang** - Pertimbangkan untuk melakukan aktivitas relaksasi seperti olahraga, meditasi, atau konseling preventif.")
                else:
                    st.success("✅ **Tingkat Stres Rendah** - Pertahankan pola hidup sehat dan keseimbangan antara akademik dan kehidupan pribadi.")
            
            # Simpan ke database HANYA SEKALI setelah semua analisis selesai
            if 'data_saved' not in st.session_state or not st.session_state.data_saved:
                for idx, row in df_proc.iterrows():
                    cluster_id = row['Cluster']
                    ref_data = CLUSTER_REFERENCE[cluster_id]
                    nama_mhs = row.get('Nama', f'Mahasiswa {idx+1}')
                    stress_score = (row['Do you have Depression?'] + row['Do you have Anxiety?'] + row['Do you have Panic attack?']) / 3
                    
                    c.execute('''
                        INSERT INTO input_manual (
                            nama_mahasiswa, gender, tahun_studi, umur, ipk, 
                            depresi, kecemasan, panic_attack, konsultasi_spesialis,
                            cluster, kategori_stres, stress_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        nama_mhs,
                        row['Choose your gender'],
                        row['Your current year of Study'],
                        int(row['Age']),
                        float(row['What is your CGPA?']),
                        'Yes' if row['Do you have Depression?'] == 1 else 'No',
                        'Yes' if row['Do you have Anxiety?'] == 1 else 'No',
                        'Yes' if row['Do you have Panic attack?'] == 1 else 'No',
                        'Yes' if row['Did you seek any specialist for a treatment?'] == 1 else 'No',
                        int(cluster_id),
                        ref_data['label'],
                        float(stress_score)
                    ))
                conn.commit()
                st.session_state.data_saved = True
        
        # === JIKA DATA >= 3, PAKAI K-MEANS ===
        else:
            st.info(f"📊 Menggunakan **K-Means Clustering** dengan {n_data} data")
            
            scaler = StandardScaler()
            X = scaler.fit_transform(df_proc[selected_cols])
            
            k = 3
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            df_proc["Cluster"] = labels

            mean_cluster = df_proc.groupby("Cluster")[selected_cols].mean()
            mean_cluster["Stress_Score"] = mean_cluster[['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']].mean(axis=1)
            sorted_clusters = mean_cluster.sort_values("Stress_Score", ascending=True).reset_index()
            
            cluster_label_map = {}
            kategori = ["🟢 Rendah", "🟠 Sedang", "🔴 Tinggi"]
            for idx, row in sorted_clusters.iterrows():
                cluster_label_map[row["Cluster"]] = kategori[idx]
            
            df_proc["Kategori_Stres"] = df_proc["Cluster"].map(cluster_label_map)

            st.subheader("📊 Hasil Clustering")
            st.dataframe(df_proc[selected_cols + ["Cluster", "Kategori_Stres"]])

            # === VISUALISASI UTAMA ===
            st.markdown("---")
            st.subheader("📈 Visualisasi Interaktif")
            
            # 3D Scatter Plot
            col1, col2 = st.columns(2)
            
            with col1:
                fig_3d = px.scatter_3d(
                    df_proc, 
                    x='Age', 
                    y='What is your CGPA?', 
                    z=df_proc[['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']].mean(axis=1),
                    color='Kategori_Stres',
                    title='Visualisasi 3D: Age vs IPK vs Stress Score',
                    labels={'z': 'Stress Score'},
                    color_discrete_map={'🟢 Rendah': '#90EE90', '🟠 Sedang': '#FFD700', '🔴 Tinggi': '#FF6B6B'},
                    hover_data=['Choose your gender', 'Your current year of Study']
                )
                fig_3d.update_layout(height=500)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with col2:
                # Sunburst Chart
                df_sunburst = df_proc.copy()
                df_sunburst['Count'] = 1
                fig_sunburst = px.sunburst(
                    df_sunburst,
                    path=['Kategori_Stres', 'Choose your gender', 'Your current year of Study'],
                    values='Count',
                    title='Distribusi Mahasiswa: Stres → Gender → Tahun',
                    color='Kategori_Stres',
                    color_discrete_map={'🟢 Rendah': '#90EE90', '🟠 Sedang': '#FFD700', '🔴 Tinggi': '#FF6B6B'}
                )
                fig_sunburst.update_layout(height=500)
                st.plotly_chart(fig_sunburst, use_container_width=True)

            # === DETAIL PER CLUSTER ===
            st.markdown("---")
            st.subheader("📋 Detail Rata-Rata per Cluster")
            
            for idx, row in sorted_clusters.iterrows():
                cluster_num = int(row["Cluster"])
                kategori_label = cluster_label_map[cluster_num]
                stress_score = row["Stress_Score"]
                jumlah_mhs = len(df_proc[df_proc["Cluster"] == cluster_num])
                
                with st.expander(f"**Cluster {cluster_num}: {kategori_label}** ({jumlah_mhs} mahasiswa)", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        gauge_fig = create_gauge_chart(stress_score, "Skor Stres")
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with col2:
                        st.metric("Rata-rata Umur", f"{row['Age']:.1f} tahun")
                        st.metric("Rata-rata IPK", f"{row['What is your CGPA?']:.2f}")
                        st.metric("Jumlah Mahasiswa", f"{jumlah_mhs} orang")
                    
                    with col3:
                        st.metric("Depresi", f"{row['Do you have Depression?']*100:.0f}%")
                        st.metric("Kecemasan", f"{row['Do you have Anxiety?']*100:.0f}%")
                        st.metric("Panic Attack", f"{row['Do you have Panic attack?']*100:.0f}%")
                    
                    # Radar Chart untuk cluster ini
                    cluster_data = df_proc[df_proc["Cluster"] == cluster_num][binary_cols].mean()
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=[cluster_data[col]*100 for col in binary_cols],
                        theta=['Depresi', 'Kecemasan', 'Panic Attack', 'Konsultasi Spesialis'],
                        fill='toself',
                        name=kategori_label
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        title=f"Profil Gejala Cluster {cluster_num}",
                        height=300
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            # === ANALISIS LANJUTAN ===
            st.markdown("---")
            st.subheader("📊 Analisis Demografis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gender Analysis
                gender_stress = df_proc.groupby("Choose your gender")[binary_cols].mean().mean(axis=1).reset_index()
                gender_stress.columns = ['Gender', 'Stress Score']
                fig_gender = px.bar(
                    gender_stress,
                    x='Gender',
                    y='Stress Score',
                    color='Stress Score',
                    title='Tingkat Stres per Gender',
                    color_continuous_scale=['#90EE90', '#FFD700', '#FF6B6B']
                )
                st.plotly_chart(fig_gender, use_container_width=True)
            
            with col2:
                # Year Analysis
                year_stress = df_proc.groupby("Your current year of Study")[binary_cols].mean().mean(axis=1).reset_index()
                year_stress.columns = ['Year', 'Stress Score']
                fig_year = px.line(
                    year_stress,
                    x='Year',
                    y='Stress Score',
                    markers=True,
                    title='Tingkat Stres per Tahun Studi',
                    line_shape='spline'
                )
                fig_year.update_traces(line_color='#FF6B6B', marker=dict(size=10))
                st.plotly_chart(fig_year, use_container_width=True)

            # Heatmap Korelasi
            st.markdown("#### 🔥 Heatmap Korelasi Antar Variabel")
            corr_data = df_proc[selected_cols].corr()
            fig_heatmap = px.imshow(
                corr_data,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdYlGn_r',
                title='Korelasi Antar Variabel Stres'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Rekomendasi
            st.markdown("---")
            st.subheader("💡 Rekomendasi Berdasarkan Analisis")
            gender_stress_dict = df_proc.groupby("Choose your gender")[binary_cols].mean().mean(axis=1).to_dict()
            year_stress_dict = df_proc.groupby("Your current year of Study")[binary_cols].mean().mean(axis=1).to_dict()
            
            if gender_stress_dict and year_stress_dict:
                top_gender = max(gender_stress_dict, key=gender_stress_dict.get)
                top_year = max(year_stress_dict, key=year_stress_dict.get)
                st.success(f"🎯 Kelompok paling berisiko: **{top_gender} - {top_year}**. Fokuskan dukungan psikologis dan program wellness pada kelompok ini.")

        # Simpan hasil ke DB
        avg_stress = df_proc[['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']].mean().mean()
        c.execute('''
            INSERT INTO hasil_analisis (sumber_data, nama_file, jumlah_data, cluster, rata_rata_stres, kategori, rekomendasi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            source,
            st.session_state.file_name,
            len(df_proc),
            3,
            float(avg_stress),
            "Mixed" if n_data >= 3 else df_proc["Kategori_Stres"].iloc[0],
            "Analisis selesai"
        ))
        conn.commit()

        # Tombol ulang analisis
        st.markdown("---")
        if st.button("🔁 Ulangi Analisis"):
            for key in ["ok_run", "clustering_done", "count", "data_saved"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ==============================
# HALAMAN 2 - RIWAYAT ANALISIS
# ==============================
elif menu == "Riwayat Analisis":
    st.title("📜 Riwayat Analisis Dataset")
    
    hasil_df = pd.read_sql_query("SELECT * FROM hasil_analisis ORDER BY id DESC", conn)
    
    if hasil_df.empty:
        st.info("Belum ada hasil analisis yang disimpan.")
    else:
        st.success(f"✅ Total {len(hasil_df)} analisis tersimpan")
        
        # Summary Cards
        st.subheader("📊 Ringkasan Analisis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_data = hasil_df['jumlah_data'].sum()
            st.metric("Total Data Dianalisis", f"{total_data:,}")
        
        with col2:
            avg_stress = hasil_df['rata_rata_stres'].mean()
            st.metric("Rata-rata Stress Score", f"{avg_stress:.2f}")
        
        with col3:
            upload_count = len(hasil_df[hasil_df['sumber_data'] == 'Upload File CSV'])
            st.metric("Analisis dari CSV", upload_count)
        
        with col4:
            manual_count = len(hasil_df[hasil_df['sumber_data'] == 'Input Manual'])
            st.metric("Analisis Manual", manual_count)
        
        # Visualisasi
        st.markdown("---")
        st.subheader("📈 Visualisasi Riwayat")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie Chart Sumber Data
            sumber_count = hasil_df['sumber_data'].value_counts().reset_index()
            sumber_count.columns = ['Sumber', 'Jumlah']
            fig_sumber = px.pie(
                sumber_count,
                values='Jumlah',
                names='Sumber',
                title='Distribusi Sumber Data',
                color_discrete_sequence=['#4CAF50', '#2196F3']
            )
            st.plotly_chart(fig_sumber, use_container_width=True, key="chart_sumber_data")
        
        with col2:
            # Bar Chart Jumlah Data per Analisis
            fig_jumlah = px.bar(
                hasil_df.head(10),
                x='nama_file',
                y='jumlah_data',
                color='rata_rata_stres',
                title='10 Analisis Terakhir (Jumlah Data)',
                labels={'nama_file': 'File', 'jumlah_data': 'Jumlah Data'},
                color_continuous_scale='RdYlGn_r'
            )
            fig_jumlah.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_jumlah, use_container_width=True, key="chart_jumlah_data")
        
        # Tren Stress Score
        st.subheader("📉 Tren Rata-rata Stress Score")
        fig_trend = px.line(
            hasil_df.iloc[::-1],  # Reverse untuk urutan kronologis
            x=hasil_df.index[::-1],
            y='rata_rata_stres',
            markers=True,
            title='Tren Rata-rata Stress Score per Analisis',
            labels={'x': 'Analisis ke-', 'rata_rata_stres': 'Stress Score'}
        )
        fig_trend.update_traces(line_color='#FF6B6B', marker=dict(size=10))
        st.plotly_chart(fig_trend, use_container_width=True, key="chart_trend_stress")
        
        # Detail per analisis
        st.markdown("---")
        st.subheader("📋 Detail Analisis")
        
        for idx, row in hasil_df.iterrows():
            # Tentukan warna berdasarkan stress score
            if row['rata_rata_stres'] >= 0.7:
                color = "🔴"
                border_color = "#FF6B6B"
            elif row['rata_rata_stres'] >= 0.4:
                color = "🟠"
                border_color = "#FFD700"
            else:
                color = "🟢"
                border_color = "#90EE90"
            
            with st.expander(f"{color} **Analisis #{row['id']}** - {row['nama_file']} ({row['jumlah_data']} data)", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📂 Informasi File**")
                    st.write(f"**Sumber:** {row['sumber_data']}")
                    st.write(f"**Nama File:** {row['nama_file']}")
                    st.write(f"**Jumlah Data:** {row['jumlah_data']} mahasiswa")
                    st.write(f"**Jumlah Cluster:** {row['cluster']}")
                
                with col2:
                    st.markdown("**📊 Hasil Analisis**")
                    st.write(f"**Rata-rata Stress Score:** {row['rata_rata_stres']:.2f}")
                    st.write(f"**Kategori Berisiko:** {row['kategori']}")
                    
                    # Gauge mini untuk stress score
                    mini_gauge = create_gauge_chart(row['rata_rata_stres'], "")
                    mini_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(mini_gauge, use_container_width=True, key=f"gauge_riwayat_{row['id']}")
                
                with col3:
                    st.markdown("**💡 Rekomendasi**")
                    st.write(row['rekomendasi'])
                    
                    # Status badge
                    if row['rata_rata_stres'] >= 0.7:
                        st.error("⚠️ **Status: Tingkat Stres Tinggi**")
                    elif row['rata_rata_stres'] >= 0.4:
                        st.warning("⚠️ **Status: Tingkat Stres Sedang**")
                    else:
                        st.success("✅ **Status: Tingkat Stres Rendah**")
        
        # Tabel Data
        st.markdown("---")
        st.subheader("📄 Tabel Data Lengkap")
        st.dataframe(
            hasil_df[['id', 'sumber_data', 'nama_file', 'jumlah_data', 'cluster', 'rata_rata_stres', 'kategori']],
            use_container_width=True
        )
        
        # Export & Delete
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            csv = hasil_df.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Ekspor Semua Riwayat ke CSV", csv, "riwayat_analisis.csv", "text/csv")
        
        with col2:
            if st.button("🗑️ Hapus Semua Riwayat", type="secondary"):
                c.execute("DELETE FROM hasil_analisis")
                conn.commit()
                st.success("✅ Semua riwayat berhasil dihapus!")
                st.rerun()

# ==============================
# HALAMAN 3 - RIWAYAT INPUT MANUAL
# ==============================
elif menu == "Riwayat Input Manual":
    st.title("👤 Riwayat Input Manual")
    
    manual_df = pd.read_sql_query("SELECT * FROM input_manual ORDER BY tanggal_analisis DESC", conn)
    
    if manual_df.empty:
        st.info("Belum ada data input manual yang disimpan.")
    else:
        st.success(f"✅ Total {len(manual_df)} data mahasiswa tersimpan")
        
        # Filter berdasarkan kategori stres
        st.subheader("🔍 Filter Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_kategori = st.multiselect(
                "Kategori Stres",
                options=manual_df['kategori_stres'].unique(),
                default=manual_df['kategori_stres'].unique()
            )
        
        with col2:
            filter_gender = st.multiselect(
                "Jenis Kelamin",
                options=manual_df['gender'].unique(),
                default=manual_df['gender'].unique()
            )
        
        with col3:
            filter_tahun = st.multiselect(
                "Tahun Studi",
                options=manual_df['tahun_studi'].unique(),
                default=manual_df['tahun_studi'].unique()
            )
        
        # Apply filters
        filtered_df = manual_df[
            (manual_df['kategori_stres'].isin(filter_kategori)) &
            (manual_df['gender'].isin(filter_gender)) &
            (manual_df['tahun_studi'].isin(filter_tahun))
        ]
        
        # Tampilkan tabel
        st.subheader("📋 Data Mahasiswa")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Statistik ringkas
        st.markdown("---")
        st.subheader("📊 Statistik Ringkas")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Mahasiswa", len(filtered_df))
        
        with col2:
            avg_stress = filtered_df['stress_score'].mean()
            st.metric("Rata-rata Stress Score", f"{avg_stress:.2f}")
        
        with col3:
            avg_ipk = filtered_df['ipk'].mean()
            st.metric("Rata-rata IPK", f"{avg_ipk:.2f}")
        
        with col4:
            avg_age = filtered_df['umur'].mean()
            st.metric("Rata-rata Umur", f"{avg_age:.1f}")
        
        # Visualisasi
        st.markdown("---")
        st.subheader("📈 Visualisasi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribusi kategori stres
            kategori_count = filtered_df['kategori_stres'].value_counts().reset_index()
            kategori_count.columns = ['Kategori', 'Jumlah']
            fig_kategori = px.pie(
                kategori_count,
                values='Jumlah',
                names='Kategori',
                title='Distribusi Kategori Stres',
                color='Kategori',
                color_discrete_map={'🟢 Rendah': '#90EE90', '🟠 Sedang': '#FFD700', '🔴 Tinggi': '#FF6B6B'}
            )
            st.plotly_chart(fig_kategori, use_container_width=True, key="chart_kategori_manual")
        
        with col2:
            # Distribusi per gender
            gender_count = filtered_df.groupby(['gender', 'kategori_stres']).size().reset_index(name='Jumlah')
            fig_gender = px.bar(
                gender_count,
                x='gender',
                y='Jumlah',
                color='kategori_stres',
                title='Distribusi Stres per Gender',
                barmode='group',
                color_discrete_map={'🟢 Rendah': '#90EE90', '🟠 Sedang': '#FFD700', '🔴 Tinggi': '#FF6B6B'}
            )
            st.plotly_chart(fig_gender, use_container_width=True, key="chart_gender_manual")
        
        # Timeline
        st.markdown("---")
        st.subheader("📅 Timeline Analisis")
        filtered_df['tanggal'] = pd.to_datetime(filtered_df['tanggal_analisis']).dt.date
        timeline_data = filtered_df.groupby(['tanggal', 'kategori_stres']).size().reset_index(name='Jumlah')
        fig_timeline = px.line(
            timeline_data,
            x='tanggal',
            y='Jumlah',
            color='kategori_stres',
            title='Timeline Input Manual',
            markers=True,
            color_discrete_map={'🟢 Rendah': '#90EE90', '🟠 Sedang': '#FFD700', '🔴 Tinggi': '#FF6B6B'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True, key="chart_timeline_manual")
        
        # Detail per mahasiswa (expandable)
        st.markdown("---")
        st.subheader("👥 Detail Mahasiswa")
        
        for idx, row in filtered_df.iterrows():
            with st.expander(f"📄 {row['nama_mahasiswa']} - {row['kategori_stres']} (Analisis: {pd.to_datetime(row['tanggal_analisis']).strftime('%d/%m/%Y %H:%M')})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Data Pribadi**")
                    st.write(f"Nama: {row['nama_mahasiswa']}")
                    st.write(f"Gender: {row['gender']}")
                    st.write(f"Umur: {row['umur']} tahun")
                    st.write(f"IPK: {row['ipk']:.2f}")
                    st.write(f"Tahun: {row['tahun_studi']}")
                
                with col2:
                    st.markdown("**Status Gejala**")
                    st.write(f"Depresi: {row['depresi']}")
                    st.write(f"Kecemasan: {row['kecemasan']}")
                    st.write(f"Panic Attack: {row['panic_attack']}")
                    st.write(f"Konsultasi: {row['konsultasi_spesialis']}")
                
                with col3:
                    st.markdown("**Hasil Analisis**")
                    st.write(f"Cluster: {row['cluster']}")
                    st.write(f"Kategori: {row['kategori_stres']}")
                    st.write(f"Stress Score: {row['stress_score']:.2f}")
                    
                    # Mini gauge
                    mini_gauge = create_gauge_chart(row['stress_score'], "")
                    mini_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(mini_gauge, use_container_width=True, key=f"gauge_manual_{row['id']}")
        
        # Export
        st.markdown("---")
        csv_manual = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📤 Ekspor Data Input Manual ke CSV", csv_manual, "riwayat_input_manual.csv", "text/csv")
        
        # Hapus data
        st.markdown("---")
        st.subheader("🗑️ Manajemen Data")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning("⚠️ Hati-hati! Tindakan ini tidak dapat dibatalkan.")
        with col2:
            if st.button("🗑️ Hapus Semua Data", type="secondary"):
                c.execute("DELETE FROM input_manual")
                conn.commit()
                st.success("✅ Semua data input manual berhasil dihapus!")
                st.rerun()