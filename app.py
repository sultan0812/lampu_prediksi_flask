from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Tanpa GUI
import matplotlib.pyplot as plt
import numpy as np  # Tren linear

app = Flask(__name__)

DATA_FILE = 'data_permintaan.csv'
HISTORI_FILE = 'histori_prediksi.csv'
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
BULAN_ID = {
    "January": "Januari", "February": "Februari", "March": "Maret", "April": "April",
    "May": "Mei", "June": "Juni", "July": "Juli", "August": "Agustus",
    "September": "September", "October": "Oktober", "November": "November", "December": "Desember"
}

# ===============================
# Inject variabel global ke template
# ===============================
@app.context_processor
def inject_current_year():
    return {"current_year": datetime.now().year}

# ===============================
# Utility
# ===============================
def _safe_read_csv(path: str) -> pd.DataFrame:
    """Baca CSV data utama (Tahun/Bulan/LED/HID/Bohlam) dengan aman."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['Tahun', 'Bulan', 'LED', 'HID', 'Bohlam'])

    df = pd.read_csv(path, encoding='utf-8')
    df.columns = df.columns.str.strip()

    for col in ['Tahun', 'Bulan', 'LED', 'HID', 'Bohlam']:
        if col not in df.columns:
            df[col] = pd.Series(dtype='int64' if col != 'Bulan' else 'object')

    df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce').astype('Int64')
    df['Bulan'] = df['Bulan'].astype(str).str.strip()
    for col in ['LED', 'HID', 'Bohlam']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df = df.dropna(subset=['Tahun'])
    df['Tahun'] = df['Tahun'].astype(int)
    return df[['Tahun', 'Bulan', 'LED', 'HID', 'Bohlam']]


def _read_histori_csv() -> pd.DataFrame:
    """Baca histori (pertahankan kolom Tanggal/Waktu)."""
    if not os.path.exists(HISTORI_FILE):
        return pd.DataFrame(columns=['Tanggal', 'Waktu', 'Bulan', 'Tahun', 'LED', 'HID', 'Bohlam'])

    df = pd.read_csv(HISTORI_FILE, encoding='utf-8')
    df.columns = df.columns.str.strip()
    df['Bulan'] = df.get('Bulan', '').astype(str).str.strip()
    df['Tahun'] = pd.to_numeric(df.get('Tahun'), errors='coerce').astype('Int64')
    for col in ['LED', 'HID', 'Bohlam']:
        df[col] = pd.to_numeric(df.get(col), errors='coerce').fillna(0).astype(int)

    return df[['Tanggal', 'Waktu', 'Bulan', 'Tahun', 'LED', 'HID', 'Bohlam']]


def load_data() -> pd.DataFrame:
    return _safe_read_csv(DATA_FILE)

# ===============================
# Prediksi
# ===============================
def prediksi_rata_rata(bulan: str) -> dict:
    """Prediksi berdasarkan rata-rata (2019–2025)."""
    df = load_data()
    df_filtered = df[(df['Tahun'] >= 2019) & (df['Tahun'] <= 2025)]
    bulanan = df_filtered[df_filtered['Bulan'].str.strip() == bulan]

    if not bulanan.empty:
        return {
            'LED': int(bulanan['LED'].mean()),
            'HID': int(bulanan['HID'].mean()),
            'Bohlam': int(bulanan['Bohlam'].mean())
        }
    return {'LED': 0, 'HID': 0, 'Bohlam': 0}


def prediksi_tahun_sebelumnya(bulan: str, tahun: int | str):
    """Prediksi = salin nilai tahun sebelumnya (jika ada)."""
    df = load_data()
    try:
        tahun = int(tahun)
    except Exception:
        return None
    tahun_lalu = tahun - 1
    data = df[(df['Bulan'].str.strip() == bulan) & (df['Tahun'] == tahun_lalu)]
    if not data.empty:
        return {
            'LED': int(data['LED'].values[0]),
            'HID': int(data['HID'].values[0]),
            'Bohlam': int(data['Bohlam'].values[0])
        }
    return None


def prediksi_bulan_dengan_tren(bulan: str, target_year: int) -> dict:
    """Prediksi per-bulan dengan tren linear 2019–2025; fallback 0."""
    df = load_data()
    d = df[(df['Bulan'].str.strip() == bulan) & (df['Tahun'] >= 2019) & (df['Tahun'] <= 2025)]

    hasil = {}
    for kol in ['LED', 'HID', 'Bohlam']:
        if d.empty or d[kol].sum() == 0:
            pred = 0
        else:
            x = d['Tahun'].values.astype(float)
            y = d[kol].values.astype(float)
            if len(d) >= 2:
                m, b = np.polyfit(x, y, 1)  # y = m*x + b
                pred = m * float(target_year) + b
            else:
                pred = y.mean()
        hasil[kol] = max(int(round(pred)), 0)
    return hasil


def prediksi_tahunan(year: int) -> pd.DataFrame:
    """
    Prediksi tahunan:
    - kalau tahun <= 2025 coba pakai data tahun sebelumnya,
    - kalau tidak ada → tren linear per bulan,
    - jika tren 0 → rata-rata historis.
    """
    rows = []
    for b in MONTHS:
        hasil = None
        if year <= 2025:
            hasil_prev = prediksi_tahun_sebelumnya(b, year)
            if hasil_prev is not None:
                hasil = hasil_prev

        if hasil is None:
            hasil_tren = prediksi_bulan_dengan_tren(b, int(year))
            hasil = hasil_tren if sum(hasil_tren.values()) != 0 else prediksi_rata_rata(b)

        total = hasil['LED'] + hasil['HID'] + hasil['Bohlam']
        rows.append({
            'Bulan': b, 'Tahun': int(year),
            'LED': hasil['LED'], 'HID': hasil['HID'], 'Bohlam': hasil['Bohlam'],
            'Total': total
        })

    df_year = pd.DataFrame(rows)
    df_year['Bulan'] = pd.Categorical(df_year['Bulan'], categories=MONTHS, ordered=True)
    return df_year.sort_values(['Tahun', 'Bulan']).reset_index(drop=True)

# ===============================
# Simpan histori
# ===============================
def simpan_histori(bulan: str, tahun: int | str, hasil: dict):
    """Simpan hasil prediksi bulanan ke histori."""
    now = datetime.now()
    try:
        tahun_int = int(tahun)
    except Exception:
        tahun_int = None

    new_row = pd.DataFrame([{
        'Tanggal': now.strftime('%Y-%m-%d'),
        'Waktu': now.strftime('%H:%M:%S'),
        'Bulan': bulan, 'Tahun': tahun_int,
        'LED': hasil['LED'], 'HID': hasil['HID'], 'Bohlam': hasil['Bohlam']
    }])

    df_histori = _read_histori_csv()
    df_histori = pd.concat([df_histori, new_row], ignore_index=True)
    df_histori.to_csv(HISTORI_FILE, index=False, encoding='utf-8')


def simpan_histori_tahunan(df: pd.DataFrame):
    """Simpan hasil prediksi tahunan (12 baris) ke histori."""
    now = datetime.now()
    df = df.copy()
    df['Tanggal'] = now.strftime('%Y-%m-%d')
    df['Waktu'] = now.strftime('%H:%M:%S')
    df = df[['Tanggal', 'Waktu', 'Bulan', 'Tahun', 'LED', 'HID', 'Bohlam']]

    histori_df = _read_histori_csv()
    histori_df = pd.concat([histori_df, df], ignore_index=True)
    histori_df.to_csv(HISTORI_FILE, index=False, encoding='utf-8')

# ===============================
# Grafik
# ===============================
def buat_grafik_histori() -> str | None:
    df = _read_histori_csv()
    if df.empty:
        return None

    x = df['Bulan'].astype(str).str.strip() + ' ' + df['Tahun'].astype(str)
    plt.figure(figsize=(10, 5))
    for kolom in ['LED', 'HID', 'Bohlam']:
        plt.plot(x, df[kolom], marker='o', label=kolom)

    plt.xticks(rotation=45, ha='right')
    plt.title('Grafik Permintaan Histori')
    plt.xlabel('Bulan'); plt.ylabel('Permintaan')
    plt.tight_layout(); plt.legend()
    path = os.path.join(STATIC_DIR, 'grafik_permintaan.png')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    return path


def buat_grafik_tahunan(df_year: pd.DataFrame, year: int) -> str:
    plt.figure(figsize=(11, 5))
    x = df_year['Bulan'].astype(str)
    plt.plot(x, df_year['LED'], marker='o', label='LED')
    plt.plot(x, df_year['HID'], marker='o', label='HID')
    plt.plot(x, df_year['Bohlam'], marker='o', label='Bohlam')
    plt.xticks(rotation=30, ha='right')
    plt.title(f'Tren Permintaan {year} (Bulanan)')
    plt.xlabel('Bulan'); plt.ylabel('Jumlah Permintaan')
    plt.tight_layout(); plt.legend()
    path = os.path.join(STATIC_DIR, f'grafik_tahunan_{year}.png')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    return path


def buat_grafik_tren_tahunan(df: pd.DataFrame) -> str:
    path = os.path.join(STATIC_DIR, 'tren_permintaan_tahunan.png')
    plt.figure(figsize=(10, 5))
    if not df.empty:
        rekap = df.groupby('Tahun')[['LED', 'HID', 'Bohlam']].sum().reset_index()
        for kolom in ['LED', 'HID', 'Bohlam']:
            plt.plot(rekap['Tahun'], rekap[kolom], marker='o', label=kolom)
        plt.legend()
    plt.title('Tren Permintaan Lampu per Tahun')
    plt.xlabel('Tahun'); plt.ylabel('Jumlah Permintaan')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    return path

# ===============================
# Routes
# ===============================
@app.route('/')
def index():
    return render_template('index.html', datetime=datetime)

@app.route('/predict', methods=['POST'])
def predict():
    bulan = request.form.get('month')
    tahun = request.form.get('year')
    hasil = prediksi_tahun_sebelumnya(bulan, tahun) or prediksi_rata_rata(bulan)
    simpan_histori(bulan, tahun, hasil)
    grafik_file = buat_grafik_histori()
    label_bulan_tahun = f"{BULAN_ID.get(bulan, bulan)} {tahun}"

    return render_template(
        "index.html",
        prediction=hasil,
        selected_month=bulan,
        selected_year=tahun,
        grafik_file=grafik_file,
        label_bulan_tahun=label_bulan_tahun,
        show_result=True,
        datetime=datetime
    )

@app.route('/predict-year', methods=['POST'])
def predict_tahunan_route():
    year_text = request.form.get('year_predict')
    if not year_text:
        return redirect(url_for('index'))

    try:
        year = int(year_text)
    except Exception:
        return redirect(url_for('index'))

    df_year = prediksi_tahunan(year)
    simpan_histori_tahunan(df_year)
    chart_path = buat_grafik_tahunan(df_year, year)

    return render_template(
        'index.html',
        annual_result=df_year.to_dict(orient='records'),
        annual_year=year,
        annual_chart_file=chart_path,
        datetime=datetime
    )

@app.route('/rekap')
def rekap():
    # Gabungan data asli + histori untuk rekap tahunan
    df1 = load_data()
    df2 = _read_histori_csv()
    df = pd.concat([
        df1[['Tahun', 'Bulan', 'LED', 'HID', 'Bohlam']],
        df2[['Tahun', 'Bulan', 'LED', 'HID', 'Bohlam']]
    ], ignore_index=True)
    df.columns = df.columns.str.strip()

    rekap_df = []
    if not df.empty:
        tmp = df.groupby('Tahun')[['LED', 'HID', 'Bohlam']].sum().reset_index()
        tmp['Total'] = tmp[['LED', 'HID', 'Bohlam']].sum(axis=1)
        rekap_df = tmp.to_dict(orient='records')

    # Grafik tren tahunan
    _ = buat_grafik_tren_tahunan(df)

    # Histori: kelompokkan per tahun
    hist = _read_histori_csv()
    histori_grouped = []
    if not hist.empty:
        hist = hist.dropna(subset=['Tahun'])
        hist['Bulan'] = pd.Categorical(hist['Bulan'], categories=MONTHS, ordered=True)
        for th, dfg in hist.sort_values(['Tahun', 'Bulan']).groupby('Tahun'):
            histori_grouped.append({
                "tahun": int(th),
                "rows": dfg[['Tanggal', 'Waktu', 'Bulan', 'Tahun', 'LED', 'HID', 'Bohlam']].to_dict(orient='records')
            })

    return render_template(
        'rekap.html',
        rekap_tahunan=rekap_df,
        histori_per_tahun=histori_grouped
    )

@app.route('/view/<int:tahun>')
def view_tahun(tahun):
    """Lihat data per tahun. Jika data asli kosong → tampilkan prediksi tahun tsb."""
    df = load_data()
    df_tahun = df[df['Tahun'] == tahun]
    is_prediksi = False
    if df_tahun.empty:
        df_pred = prediksi_tahunan(tahun)
        df_tahun = df_pred[['Bulan', 'LED', 'HID', 'Bohlam']].copy()
        is_prediksi = True

    return render_template(
        'view_tahun.html',
        tahun=tahun,
        data=df_tahun.to_dict(orient='records'),
        is_prediksi=is_prediksi
    )

@app.route('/hapus/<int:tahun>', methods=['POST'])
def hapus_tahun(tahun):
    try:
        if os.path.exists(HISTORI_FILE):
            df_histori = _read_histori_csv()
            df_histori = df_histori[df_histori['Tahun'] != tahun]
            df_histori.to_csv(HISTORI_FILE, index=False, encoding='utf-8')

        if os.path.exists(DATA_FILE):
            df_data = _safe_read_csv(DATA_FILE)
            df_data = df_data[df_data['Tahun'] != tahun]
            df_data.to_csv(DATA_FILE, index=False, encoding='utf-8')

        # Hapus grafik tahun itu jika ada
        grafik_file = os.path.join(STATIC_DIR, f'grafik_tahunan_{tahun}.png')
        if os.path.exists(grafik_file):
            os.remove(grafik_file)

    except Exception as e:
        print(f"[ERROR HAPUS]: {e}")

    return redirect(url_for('rekap'))

@app.route('/unduh')
def unduh():
    if os.path.exists(HISTORI_FILE):
        return send_file(HISTORI_FILE, as_attachment=True)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
