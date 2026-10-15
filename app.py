"""
Dashboard Peramalan Harga Kontrak Berjangka Kopi Arabika dan Robusta
=====================================================================
LSTM berbasis Attention Mechanism -- Langkah Analisis 9 (Subbab 3.4 proposal)

Struktur folder yang dibutuhkan:
    outputs/
        arabika/
            model_lstm_attention.keras      model refit pada seluruh data (2009 s.d. 27 Juli 2026),
                                            dipakai untuk peramalan (Langkah 8)
            scaler_params.json              ret_scale, vol_mean, vol_std milik model refit
            best_hyperparameters.json       termasuk final_epochs (22 arabika, 17 robusta)
            metrics_overall.json            MAE, RMSE, MAPE model evaluasi pada data testing (Tabel 4.6)
            metrics_per_horizon.csv         MAPE per langkah horizon (Gambar 4.5)
            forecast_next_30_days.csv       hasil peramalan run v6 (Tabel 4.12)
            forecast_next_30_days_interval.csv   (opsional, pita interval 95% run v6)
            historical_clean.csv
        robusta/
            (file yang sama)

Jalankan lokal dengan:  streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.models import load_model

# ============================================================
# Konstanta (harus identik dengan notebook Bab 4)
# ============================================================
FEATURES = ['Close', 'Open', 'High', 'Low', 'Volume']
PRICE_COLS = ['Close', 'Open', 'High', 'Low']
HORIZON = 30
OUTPUT_DIR = 'outputs'
SEED = 42
HOLDOUT_TEST_DAYS = 90      # Subbab 3.4 poin 9d
KONTRAK = ['Arabika', 'Robusta']   # urutan sama dengan laporan
SATUAN = {'Arabika': 'sen AS/pon', 'Robusta': 'USD/metrik ton'}
LINK_INVESTING = {'Arabika': 'https://www.investing.com/commodities/us-coffee-c-historical-data',
                  'Robusta': 'https://www.investing.com/commodities/london-coffee-historical-data'}
BULAN = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']

st.set_page_config(page_title='Dashboard Peramalan Harga Kopi', page_icon='☕', layout='wide')


def fmt(x, dec=2):
    """Format angka Indonesia: 3.799,00"""
    return f'{x:,.{dec}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def fmt_tgl(t):
    t = pd.Timestamp(t)
    return f'{t.day} {BULAN[t.month - 1]} {t.year}'


# ============================================================
# Arsitektur model -- WAJIB identik dengan notebook, termasuk
# custom layer, supaya load_model() bisa membaca file .keras
# ============================================================
class ScaledDotProductAttention(layers.Layer):
    """Persamaan 2.12-2.13: Q=K=V=H (seluruh hidden state LSTM), tanpa
    transformasi linear tambahan."""

    def call(self, H):
        d_k = tf.cast(tf.shape(H)[-1], tf.float32)
        scores = tf.matmul(H, H, transpose_b=True) / tf.sqrt(d_k)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, H)
        return context, weights

    def get_config(self):
        return super().get_config()


def build_model(window_size, n_features, horizon, lstm_units=64, hidden_layers=1,
                dropout=0.2, learning_rate=0.001):
    """Sama persis dengan build_model() di notebook (aggregation='pool')."""
    inputs = layers.Input(shape=(window_size, n_features), name='input_ohlcv')
    x = inputs
    for i in range(hidden_layers):
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout,
                        name=f'lstm_{i + 1}')(x)
    context, attn_weights = ScaledDotProductAttention(
        name='scaled_dot_product_attention')(x)
    pooled = layers.GlobalAveragePooling1D(name='context_vector')(context)
    outputs = layers.Dense(horizon, name='dense_output')(pooled)

    model = Model(inputs, outputs, name='LSTM_Attention')
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# ============================================================
# Representasi relatif -- WAJIB identik dengan notebook Bagian 5
#   OHLC   -> ln(P / Close terakhir di window) / ret_scale
#   Volume -> (log1p(Volume) - vol_mean) / vol_std
#   Target -> ln(Close_{t+h} / Close_t) / ret_scale
#   Invers -> Close_t * exp(y_hat * ret_scale)
# ============================================================
def normalize_fit(train_df, horizon=HORIZON):
    """Parameter skala dihitung HANYA dari data training."""
    close = train_df['Close'].values.astype(float)
    log_ret = np.concatenate([np.log(close[h:] / close[:-h]) for h in range(1, horizon + 1)])
    vol = np.log1p(train_df['Volume'].values.astype(float))
    return {'ret_scale': float(log_ret.std()),
            'vol_mean': float(vol.mean()), 'vol_std': float(vol.std() + 1e-9)}


def make_input_window(window_df, scaler):
    prices = window_df[PRICE_COLS].values.astype(float)
    last_close = float(window_df['Close'].values[-1])
    px = np.log(prices / last_close) / scaler['ret_scale']
    vol = (np.log1p(window_df['Volume'].values.astype(float)) - scaler['vol_mean']) / scaler['vol_std']
    return np.hstack([px, vol[:, None]]).astype('float32'), last_close


def build_windows(df, scaler, window_size, horizon=HORIZON):
    df = df.reset_index(drop=True)
    close = df['Close'].values.astype(float)
    X, Y, last = [], [], []
    for i in range(max(len(df) - window_size - horizon + 1, 0)):
        x, c_t = make_input_window(df.iloc[i:i + window_size], scaler)
        X.append(x)
        Y.append(np.log(close[i + window_size: i + window_size + horizon] / c_t) / scaler['ret_scale'])
        last.append(c_t)
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32'), np.array(last)


def denormalize_target(y_norm, last_close, scaler):
    return np.asarray(last_close).reshape(-1, 1) * np.exp(np.asarray(y_norm) * scaler['ret_scale'])


# ============================================================
# Memuat artefak (model pretrained, scaler, hyperparameter, data)
# ============================================================
def _read_forecast(path):
    """Kolom peramalan di notebook bernama 'Forecast_Close', disamakan menjadi 'Forecast'."""
    df = pd.read_csv(path, parse_dates=['Date'])
    if 'Forecast' not in df.columns and 'Forecast_Close' in df.columns:
        df = df.rename(columns={'Forecast_Close': 'Forecast'})
    return df


@st.cache_resource
def load_artifacts(label):
    d = os.path.join(OUTPUT_DIR, label.lower())

    model = load_model(
        os.path.join(d, 'model_lstm_attention.keras'),
        custom_objects={'ScaledDotProductAttention': ScaledDotProductAttention},
    )
    with open(os.path.join(d, 'scaler_params.json')) as f:
        scaler = json.load(f)
    if 'ret_scale' not in scaler:
        raise ValueError('scaler_params.json masih format lama (Min-Max). Gunakan artefak '
                         'dari notebook v6 (representasi relatif).')

    with open(os.path.join(d, 'best_hyperparameters.json')) as f:
        best_hp = json.load(f)
    with open(os.path.join(d, 'metrics_overall.json')) as f:
        metrics_overall = json.load(f)
    metrics_per_horizon = pd.read_csv(os.path.join(d, 'metrics_per_horizon.csv'))

    forecast_df = _read_forecast(os.path.join(d, 'forecast_next_30_days.csv'))
    hist_df = pd.read_csv(os.path.join(d, 'historical_clean.csv'), parse_dates=['Date'])

    # Interval prediksi 95% (opsional, hasil Bagian 11 notebook). Disimpan sebagai rasio
    # batas terhadap nilai peramalan per hari ke-h, sehingga bisa dipakai ulang.
    interval_ratio = None
    ipath = os.path.join(d, 'forecast_next_30_days_interval.csv')
    if os.path.exists(ipath):
        idf = _read_forecast(ipath)
        interval_ratio = {'lo': (idf['Lower_95'] / idf['Forecast']).values,
                          'hi': (idf['Upper_95'] / idf['Forecast']).values}

    return {
        'model': model, 'scaler': scaler, 'best_hp': best_hp,
        'metrics_overall': metrics_overall, 'metrics_per_horizon': metrics_per_horizon,
        'forecast_df': forecast_df, 'hist_df': hist_df, 'interval_ratio': interval_ratio,
    }


def forecast_future(hist_df, model, scaler, window_size, horizon=HORIZON):
    """Pakai window_size hari terakhir sebagai input, model TIDAK dilatih ulang di sini."""
    x_last, last_close = make_input_window(hist_df.tail(window_size), scaler)
    y_norm = model.predict(x_last[np.newaxis, :, :], verbose=0)
    y_price = denormalize_target(y_norm, [last_close], scaler)[0][:horizon]
    last_date = hist_df['Date'].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({'Date': future_dates, 'Forecast': y_price})


def _train(df, best_hp, scaler=None):
    """Melatih model dengan hyperparameter dan jumlah epoch tetap (sama dengan Langkah 7f),
    tanpa early stopping, sehingga tidak ada data yang dipakai untuk menentukan kapan berhenti."""
    window_size = best_hp['sliding_window']
    scaler = scaler or normalize_fit(df)
    X, Y, _ = build_windows(df, scaler, window_size, HORIZON)
    tf.keras.utils.set_random_seed(SEED)
    model = build_model(window_size, len(FEATURES), HORIZON,
                        lstm_units=best_hp['lstm_units'], hidden_layers=best_hp['hidden_layers'],
                        dropout=best_hp['dropout'], learning_rate=best_hp['learning_rate'])
    model.fit(X, Y, epochs=int(best_hp['final_epochs']), batch_size=best_hp['batch_size'],
              shuffle=True, verbose=0)
    return model, scaler


def retrain_with_holdout(hist_df, best_hp, date_start, date_end, holdout_days=HOLDOUT_TEST_DAYS,
                         progress=None):
    """Subbab 3.4 poin 9d, mengikuti alur Bab 4:
    1) model dilatih pada rentang terpilih kecuali holdout_days terakhir, lalu dievaluasi
       pada holdout_days terakhir (MAE, RMSE, MAPE)
    2) model dilatih ulang pada seluruh rentang terpilih untuk menghasilkan peramalan."""
    window_size = best_hp['sliding_window']
    df_range = hist_df[(hist_df['Date'] >= pd.Timestamp(date_start)) &
                       (hist_df['Date'] <= pd.Timestamp(date_end))].reset_index(drop=True)

    min_len = window_size + HORIZON + holdout_days + 60
    if len(df_range) < min_len:
        raise ValueError(f'Rentang tanggal terlalu pendek. Minimal sekitar {min_len} hari '
                         f'perdagangan, rentang yang dipilih hanya {len(df_range)} hari.')

    train_df = df_range.iloc[:-holdout_days].reset_index(drop=True)
    test_df = df_range.iloc[-holdout_days:].reset_index(drop=True)

    # Tahap 1: evaluasi pada holdout
    if progress: progress(0.05, 'Tahap 1 dari 2: melatih model untuk evaluasi holdout...')
    model_eval, scaler = _train(train_df, best_hp)
    combined = pd.concat([train_df.tail(window_size), test_df], ignore_index=True)
    X_test, Y_test, last_test = build_windows(combined, scaler, window_size, HORIZON)
    pred_price = denormalize_target(model_eval.predict(X_test, verbose=0), last_test, scaler)
    actual_price = denormalize_target(Y_test, last_test, scaler)
    mae = float(np.mean(np.abs(actual_price - pred_price)))
    rmse = float(np.sqrt(np.mean((actual_price - pred_price) ** 2)))
    mape = float(np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100)

    # Tahap 2: latih ulang pada seluruh rentang terpilih untuk peramalan
    if progress: progress(0.55, 'Tahap 2 dari 2: melatih ulang pada seluruh rentang untuk peramalan...')
    model_full, scaler_full = _train(df_range, best_hp)
    forecast_retrain = forecast_future(df_range, model_full, scaler_full, window_size)
    if progress: progress(1.0, 'Selesai.')

    return {'metrics': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape},
            'n_train': len(train_df), 'n_test_holdout': len(test_df),
            'n_full': len(df_range), 'date_end': df_range['Date'].max(),
            'forecast_df': forecast_retrain}


def _to_number(series):
    """Mengubah teks angka Investing.com ('3,405.00', '15.19K', '1.2M') menjadi float."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.replace(',', '', regex=False).str.strip()
    mult = s.str[-1].map({'K': 1e3, 'M': 1e6, 'B': 1e9}).fillna(1.0)
    s = s.where(~s.str[-1].isin(['K', 'M', 'B']), s.str[:-1])
    return pd.to_numeric(s.replace({'-': np.nan, '': np.nan}), errors='coerce') * mult


def standardize_upload(df):
    """Menerima format standar (Date, Open, High, Low, Close, Volume) maupun format unduhan
    Investing.com (Date, Price, Open, High, Low, Vol., Change %)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={'Price': 'Close', 'Vol.': 'Volume'})
    if 'Date' in df.columns:
        raw = df['Date'].astype(str)
        parsed = pd.to_datetime(raw, format='%m/%d/%Y', errors='coerce')
        if parsed.isna().any():
            parsed = pd.to_datetime(raw, errors='coerce')
        df['Date'] = parsed
    for c in FEATURES:
        if c in df.columns:
            df[c] = _to_number(df[c])
    return df


def merge_new_data(existing_df, uploaded_df):
    """Menambahkan baris tanggal baru dari file unggahan. Tanggal yang tumpang tindih
    tetap memakai nilai lama."""
    uploaded_df = standardize_upload(uploaded_df)
    required_cols = ['Date'] + FEATURES
    missing = [c for c in required_cols if c not in uploaded_df.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di file: {', '.join(missing)}")
    uploaded_df = uploaded_df[required_cols].dropna(subset=['Date'])
    if (uploaded_df[PRICE_COLS] <= 0).any().any():
        raise ValueError('Harga (Open, High, Low, Close) harus bernilai positif.')

    existing_dates = set(existing_df['Date'])
    new_rows = uploaded_df[~uploaded_df['Date'].isin(existing_dates)]
    n_overlap = len(uploaded_df) - len(new_rows)

    combined = pd.concat([existing_df, new_rows], ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    combined[FEATURES] = combined[FEATURES].ffill()   # Subbab 2.2.1: forward fill
    return combined, len(new_rows), n_overlap


# ============================================================
# Gaya tampilan
# ============================================================
WARNA = {'historis': '#2a78d6', 'peramalan': '#eb6834', 'retrain': '#1baf7a',
         'awal': '#8a8984', 'grid': 'rgba(0,0,0,0.06)'}
st.markdown("""
<style>
.block-container {padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1250px;}
.judul-kecil {font-size: 0.85rem; color: #6b6a66; margin-bottom: -0.4rem;}
.kpi-box {min-height: 118px;}
.kpi-label {font-size: 0.85rem; color: #6b6a66; margin-bottom: 0.15rem;}
.kpi-nilai {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.kpi-sub {font-size: 0.82rem; color: #6b6a66; margin-top: 0.2rem;}
.naik {color: #1a7f37; font-weight: 600;} .turun {color: #c0392b; font-weight: 600;}
.datar {color: #6b6a66; font-weight: 600;}
.lencana {display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
          background: #e8f5ec; color: #1a7f37; font-size: 0.78rem; font-weight: 600;}
section[data-testid="stSidebar"] .stMarkdown p {font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


def kartu(label, nilai, sub=''):
    with st.container(border=True):
        st.markdown(f'<div class="kpi-box"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-nilai">{nilai}</div><div class="kpi-sub">{sub}</div></div>',
                    unsafe_allow_html=True)


def arah(chg):
    if chg > 1:
        return f'<span class="naik">▲ naik {fmt(chg)}%</span>'
    if chg < -1:
        return f'<span class="turun">▼ turun {fmt(abs(chg))}%</span>'
    return f'<span class="datar">● relatif mendatar ({fmt(chg)}%)</span>'


def sumbu_tanggal(fig, start, end):
    """Label bulan berbahasa Indonesia pada sumbu x."""
    months = pd.date_range(pd.Timestamp(start).to_period('M').to_timestamp(), end, freq='MS')
    step = max(1, int(np.ceil(len(months) / 10)))
    months = months[::step]
    fig.update_xaxes(tickvals=months, ticktext=[f'{BULAN[m.month - 1]} {m.year}' for m in months])


def gaya_grafik(fig, unit, tinggi=460):
    fig.update_layout(template='plotly_white', height=tinggi, separators=',.',
                      margin=dict(l=10, r=10, t=30, b=10), hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                      yaxis_title=unit, font=dict(size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=WARNA['grid'], tickformat=',.0f')
    return fig


def garis(x, y, nama, warna, dash=None, marker=False, lebar=2):
    return go.Scatter(x=x, y=y, name=nama, mode='lines+markers' if marker else 'lines',
                      line=dict(color=warna, width=lebar, dash=dash), marker=dict(size=5),
                      customdata=[fmt_tgl(d) for d in x],
                      hovertemplate=f'{nama}: %{{y:,.2f}}<extra></extra>')


KONFIG_GRAFIK = {'displaylogo': False,
                 'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines']}

for key in ('extended_hist', 'retrain_result'):
    if key not in st.session_state:
        st.session_state[key] = {}


# ============================================================
# Header dan pilihan utama
# ============================================================
st.markdown('<div class="judul-kecil">Dashboard Peramalan · LSTM berbasis Attention Mechanism</div>',
            unsafe_allow_html=True)
st.title('Peramalan Harga Kontrak Berjangka Kopi')

with st.expander('Cara menggunakan dashboard'):
    st.markdown("""
**Melihat hasil peramalan**
1. Pilih kontrak **Arabika** atau **Robusta** di bawah judul.
2. Geser *slider* horizon untuk menentukan berapa hari perdagangan ke depan yang ingin dilihat (1 sampai 30 hari).
3. Baca empat kartu ringkasan: harga terakhir, hasil peramalan dan arahnya, kisaran harga yang mungkin terjadi, serta akurasi model.
4. Lihat grafik. Garis biru adalah harga aktual, garis oranye putus-putus adalah hasil peramalan, dan area oranye muda adalah kisaran harga dengan tingkat keyakinan 95%. Pilih periode di atas grafik untuk memperpendek atau memperpanjang data historis yang ditampilkan.
5. Buka tab di bawah grafik untuk melihat tabel peramalan (bisa diunduh), grafik *candlestick*, kinerja model, dan penjelasan model.

**Menambahkan data terbaru** (sidebar, langkah ①)
1. Klik tautan Investing.com di sidebar, atur rentang tanggal mulai dari 28 Juli 2026, lalu unduh data harian dalam format CSV.
2. Unggah berkas tersebut. Peramalan otomatis dihitung ulang dari tanggal data terakhir, dan muncul kotak yang membandingkan peramalan awal dengan harga aktual.
3. Klik **Hapus data unggahan** untuk kembali ke data awal.

**Melatih ulang model** (sidebar, langkah ②)
1. Pilih rentang tanggal data, minimal 1 tahun. Perkiraan waktu proses ditampilkan di bawahnya.
2. Klik **Latih Ulang Model** dan tunggu hingga selesai.
3. Hasilnya muncul sebagai garis hijau pada grafik dan sebagai perbandingan metrik pada tab Kinerja model.

Klik **›** di pojok kiri atas jika sidebar tidak terlihat.
""")

c_k, c_h = st.columns([1, 2], gap='large')
with c_k:
    try:
        kontrak = st.segmented_control('Kontrak', KONTRAK, default='Arabika', key='kontrak')
    except AttributeError:
        kontrak = st.radio('Kontrak', KONTRAK, horizontal=True, key='kontrak')
    kontrak = kontrak or 'Arabika'
with c_h:
    horizon_pilihan = st.slider('Horizon peramalan (hari perdagangan ke depan)', 1, HORIZON, HORIZON)

with st.spinner('Memuat model dan data, mohon tunggu sebentar...'):
    art = load_artifacts(kontrak)
unit = SATUAN[kontrak]
nama_kontrak = {'Arabika': 'Coffee C Futures (ICE New York)',
                'Robusta': 'Robusta Coffee Futures (ICE London)'}[kontrak]
window_size = art['best_hp']['sliding_window']
hist_df = st.session_state['extended_hist'].get(kontrak, art['hist_df'])


# ============================================================
# Sidebar: kelola data dan model (fitur lanjutan)
# ============================================================
st.sidebar.header('Kelola Data dan Model')
st.sidebar.caption(f'Pengaturan berlaku untuk kontrak **{kontrak}**.')

with st.sidebar.expander('① Tambah data terbaru', expanded=True):
    st.caption('Unduh data harian terbaru dari Investing.com, lalu unggah file CSV-nya di sini. '
               'Hanya tanggal yang belum ada yang ditambahkan.')
    st.markdown(f'[Buka halaman data historis {kontrak} di Investing.com ↗]({LINK_INVESTING[kontrak]})')
    contoh = pd.DataFrame({'Date': ['08/03/2026', '07/31/2026'],
                           'Price': ['3,784.00', '3,775.00'], 'Open': ['3,771.00', '3,755.00'],
                           'High': ['3,828.00', '3,791.00'], 'Low': ['3,765.00', '3,744.00'],
                           'Vol.': ['9.26K', '8.59K'], 'Change %': ['0.24%', '0.37%']})
    st.download_button('Unduh contoh format CSV', contoh.to_csv(index=False).encode('utf-8'),
                       file_name='contoh_format_investing.csv', mime='text/csv')
    uploaded_file = st.file_uploader('File CSV', type=['csv'], key=f'up_{kontrak}',
                                     label_visibility='collapsed')
    status_key = f'upload_status_{kontrak}'
    if uploaded_file is not None and st.session_state.get(f'upload_id_{kontrak}') != uploaded_file.file_id:
        st.session_state[f'upload_id_{kontrak}'] = uploaded_file.file_id
        try:
            raw = standardize_upload(pd.read_csv(uploaded_file))
            # cek salah kontrak: level harga harus sebanding dengan data historis terakhir
            ref = art['hist_df']['Close'].tail(60).median()
            new_med = raw['Close'].median()
            if not (0.5 <= new_med / ref <= 2):
                lain = [k for k in KONTRAK if k != kontrak][0]
                raise ValueError(f'level harga di file (sekitar {fmt(new_med, 0)}) jauh berbeda '
                                 f'dengan harga {kontrak} (sekitar {fmt(ref, 0)}). File ini '
                                 f'kemungkinan data {lain}.')
            merged, n_new, n_overlap = merge_new_data(art['hist_df'], raw)
            pesan = [('success', f'{n_new} hari perdagangan baru ditambahkan.' +
                      (f' {n_overlap} tanggal yang sudah ada diabaikan.' if n_overlap else ''))]
            if n_new == 0:
                pesan = [('info', 'Tidak ada tanggal baru. Semua tanggal di file sudah ada di data.')]
            else:
                st.session_state['extended_hist'][kontrak] = merged
                baru = merged[merged['Date'] > art['hist_df']['Date'].max()]
                jeda = (baru['Date'].min() - art['hist_df']['Date'].max()).days if len(baru) else 0
                if jeda > 10:
                    pesan.append(('warning', f'Ada jeda {jeda} hari kalender antara data lama dan data '
                                  'baru. Peramalan tetap dihitung, tetapi sebaiknya lengkapi tanggal '
                                  'yang hilang agar urutan data tidak terputus.'))
            st.session_state[status_key] = pesan
        except (ValueError, KeyError) as e:
            st.session_state[status_key] = [('error', f'Gagal memproses file: {e}')]
    for jenis, teks in st.session_state.get(status_key, []):
        getattr(st, jenis)(teks)
    if kontrak in st.session_state['extended_hist']:
        if st.button('Hapus data unggahan', key=f'hapus_{kontrak}'):
            st.session_state['extended_hist'].pop(kontrak)
            st.session_state.pop(status_key, None)
            st.rerun()
    hist_df = st.session_state['extended_hist'].get(kontrak, art['hist_df'])

with st.sidebar.expander('② Latih ulang model'):
    st.caption(f"Model dilatih ulang dengan hyperparameter tetap ({art['best_hp']['final_epochs']} "
               f'epoch). {HOLDOUT_TEST_DAYS} hari perdagangan terakhir dipakai untuk menguji, lalu '
               'model dilatih ulang pada seluruh rentang untuk meramal. Rentang minimal 1 tahun. '
               'Rentang yang panjang membutuhkan waktu lebih lama.')
    min_date, max_date = hist_df['Date'].min().date(), hist_df['Date'].max().date()
    default_start = max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(years=3)).date())
    date_range = st.date_input('Rentang data', value=(default_start, max_date),
                               min_value=min_date, max_value=max_date, format='DD/MM/YYYY')
    if len(date_range) == 2:
        n_hari = int(((hist_df['Date'] >= pd.Timestamp(date_range[0])) &
                      (hist_df['Date'] <= pd.Timestamp(date_range[1]))).sum())
        n_batch = max(n_hari - window_size - HORIZON, 0) / art['best_hp']['batch_size']
        detik_per_batch = 0.22 if art['best_hp']['lstm_units'] > 150 else 0.13
        menit = max(1, round(2 * art['best_hp']['final_epochs'] * n_batch * detik_per_batch / 60))
        st.caption(f'{fmt(n_hari, 0)} hari perdagangan terpilih. Perkiraan waktu sekitar {menit} menit '
                   '(tergantung kecepatan komputer).')
    if st.button('Latih Ulang Model', type='primary'):
        if len(date_range) != 2:
            st.error('Pilih tanggal mulai dan tanggal akhir.')
        elif (pd.Timestamp(date_range[1]) - pd.Timestamp(date_range[0])).days < 365:
            st.error('Rentang tanggal minimal 1 tahun.')
        else:
            bar = st.progress(0.0, text='Menyiapkan data...')
            try:
                st.session_state['retrain_result'][kontrak] = retrain_with_holdout(
                    hist_df, art['best_hp'], date_range[0], date_range[1],
                    progress=lambda v, t: bar.progress(v, text=t))
                st.success('Selesai. Lihat garis hijau pada grafik dan tab Kinerja Model.')
            except ValueError as e:
                bar.empty()
                st.error(str(e))
    if kontrak in st.session_state['retrain_result']:
        if st.button('Hapus hasil latih ulang'):
            st.session_state['retrain_result'].pop(kontrak)
            st.rerun()

st.sidebar.markdown('---')
st.sidebar.caption('Data: Investing.com, 2 Jan 2009 sampai 27 Jul 2026. '
                   'Hasil peramalan merupakan estimasi berdasarkan pola historis, '
                   'bukan kepastian harga.')


# ============================================================
# Perhitungan peramalan yang ditampilkan
# ============================================================
extended = kontrak in st.session_state['extended_hist']
has_retrain = kontrak in st.session_state['retrain_result']
forecast_awal = art['forecast_df'][['Date', 'Forecast']]
forecast_main = (forecast_future(hist_df, art['model'], art['scaler'], window_size)
                 if extended else forecast_awal)
fc = forecast_main.head(horizon_pilihan).copy()
ada_interval = art['interval_ratio'] is not None
if ada_interval:
    fc['Lower_95'] = fc['Forecast'].values * art['interval_ratio']['lo'][:horizon_pilihan]
    fc['Upper_95'] = fc['Forecast'].values * art['interval_ratio']['hi'][:horizon_pilihan]

last_price = hist_df['Close'].iloc[-1]
last_date = hist_df['Date'].max()
end_fc = fc['Forecast'].iloc[-1]
chg = (end_fc - last_price) / last_price * 100
mape = art['metrics_overall']['MAPE']

st.caption(f'{kontrak}: {nama_kontrak} · satuan {unit} · data sampai {fmt_tgl(last_date)}'
           + (' (termasuk data unggahan)' if extended else ''))


# ============================================================
# Ringkasan (KPI)
# ============================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    kartu('Harga penutupan terakhir', fmt(last_price), fmt_tgl(last_date))
with k2:
    kartu(f'Peramalan hari ke-{horizon_pilihan}', fmt(end_fc),
          f"{fmt_tgl(fc['Date'].iloc[-1])} · {arah(chg)}")
with k3:
    if ada_interval:
        kartu(f'Kisaran 95% hari ke-{horizon_pilihan}',
              f"{fmt(fc['Lower_95'].iloc[-1], 0)}–{fmt(fc['Upper_95'].iloc[-1], 0)}",
              'harga aktual diperkirakan berada di rentang ini')
    else:
        lo, hi = fc['Forecast'].min(), fc['Forecast'].max()
        kartu(f'Rentang peramalan 1–{horizon_pilihan} hari', f'{fmt(lo, 0)}–{fmt(hi, 0)}',
              'nilai terendah dan tertinggi hasil peramalan')
with k4:
    kategori = 'Sangat baik' if mape < 10 else ('Baik' if mape < 20 else 'Cukup')
    kartu('Akurasi model (MAPE)', f'{fmt(mape)}%',
          f'<span class="lencana">{kategori}</span> data testing 2024 sampai Jul 2026')


# ============================================================
# Grafik utama
# ============================================================
if extended:
    aktual_baru = hist_df[hist_df['Date'] > art['hist_df']['Date'].max()].reset_index(drop=True)
    n_cek = min(len(aktual_baru), len(art['forecast_df']))
    if n_cek >= 1:
        a = aktual_baru['Close'].values[:n_cek]
        f0 = art['forecast_df']['Forecast'].values[:n_cek]
        mape_cek = float(np.mean(np.abs(f0 - a) / a) * 100)
        teks = (f'**Uji ramalan awal terhadap harga aktual.** Ramalan dari data sampai 27 Jul 2026 '
                f'dibandingkan dengan {n_cek} hari perdagangan aktual yang diunggah: MAPE '
                f'**{fmt(mape_cek)}%**')
        if ada_interval:
            lo0 = f0 * art['interval_ratio']['lo'][:n_cek]
            hi0 = f0 * art['interval_ratio']['hi'][:n_cek]
            masuk = int(((a >= lo0) & (a <= hi0)).sum())
            teks += f', dan **{masuk} dari {n_cek} hari** berada di dalam interval prediksi 95%'
        arah_f = 'naik' if f0[-1] > art['hist_df']['Close'].iloc[-1] else 'turun'
        arah_a = 'naik' if a[-1] > art['hist_df']['Close'].iloc[-1] else 'turun'
        teks += (f'. Arah pergerakan diramalkan {arah_f}, harga aktual {arah_a}'
                 + (' (sesuai).' if arah_f == arah_a else ' (tidak sesuai).'))
        st.info(teks)

st.subheader('Harga historis dan hasil peramalan')
periode = st.radio('Tampilkan data historis', ['3 bulan', '6 bulan', '1 tahun', '3 tahun', 'Semua'],
                   index=1, horizontal=True, label_visibility='collapsed')
mundur = {'3 bulan': 92, '6 bulan': 183, '1 tahun': 366, '3 tahun': 1096, 'Semua': 100000}[periode]
hist_view = hist_df[hist_df['Date'] >= last_date - pd.Timedelta(days=mundur)]

fig = go.Figure()
if ada_interval:
    fig.add_trace(go.Scatter(x=fc['Date'], y=fc['Upper_95'], mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fc['Date'], y=fc['Lower_95'], mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(235,104,52,0.15)',
                             name='Interval prediksi 95%', hoverinfo='skip'))
fig.add_trace(garis(hist_view['Date'], hist_view['Close'], 'Harga aktual', WARNA['historis']))
if extended:
    fa = forecast_awal.head(horizon_pilihan)
    fig.add_trace(garis(fa['Date'], fa['Forecast'], 'Peramalan awal (27 Jul 2026)',
                        WARNA['awal'], dash='dot'))
# sambungkan garis peramalan ke harga terakhir agar tidak terputus
fx = pd.concat([pd.Series([last_date]), fc['Date']], ignore_index=True)
fy = pd.concat([pd.Series([last_price]), fc['Forecast']], ignore_index=True)
fig.add_trace(garis(fx, fy, 'Peramalan', WARNA['peramalan'], dash='dash', marker=True))
if has_retrain:
    rr_fc = st.session_state['retrain_result'][kontrak]['forecast_df'].head(horizon_pilihan)
    fig.add_trace(garis(rr_fc['Date'], rr_fc['Forecast'], 'Peramalan model latih ulang',
                        WARNA['retrain'], dash='dot', marker=True))
fig.add_vline(x=last_date, line_width=1, line_dash='dot', line_color='#8a8984')
fig.add_annotation(x=last_date, y=1, yref='paper', text='mulai peramalan', showarrow=False,
                   xanchor='right', yanchor='bottom', font=dict(size=11, color='#6b6a66'))
gaya_grafik(fig, unit)
sumbu_tanggal(fig, hist_view['Date'].min(), fc['Date'].max())
st.plotly_chart(fig, width='stretch', config=KONFIG_GRAFIK)

catatan = ['Garis putus-putus oranye adalah hasil peramalan. Semakin jauh harinya, semakin besar '
           'ketidakpastiannya.']
if ada_interval:
    catatan.append('Area oranye muda adalah interval prediksi 95%.')
if extended:
    catatan.append('Garis abu-abu titik adalah peramalan awal dari data sampai 27 Juli 2026, '
                   'untuk dibandingkan dengan harga aktual yang diunggah.')
st.caption(' '.join(catatan))


# ============================================================
# Detail dalam tab
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(['Tabel peramalan', 'Data historis', 'Kinerja model', 'Tentang model'])

with tab1:
    tbl = fc.copy()
    tbl.insert(0, 'Hari ke-', range(1, len(tbl) + 1))
    tbl['Perubahan dari harga terakhir'] = ((tbl['Forecast'] / last_price - 1) * 100).apply(
        lambda v: f'{fmt(v)}%')
    tbl['Date'] = tbl['Date'].apply(fmt_tgl)
    for c in ['Forecast', 'Lower_95', 'Upper_95']:
        if c in tbl:
            tbl[c] = tbl[c].apply(fmt)
    tbl = tbl.rename(columns={'Date': 'Tanggal', 'Forecast': f'Peramalan ({unit})',
                              'Lower_95': 'Batas bawah 95%', 'Upper_95': 'Batas atas 95%'})
    st.dataframe(tbl, width='stretch', hide_index=True)
    unduh = fc.copy()
    unduh['Date'] = unduh['Date'].dt.strftime('%Y-%m-%d')
    st.download_button('Unduh hasil peramalan (CSV)', unduh.to_csv(index=False).encode('utf-8'),
                       file_name=f'peramalan_{kontrak.lower()}.csv', mime='text/csv')

with tab2:
    fig_c = go.Figure(go.Candlestick(
        x=hist_view['Date'], open=hist_view['Open'], high=hist_view['High'],
        low=hist_view['Low'], close=hist_view['Close'], name=kontrak,
        increasing_line_color='#1a7f37', decreasing_line_color='#c0392b'))
    gaya_grafik(fig_c, unit, tinggi=420)
    fig_c.update_layout(xaxis_rangeslider_visible=False, hovermode='x', showlegend=False)
    sumbu_tanggal(fig_c, hist_view['Date'].min(), hist_view['Date'].max())
    st.plotly_chart(fig_c, width='stretch', config=KONFIG_GRAFIK)
    st.caption(f'Grafik candlestick periode {periode.lower()} terakhir. Hijau: harga penutupan di atas '
               'harga pembukaan. Merah: harga penutupan di bawah harga pembukaan.')
    ringkas = hist_view[['Date'] + FEATURES].sort_values('Date', ascending=False).copy()
    ringkas['Date'] = ringkas['Date'].apply(fmt_tgl)
    for c in PRICE_COLS:
        ringkas[c] = ringkas[c].apply(fmt)
    ringkas['Volume'] = ringkas['Volume'].apply(lambda v: fmt(v, 0))
    with st.expander('Lihat tabel data historis'):
        st.dataframe(ringkas.rename(columns={'Date': 'Tanggal'}), width='stretch',
                     hide_index=True, height=320)

with tab3:
    mo = art['metrics_overall']
    st.markdown('**Model utama**, dievaluasi pada data testing 2024 sampai 27 Juli 2026')
    m1, m2, m3 = st.columns(3)
    m1.metric('MAE', fmt(mo['MAE']), help=f'Rata-rata selisih mutlak, dalam {unit}')
    m2.metric('RMSE', fmt(mo['RMSE']), help=f'Akar rata-rata kuadrat galat, dalam {unit}')
    m3.metric('MAPE', f"{fmt(mo['MAPE'])}%", help='Rata-rata persentase galat. Di bawah 10% = sangat baik')
    if has_retrain:
        rr = st.session_state['retrain_result'][kontrak]
        m = rr['metrics']
        st.markdown(f"**Model latih ulang**, diuji pada {rr['n_test_holdout']} hari perdagangan "
                    f"terakhir dari rentang terpilih ({fmt(rr['n_train'], 0)} hari untuk pelatihan)")
        r1, r2, r3 = st.columns(3)
        r1.metric('MAE', fmt(m['MAE']), f"{fmt(m['MAE'] - mo['MAE'])}", delta_color='inverse')
        r2.metric('RMSE', fmt(m['RMSE']), f"{fmt(m['RMSE'] - mo['RMSE'])}", delta_color='inverse')
        r3.metric('MAPE', f"{fmt(m['MAPE'])}%", f"{fmt(m['MAPE'] - mo['MAPE'])} poin",
                  delta_color='inverse')
        st.caption('Perbandingan bersifat indikatif karena kedua model diuji pada periode yang berbeda.')
    st.markdown('**MAPE menurut horizon peramalan**')
    mph = art['metrics_per_horizon']
    fig_h = go.Figure(garis(mph[mph.columns[0]], mph['MAPE'], 'MAPE (%)', WARNA['historis'],
                            marker=True))
    fig_h.data[0].hovertemplate = 'Hari ke-%{x}: %{y:,.2f}%<extra></extra>'
    fig_h.add_hline(y=10, line_dash='dot', line_color='#8a8984',
                    annotation_text='batas kategori sangat baik (10%)', annotation_position='top left')
    gaya_grafik(fig_h, 'MAPE (%)', tinggi=320)
    fig_h.update_layout(showlegend=False, xaxis_title='Hari ke-', hovermode='closest')
    fig_h.update_yaxes(tickformat=',.0f')
    st.plotly_chart(fig_h, width='stretch', config=KONFIG_GRAFIK)
    st.caption('Kesalahan peramalan meningkat seiring jauhnya hari yang diramalkan.')

with tab4:
    hp = art['best_hp']
    st.markdown(f"""
**Cara kerja singkat.** Model membaca pergerakan harga pembukaan, tertinggi, terendah,
penutupan, dan volume selama **{hp['sliding_window']} hari perdagangan terakhir**, lalu
meramalkan harga penutupan **{HORIZON} hari perdagangan ke depan** sekaligus. Lapisan
LSTM mempelajari pola urutan waktu, sedangkan *Attention Mechanism* memberi bobot pada
hari-hari yang dinilai lebih relevan.

**Konfigurasi model {kontrak}.** {hp['hidden_layers']} hidden layer, {hp['lstm_units']} unit LSTM,
batch size {hp['batch_size']}, learning rate {str(hp['learning_rate']).replace('.', ',')}, dropout
{fmt(hp['dropout'], 1)}, dan {hp['final_epochs']} epoch.

**Cara membaca hasil.**
- Garis oranye menunjukkan arah dan besar perubahan harga yang diperkirakan.
- Akurasi paling baik pada beberapa hari pertama dan menurun pada horizon yang lebih jauh.
- Tanggal peramalan mengasumsikan setiap Senin sampai Jumat adalah hari perdagangan, sehingga
  hari libur bursa dapat menggeser tanggal yang sebenarnya.
- Model hanya belajar dari pola harga historis. Kejadian di luar data tersebut, seperti
  cuaca ekstrem, kebijakan tarif, atau gangguan panen, tidak dapat diantisipasi.

**Penting.** Hasil peramalan merupakan estimasi dan sebaiknya digunakan sebagai salah satu
bahan pertimbangan, bukan satu-satunya dasar pengambilan keputusan.
""")
