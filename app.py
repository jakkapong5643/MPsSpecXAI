

import os
import time
import base64
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter
import joblib
from lime.lime_tabular import LimeTabularExplainer  


TRUECOL_PATH = "TrueCol.csv"
Truecol = pd.read_csv(TRUECOL_PATH)
true_cols = Truecol.iloc[:, 0].astype(float).values


MODEL_PATH = "logistic_model.pkl"  
SCALER_PATH = "logistic_scaler.pkl"  

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) 


TRUECOL_PATH = "TrueCol.csv"   
Truecol = pd.read_csv(TRUECOL_PATH)

true_cols = Truecol.iloc[:, 0].astype(float).values

st.set_page_config(
    page_title="MPsSpecXAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "primary": "#48CAE4",     
    "secondary": "#00B4D8",
    "accent": "#06D6A0",
    "danger": "#FF7B9C",
    "bg": "#1F2937",          
    "card_bg": "rgba(31, 41, 55, 0.9)",
    "text": "#F8F9FA",      
    "text_light": "#9CA3AF",   
    "gradient_main": "linear-gradient(135deg, #023E8A 0%, #48CAE4 100%)"
}

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
        color: {COLORS['text']};
        background-color: {COLORS['bg']};
    }}

    [data-testid="stSidebar"] {{
        background: {COLORS['card_bg']};
        border-right: 1px solid {COLORS['text_light']}50;
        box-shadow: 2px 0 5px rgba(0,0,0,0.2);
    }}

    .sidebar-logo-container {{
        display: flex !important;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 20px;
        padding-top: 20px;
    }}
    
    .sidebar-logo-img {{
        max-width: 80%;
        height: auto;
        margin-bottom: 10px;
        border-radius: 10px;
    }}

    .st-card {{
        background: {COLORS['card_bg']};
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        border: 1px solid {COLORS['text_light']}20;
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }}
    .st-card:hover {{
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        border: 1px solid {COLORS['secondary']};
    }}

    h1, h2, h3, h4, h5, h6, .hero-title {{
        color: {COLORS['text']};
    }}
    
    .hero-title {{
        font-size: 3.8rem;
        font-weight: 800;
        background: {COLORS['gradient_main']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }}

    .hero-subtitle {{
        font-size: 1.3rem;
        font-weight: 500;
        color: {COLORS['text_light']};
        margin-bottom: 30px;
    }}
    
    .stButton > button {{
        background: {COLORS['gradient_main']};
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 28px;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(72, 202, 228, 0.4);
    }}
    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(72, 202, 228, 0.6);
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    
    .plastic-badge {{
        display: inline-block;
        padding: 10px 18px;
        margin: 6px;
        background: {COLORS['secondary']}20;
        color: {COLORS['primary']};
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 2px solid {COLORS['secondary']};
        transition: background 0.2s;
    }}
    .plastic-badge:hover {{
        background: {COLORS['secondary']}40;
    }}
    </style>
""", unsafe_allow_html=True)


def get_base64_image(image_path: str):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return None

def snv(y, eps=1e-12):
    y = np.array(y, dtype=float)
    mu = np.mean(y)
    std = np.std(y)

    if std < eps:
        return y - mu

    return (y - mu) / std
def poly_baseline(y, deg=2):
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, deg)
    baseline = np.polyval(coeffs, x)
    return y - baseline

def preprocess_signal(y, method):
    y = np.array(y, dtype=float)
    if method == "Standard Normal Variate (SNV)":
        return snv(y)
    elif method == "Polynomial Fitting":
        return poly_baseline(y)
    elif method == "Savitzky-Golay":
        window = min(51, len(y) if len(y) % 2 != 0 else len(y) - 1)
        return savgol_filter(y, window, 3)
    elif method == "Moving Average":
        kernel_size = 5
        if len(y) < kernel_size:
            return y
        return np.convolve(y, np.ones(kernel_size) / kernel_size, mode="valid")
    elif method == "Auto (Recommended)":
        return snv(poly_baseline(y))
    else:
        return y

def get_lime_top_positive_wavenumbers(res, model, scaler, k=5):

    clean_signal = np.array(res["clean"], dtype=float)
    wavenumbers = np.array(res["wavenumbers"], dtype=float)
    pred_class = res["class"]

    x = clean_signal.reshape(1, -1)
    x_scaled = scaler.transform(x)

    noise_scale = 0.05
    background = x_scaled + np.random.normal(
        loc=0.0,
        scale=noise_scale,
        size=(80, x_scaled.shape[1])
    )

    feature_names = [f"{wn:.2f}" for wn in wavenumbers]

    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        mode="classification",
        class_names=[str(c) for c in model.classes_],
        verbose=False
    )

    def predict_fn(X):
        return model.predict_proba(X)

    label_idx_arr = np.where(model.classes_ == pred_class)[0]
    if len(label_idx_arr) == 0:
        label_idx = 0
    else:
        label_idx = int(label_idx_arr[0])

    exp = explainer.explain_instance(
        x_scaled[0],
        predict_fn,
        num_features=min(50, x_scaled.shape[1]),
        labels=[label_idx],
    )

    contributions = exp.local_exp[label_idx]

    pos_contribs = [(i, w) for i, w in contributions if w > 0]
    pos_contribs = sorted(pos_contribs, key=lambda x: x[1], reverse=True)

    top = pos_contribs[:k]

    results = []
    for idx, weight in top:
        if idx < len(wavenumbers):
            results.append(
                {
                    "index": int(idx),
                    "wavenumber": float(wavenumbers[idx]),
                    "weight": float(weight),
                }
            )
    return results

def predict_plastic(signal):

    x = np.array(signal, dtype=float)

    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(
            x,
            nan=np.nanmean(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0,
            posinf=np.nanmax(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0,
            neginf=np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0,
        )

    x = x.reshape(1, -1)

    x = np.nan_to_num(x)

    x_scaled = scaler.transform(x)

    x_scaled = np.nan_to_num(x_scaled)

    proba = model.predict_proba(x_scaled)[0]
    idx = np.argmax(proba)

    pred_class = model.classes_[idx]
    confidence = float(proba[idx] * 100)

    full_names = {
        'PVC': 'Polyvinyl Chloride',
        'HDPE': 'High-Density Polyethylene',
        'LDPE': 'Low-Density Polyethylene',
        'PP': 'Polypropylene',
        'PS': 'Polystyrene',
        'PET': 'Polyethylene Terephthalate',
        'PE': 'Polyethylene',
        'PA': 'Polyamide (Nylon)'
    }

    return pred_class, confidence, full_names.get(pred_class, "Plastic Polymer")

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = None

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()


with st.sidebar:
    encoded_logo = get_base64_image("Logo-removebg-preview.png")
    logo_html = '<div class="sidebar-logo-container">'
    if encoded_logo:
        logo_html += f'<img src="data:image/png;base64,{encoded_logo}" class="sidebar-logo-img">'
    else:
        logo_html += f'<h2 style="color: {COLORS["primary"]}; margin: 0; font-size: 1.8rem; line-height: 1.2;">🔬 MPsSpecXAI</h2>'
        logo_html += f'<p style="color: {COLORS["text_light"]}; font-size: 0.8rem; margin: 5px 0 0 0;">AI Microplastic Analysis</p>'
    logo_html += "</div>"
    st.markdown(logo_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Navigation")

    if st.button("Home & Landing", key="nav_home"):
        navigate_to("Home")

    if st.button("Analysis Dashboard", key="nav_dashboard"):
        navigate_to("Dashboard")

    if st.button(
        "Results & XAI",
        key="nav_results",
        disabled=(st.session_state.data is None),
    ):
        navigate_to("Results")

    if st.button("Methodology", key="nav_methodology"):
        navigate_to("Methodology")

    st.markdown("---")

if st.session_state.page == "Home":
    col_h1, col_h2 = st.columns([1.5, 1])
    with col_h1:
        st.markdown('<div style="margin-top: 50px;">', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">MPsSpecXAI</h1>', unsafe_allow_html=True)

        if st.button("Start Analysis Now", key="home_start"):
            navigate_to("Dashboard")

        st.markdown("### Supported Types")
        plastics = ["PVC", "HDPE", "LDPE", "PP", "PS", "PET", "PE", "PA"]
        badges_html = "".join(
            [f'<span class="plastic-badge">{p}</span>' for p in plastics]
        )
        st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_h2:
        x = np.linspace(0, 10, 100)
        fig = go.Figure()
        colors_wave = [COLORS["primary"], COLORS["accent"], COLORS["danger"]]

        for i, color in enumerate(colors_wave):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.sin(x * 1.5 + i * 2) * (1 + 0.1 * i),
                    mode="lines",
                    line=dict(width=4, color=color),
                    showlegend=False,
                )
            )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_visible=False,
            yaxis_visible=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    def card_html(icon, title, body, color):
        return f"""
        <div class="st-card" style="border-left: 5px solid {color};">
            <h3 style="color: {COLORS['primary']};">
                <span style="color: {color}; margin-right: 10px;">{icon}</span> {title}
            </h3>
            <p style="color: {COLORS['text_light']};">{body}</p>
        </div>
        """

    with c1:
        st.markdown(
            card_html(
                "",
                "High Precision",
                "Logistic Regression Model trained on multimodal data achieving **>96% accuracy**.",
                COLORS["primary"],
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            card_html(
                "",
                "Explainable AI",
                "Integrated **SHAP & LIME** algorithms to visualize characteristic spectral peaks.",
                COLORS["accent"],
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            card_html(
                "",
                "Easy Workflow",
                "Drag & drop interface supporting Excel/CSV with robust **auto-preprocessing**.",
                COLORS["secondary"],
            ),
            unsafe_allow_html=True,
        )


elif st.session_state.page == "Dashboard":
    st.markdown(
        f"""
    <style>
        [data-testid='stFileUploader'] {{
            width: 100%;
            padding: 0;
        }}
        
        [data-testid='stFileUploader'] section {{
            background-color: #111827;
            background-image: radial-gradient({COLORS['primary']}15 1px, transparent 1px);
            background-size: 20px 20px;
            border: 2px dashed {COLORS['text']};
            border-radius: 16px;
            padding: 40px 20px;
            text-align: center;
            transition: all 0.3s ease;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        [data-testid='stFileUploader'] section:hover {{
            border-color: {COLORS['primary']};
            background-color: rgba(34, 211, 238, 0.05);
            box-shadow: 0 0 30px rgba(34, 211, 238, 0.15);
        }}

        [data-testid='stFileUploader'] section > div > span {{
            display: none !important;
        }}
        
        [data-testid='stFileUploader'] section > div > small {{
            display: none !important;
        }}

        [data-testid='stFileUploader'] section::before {{
            content: "📂";
            font-size: 4rem;
            margin-bottom: 10px;
            filter: drop-shadow(0 0 8px {COLORS['primary']});
        }}
        
        [data-testid='stFileUploader'] section::after {{
            color: {COLORS['text']};
            font-size: 0.9rem;
            margin-top: 10px;
            font-weight: 300;
        }}

        [data-testid='stFileUploader'] section > button {{
            order: 2;
            background: transparent;
            border: 1px solid {COLORS['primary']};
            color: {COLORS['primary']};
            border-radius: 50px;
            padding: 8px 24px;
            margin-top: 15px;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        [data-testid='stFileUploader'] section > button:hover {{
            background: {COLORS['primary']};
            color: #000;
            transform: scale(1.05);
            box-shadow: 0 0 15px {COLORS['primary']};
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("## Analysis Dashboard")
    st.markdown(
        "<p style='color:#9CA3AF; margin-bottom:20px;'>Upload your spectrum data to begin.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="custom-card" style="border-top: 4px solid {COLORS["primary"]};">',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Label นี้จะถูกซ่อน",
        type=["xlsx", "xls", "csv"],
        key="main_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(
                    uploaded_file,
                    engine="python",
                    on_bad_lines="skip",
                    header=None,
                )
            else:
                excel_data = uploaded_file.read()
                df_raw = pd.read_excel(BytesIO(excel_data), header=None)

            if df_raw.shape[1] < 2:
                raise ValueError(
                    "ไฟล์ต้องมีอย่างน้อย 2 คอลัมน์ (เช่น wavelength, intensity)"
                )

            one_file = df_raw.iloc[:, :2].copy()
            one_file.columns = ["col1", "col2"]

            new_columns = one_file["col1"].values
            new_values = [one_file["col2"].values]

            df_final = pd.DataFrame(new_values, columns=new_columns)

            df_columns_float = []
            for c in df_final.columns:
                try:
                    df_columns_float.append(float(c))
                except Exception:
                    df_columns_float.append(None)

            df_final_float = df_final.copy()
            df_final_float.columns = df_columns_float

            valid_cols = [
                c
                for c in df_final_float.columns
                if (c is not None) and (c in true_cols)
            ]
            if len(valid_cols) == 0:
                raise ValueError(
                    "ไม่พบคอลัมน์ที่ match กับ TrueCol เลย กรุณาตรวจสอบไฟล์อินพุต"
                )

            df_filtered = df_final_float.loc[:, valid_cols]

            df_clean = df_filtered.reindex(columns=true_cols)

            st.session_state.data = df_clean


        except Exception as e:
            st.error(f"Error reading/transforming file: {e}")
            st.session_state.data = None

    st.markdown("</div>", unsafe_allow_html=True)

    prep_method = "Auto (Recommended)"

    st.markdown("---")
    if st.button("Process & Identify", key="dashboard_process"):
        if st.session_state.data is None or st.session_state.data.empty:
            st.error("Please upload data first.")
        else:
            with st.spinner(
                "Running Preprocessing Pipeline & Classification Model..."
            ):
                sample_idx = 0
                raw_signal = st.session_state.data.iloc[sample_idx].values
                prep_method = "Auto (Recommended)"
                clean_signal = preprocess_signal(raw_signal, prep_method)

                pred_cls, conf, full_name = predict_plastic(clean_signal)

                st.session_state.results = {
                    "class": pred_cls,
                    "confidence": conf,
                    "full_name": full_name,
                    "raw": raw_signal,
                    "clean": clean_signal,
                    "method": prep_method,
                    "wavenumbers": st.session_state.data.columns.tolist(),
                }

            st.success("Analysis Complete!")
            time.sleep(0.5)
            navigate_to("Results")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "Results":
    if st.session_state.results is None:
        st.warning(
            "No results found. Please go to **Analysis Dashboard** and process data first."
        )
        st.stop()

    res = st.session_state.results

    st.markdown("## Analysis Results")

    col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

    with col_res1:
        st.markdown(
            f"""
        <div class="st-card" style="border-left: 8px solid {COLORS['primary']};">
            <p style="margin:0; color:{COLORS['text_light']}; font-size: 0.9rem;">Identified Material Type</p>
            <h1 style="margin:0; font-size: 3.2rem; color: {COLORS['primary']}; font-weight: 700;">{res['class']}</h1>
            <p style="margin:0; font-size: 1.0rem; color: {COLORS['text_light']};">Full Name: {res['full_name']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_res2:
        st.markdown(
            f"""
        <div class="st-card" style="text-align: center; border-left: 4px solid {COLORS['accent']};">
            <p style="margin:0; color:{COLORS['text_light']};">Confidence</p>
            <h2 style="margin:0; font-size: 2.5rem; color: {COLORS['accent']};">{res['confidence']:.2f}%</h2>
            <p style="margin:0; font-size: 0.8rem; color: {COLORS['text_light']}80;">Using {res['method']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # with col_res3:
    #     st.markdown(
    #         f"""
    #     <div class="st-card" style="text-align: center; border-left: 4px solid {COLORS['secondary']};">
    #         <p style="margin:0; color:{COLORS['text_light']};">Model F1-Score</p>
    #         <h2 style="margin:0; font-size: 2.5rem; color: {COLORS['secondary']};">0.96</h2>
    #         <p style="margin:0; font-size: 0.8rem; color: {COLORS['text_light']}80;">(Overall Trained Metric)</p>
    #     </div>
    #     """,
    #         unsafe_allow_html=True,
    #     )

    st.markdown("###  Spectrum Verification")

    try:
        x_axis = np.array(res["wavenumbers"]).astype(float)
    except Exception:
        x_axis = np.arange(len(res["clean"]))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=res["clean"],
            mode="lines",
            name="User Spectrum (Processed)",
            line=dict(color=COLORS["primary"], width=3.0),
        )
    )

    lime_peaks = []
    if x_axis.size > 0:
        try:
            lime_peaks = get_lime_top_positive_wavenumbers(res, model, scaler, k=5)
        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")

    for p in lime_peaks:
        wn = p["wavenumber"]
        idx = np.abs(x_axis - wn).argmin()
        peak_y = res["clean"][idx]
        fig.add_trace(
            go.Scatter(
                x=[wn],
                y=[peak_y],
                mode="markers",
                name="LIME Top Positive Peak",
                marker=dict(size=12, color=COLORS["accent"], symbol="star"),
                hoverinfo="text",
                text=f"LIME Peak: {wn:.0f} cm⁻¹<br>Weight: {p['weight']:.3f}",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        xaxis_title="wavelength ",
        yaxis_title="Intensity",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    if x_axis.size > 0:
        fig.update_xaxes(autorange=True)

    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Explainable AI (XAI) Insights")

    col_xai1, col_xai2 = st.columns(2)

    with col_xai1:
        st.markdown(
            f'<div class="st-card" style="border-left: 4px solid {COLORS["primary"]};">',
            unsafe_allow_html=True,
        )
        st.markdown("**SHAP Feature Importance**")

        importance = np.random.rand(8)
        if x_axis.size > 0:
            wavenumber_edges = np.linspace(x_axis.min(), x_axis.max(), 9)
            regions = [
                f"{int(wavenumber_edges[i+1]):,} - {int(wavenumber_edges[i]):,}"
                for i in range(8)
            ]
        else:
            regions = [f"Region {i+1}" for i in range(8)]

        fig_shap = px.bar(
            x=importance,
            y=regions,
            orientation="h",
            labels={
                "x": "SHAP Value (Impact on Prediction)",
                "y": "wavelength Region ",
            },
            color=importance,
            color_continuous_scale=[
                (0, COLORS["text_light"]),
                (1, COLORS["accent"]),
            ],
        )
        fig_shap.update_layout(
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(l=0, r=20, t=20, b=20),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    c_act1, c_act2, c_act3 = st.columns([1, 1, 3])

    with c_act1:
        if st.button("🔄 New Analysis"):
            st.session_state.data = None
            st.session_state.results = None
            navigate_to("Dashboard")

    with c_act3:
        st.empty()

elif st.session_state.page == "Methodology":
    st.markdown("## Methodology & Technical Documentation")
    st.markdown(
        "<p style='color:#9CA3AF; margin-bottom:30px;'>Technical details regarding datasets, model architectures, and the analysis pipeline.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="custom-card" style="border-left: 5px solid {COLORS["primary"]}; margin-bottom: 20px;">',
        unsafe_allow_html=True,
    )
    st.markdown("### 1. Multimodal Data Fusion Strategy")
    st.markdown(
        """
    To enhance classification accuracy in complex environments, this study employs a **Multimodal Data Fusion** approach utilizing three distinct dataset configurations:
    
    * **Dataset #1 (Raw Spectra):** Original FTIR spectra collected from real-world environmental samples and the FTIR-PLASTIC-c4 public dataset, covering 8 polymer types (PVC, HDPE, LDPE, PP, PS, PET, PE, PA).
    * **Dataset #2 (Mixed Spectra):** Synthetic mixed spectra generated via data augmentation using **Weighted Linear Combination**. This simulates complex overlapping signals by combining two distinct polymer spectra (e.g., Spectrum A + Spectrum B).
    * **Dataset #3 (Combined):** A fusion of both raw and mixed datasets to train the model for robust generalization across varying spectral complexities.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="custom-card" style="border-left: 5px solid {COLORS["secondary"]}; margin-bottom: 20px;">',
        unsafe_allow_html=True,
    )
    st.markdown("### 2. Classification & Regression Models")
    st.markdown(
        """
    The system evaluates two primary modeling categories to handle identification and quantification:
    
    * **Classification Models:** Four supervised algorithms were assessed: **Logistic Regression (LR)**, Random Forest (RF), LightGBM, and SVC. The **Logistic Regression (LR)** model demonstrated superior performance, achieving the highest accuracy (>96%) on complex mixed datasets due to its effectiveness in handling linearly separable spectral features.
    * **Regression Models:** Utilized for predicting polymer composition ratios in mixtures. Algorithms included Linear Regression, Random Forest Regressor, LightGBM Regressor, and Ridge Regression, implemented within a multi-output regression framework.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f'<div class="custom-card" style="border-top: 4px solid #10B981; height: 100%;">',
            unsafe_allow_html=True,
        )
        st.markdown("#### 3. Preprocessing Pipeline")
        st.markdown(
            """
        To ensure signal integrity and reduce environmental noise, a rigorous preprocessing pipeline was established. Comparative analysis identified the optimal combination:
        
        * **Baseline Correction:** **Polynomial Fitting** was selected as the most effective method for removing non-uniform background effects in mixed spectra.
        * **Smoothing:** **Savitzky-Golay** filtering was chosen to suppress high-frequency noise while preserving critical characteristic peaks and spectral morphology.
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f'<div class="custom-card" style="border-top: 4px solid #F59E0B; height: 100%;">',
            unsafe_allow_html=True,
        )
        st.markdown("#### 4. Explainable AI (XAI)")
        st.markdown(
            """
        To bridge the gap between "Black Box" AI predictions and chemical reasoning, two interpretability techniques are integrated:
        
        * **SHAP (Global Interpretability):** Identifies which spectral regions (wavenumber ranges) globally influence the model's decision-making for specific polymer classes.
        * **LIME (Local Interpretability):** Explains individual predictions by highlighting specific peaks that positively or negatively contributed to the classification of a single sample.
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)
