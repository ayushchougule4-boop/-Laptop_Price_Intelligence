import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Laptop Price Intelligence",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "laptop_dataset.csv")
COMPATIBLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "laptop_price_prediction_model_compatible.pkl"
)


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(37,99,235,.07), transparent 30%),
            radial-gradient(circle at 90% 0%, rgba(14,165,233,.07), transparent 30%),
            #f5f7fb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] .stCaption {
        color: #cbd5e1 !important;
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 34px;
        border-radius: 24px;
        color: white;
        background:
            radial-gradient(circle at 85% 20%, rgba(96,165,250,.20), transparent 25%),
            linear-gradient(135deg, #0b1220 0%, #162033 52%, #26364f 100%);
        box-shadow: 0 20px 45px rgba(15,23,42,.15);
        margin-bottom: 22px;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -70px;
        top: -90px;
        border-radius: 50%;
        background: rgba(59,130,246,.10);
    }

    .hero-kicker {
        color: #93c5fd;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .hero-title {
        font-size: 42px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin: 0;
    }

    .hero-subtitle {
        color: #dbeafe;
        font-size: 16px;
        margin-top: 9px;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 16px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.09);
        border: 1px solid rgba(255,255,255,.12);
        color: #e2e8f0;
        font-size: 12px;
        font-weight: 650;
    }

    /* ---------- KPI cards ---------- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.93);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 17px 18px;
        box-shadow: 0 8px 24px rgba(15,23,42,.06);
        min-height: 115px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -.5px;
    }

    /* ---------- Section cards ---------- */
    .section-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px 20px 8px 20px;
        box-shadow: 0 8px 24px rgba(15,23,42,.05);
        margin-top: 18px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 780;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 8px;
    }

    .insight-box {
        background: linear-gradient(135deg, #eff6ff, #f8fafc);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 16px;
    }

    .insight-title {
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 3px;
    }

    .insight-text {
        color: #475569;
        font-size: 13px;
    }

    .filter-summary {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 10px 14px;
        margin-bottom: 14px;
        color: #475569;
        font-size: 13px;
    }

    .filter-summary b {
        color: #0f172a;
    }

    /* ---------- Buttons ---------- */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 11px;
        font-weight: 700;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 7px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return data


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(data: pd.DataFrame, possible_names):
    normalized = {
        normalize_name(column): column
        for column in data.columns
    }

    for name in possible_names:
        key = normalize_name(name)
        if key in normalized:
            return normalized[key]

    return None


def find_first_column(data: pd.DataFrame, groups):
    for group in groups:
        column = find_column(data, group)
        if column is not None:
            return column
    return None


def extract_unsigned_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )


def extract_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )


def extract_storage(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower()

    values = pd.to_numeric(
        text.str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )

    tb_mask = text.str.contains(r"\btb\b", na=False)
    values.loc[tb_mask] = values.loc[tb_mask] * 1024

    return values


def money(value) -> str:
    if value is None or pd.isna(value):
        return "—"

    value = float(value)

    if value >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"
    if value >= 100_000:
        return f"₹{value / 1_000:.1f}K"
    return f"₹{value:,.0f}"


def format_axis_currency(fig):
    fig.update_yaxes(tickprefix="₹", separatethousands=True)
    return fig


def clean_price_data(
    data: pd.DataFrame,
    min_price: int = 10_000,
    max_price: int = 500_000,
):
    """
    Creates a separate analytics dataset.

    Raw CSV is never modified.
    """
    data = data.copy()

    if "Price" not in data.columns:
        return data, {
            "invalid": 0,
            "range_removed": 0,
            "outliers": 0,
            "removed": 0,
        }

    price = pd.to_numeric(data["Price"], errors="coerce")

    invalid_mask = price.isna() | (price <= 0)
    range_mask = (~invalid_mask) & (
        (price < min_price) | (price > max_price)
    )

    cleaned = data.loc[~invalid_mask].copy()
    cleaned = cleaned.loc[
        cleaned["Price"].between(min_price, max_price)
    ].copy()

    outliers = 0

    if len(cleaned) >= 20:
        q1 = cleaned["Price"].quantile(0.25)
        q3 = cleaned["Price"].quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            lower = max(min_price, q1 - 1.5 * iqr)
            upper = min(max_price, q3 + 1.5 * iqr)
            keep = cleaned["Price"].between(lower, upper)

            outliers = int((~keep).sum())
            cleaned = cleaned.loc[keep].copy()

    stats = {
        "invalid": int(invalid_mask.sum()),
        "range_removed": int(range_mask.sum()),
        "outliers": int(outliers),
        "removed": int(len(data) - len(cleaned)),
    }

    return cleaned, stats


def get_unique_values(data: pd.DataFrame, column_name):
    if column_name is None or column_name not in data.columns:
        return []

    return sorted(
        data[column_name]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Keep laptop_dataset.csv in the same folder as app.py."
        )

    data = pd.read_csv(DATA_PATH, low_memory=False)
    data = clean_column_names(data)

    if data.empty:
        raise ValueError("The dataset is empty.")

    return data


@st.cache_data(show_spinner=False)
def prepare_data(data: pd.DataFrame):
    data = data.copy()

    ram_col = find_first_column(
        data,
        [
            ["RAM", "Ram", "RAM (GB)", "RAM GB", "Memory"],
        ],
    )

    ssd_col = find_first_column(
        data,
        [
            ["SSD Capacity", "SSD", "SSD Capacity (GB)"],
            ["Solid State Drive"],
        ],
    )

    hdd_col = find_first_column(
        data,
        [
            ["HDD Capacity", "HDD", "HDD Capacity (GB)"],
            ["Hard Disk"],
        ],
    )

    display_col = find_first_column(
        data,
        [
            [
                "Display Size",
                "Screen Size",
                "Display Size (inches)",
                "Screen Size (inches)",
            ],
        ],
    )

    weight_col = find_first_column(
        data,
        [
            ["Weight", "Weight (kg)", "Laptop Weight"],
        ],
    )

    price_col = find_first_column(
        data,
        [
            ["Price (Rs)", "Price", "Price (₹)", "Price Rs", "Selling Price"],
        ],
    )

    clock_col = find_column(
        data,
        [
            "Clock Speed",
            "Clock Speed (GHz)",
            "Clock Speed GHz",
            "CPU Clock Speed",
            "Processor Clock Speed",
        ],
    )

    pixel_col = find_column(
        data,
        [
            "Pixel Density",
            "Pixel Density (PPI)",
            "PPI",
            "Pixels Per Inch",
        ],
    )

    refresh_col = find_column(
        data,
        [
            "Refresh Rate",
            "Refresh Rate (Hz)",
            "RefreshRate",
            "Screen Refresh Rate",
        ],
    )

    cores_col = find_column(
        data,
        [
            "Number of Cores",
            "Cores",
            "CPU Cores",
            "Processor Cores",
        ],
    )

    battery_col = find_column(
        data,
        [
            "Battery Capacity",
            "Battery Capacity (mAh)",
            "Battery (mAh)",
            "Battery Capacity (Wh)",
        ],
    )

    data["RAM_GB"] = (
        extract_unsigned_number(data[ram_col])
        if ram_col
        else np.nan
    )

    data["SSD_GB"] = (
        extract_storage(data[ssd_col])
        if ssd_col
        else np.nan
    )

    data["HDD_GB"] = (
        extract_storage(data[hdd_col])
        if hdd_col
        else np.nan
    )

    data["Display_Inches"] = (
        extract_unsigned_number(data[display_col])
        if display_col
        else np.nan
    )

    data["Weight_KG"] = (
        extract_unsigned_number(data[weight_col])
        if weight_col
        else np.nan
    )

    data["Price"] = (
        extract_price(data[price_col])
        if price_col
        else np.nan
    )

    data["clock_speed_ghz"] = (
        extract_unsigned_number(data[clock_col])
        if clock_col
        else np.nan
    )

    data["pixel_density"] = (
        extract_unsigned_number(data[pixel_col])
        if pixel_col
        else np.nan
    )

    data["refresh_rate_hz"] = (
        extract_unsigned_number(data[refresh_col])
        if refresh_col
        else np.nan
    )

    data["cores"] = (
        extract_unsigned_number(data[cores_col])
        if cores_col
        else np.nan
    )

    data["battery_capacity"] = (
        extract_unsigned_number(data[battery_col])
        if battery_col
        else np.nan
    )

    return data


# ============================================================
# PREDICTION MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def build_prediction_model(training_frame: pd.DataFrame):
    """
    Train a fresh model in the current Python/scikit-learn environment.

    This intentionally avoids loading the incompatible legacy .pkl model.
    """
    feature_columns = [
        "Brand",
        "Series",
        "Operating System",
        "Processor",
        "Graphic Processor",
        "RAM Type",
        "SSD Type",
        "Display Type",
        "Display Touchscreen",
        "ram_gb",
        "ssd_gb",
        "hdd_gb",
        "display_inches",
        "weight_kg",
        "clock_speed_ghz",
        "pixel_density",
        "refresh_rate_hz",
        "cores",
        "battery_capacity",
    ]

    categorical_features = [
        "Brand",
        "Series",
        "Operating System",
        "Processor",
        "Graphic Processor",
        "RAM Type",
        "SSD Type",
        "Display Type",
        "Display Touchscreen",
    ]

    numeric_features = [
        "ram_gb",
        "ssd_gb",
        "hdd_gb",
        "display_inches",
        "weight_kg",
        "clock_speed_ghz",
        "pixel_density",
        "refresh_rate_hz",
        "cores",
        "battery_capacity",
    ]

    alias_map = {
        "Brand": ["Brand", "Manufacturer"],
        "Series": ["Series", "Model Series"],
        "Operating System": ["Operating System", "OS"],
        "Processor": ["Processor", "CPU"],
        "Graphic Processor": [
            "Graphic Processor",
            "Graphics Processor",
            "GPU",
        ],
        "RAM Type": ["RAM Type", "Memory Type"],
        "SSD Type": ["SSD Type", "Storage Type"],
        "Display Type": ["Display Type", "Screen Type"],
        "Display Touchscreen": [
            "Display Touchscreen",
            "Touchscreen",
            "Touch Screen",
        ],
    }

    raw = training_frame.copy()
    train = pd.DataFrame(index=raw.index)

    for target_col, aliases in alias_map.items():
        source_col = find_column(raw, aliases)

        if source_col is not None:
            train[target_col] = raw[source_col]
        else:
            train[target_col] = "Unknown"

    for column in numeric_features:
        train[column] = (
            raw[column]
            if column in raw.columns
            else np.nan
        )

    train["Price"] = pd.to_numeric(
        raw["Price"],
        errors="coerce"
    )

    train = train[
        feature_columns + ["Price"]
    ].replace([np.inf, -np.inf], np.nan)

    train = train[
        train["Price"].between(10_000, 500_000)
    ].copy()

    if len(train) < 20:
        raise ValueError(
            "At least 20 valid laptop records are required to train the model."
        )

    for column in categorical_features:
        train[column] = (
            train[column]
            .fillna("Unknown")
            .astype(str)
        )

    for column in numeric_features:
        train[column] = pd.to_numeric(
            train[column],
            errors="coerce"
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    ),
                ]),
                categorical_features,
            ),
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                ]),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=220,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    pipeline.fit(
        train[feature_columns],
        train["Price"]
    )

    try:
        joblib.dump(
            pipeline,
            COMPATIBLE_MODEL_PATH
        )
    except Exception:
        pass

    return pipeline


# ============================================================
# INITIALIZE
# ============================================================

try:
    df_raw = load_data()
    df = prepare_data(df_raw)
except Exception as exc:
    st.error("Unable to load the laptop dataset.")
    st.exception(exc)
    st.stop()


df_clean, cleaning_stats = clean_price_data(
    df,
    min_price=10_000,
    max_price=500_000
)


# ============================================================
# COLUMN MAP FOR FILTERS
# ============================================================

BRAND_COL = find_first_column(
    df_clean,
    [["Brand", "Manufacturer"]]
)

LAPTOP_NAME_COL = find_first_column(
    df_clean,
    [
        [
            "Laptop Name",
            "Laptop",
            "Laptop Name",
            "Product Name",
            "Product",
            "Title",
            "Notebook Name",
        ],
    ],
)

MODEL_COL = find_first_column(
    df_clean,
    [
        [
            "Model",
            "Laptop Model",
            "Model Name",
            "Product Model",
        ],
    ],
)

SERIES_COL = find_first_column(
    df_clean,
    [["Series", "Model Series"]]
)

OS_COL = find_first_column(
    df_clean,
    [["Operating System", "OS"]]
)

PROCESSOR_COL = find_first_column(
    df_clean,
    [["Processor", "CPU"]]
)

GPU_COL = find_first_column(
    df_clean,
    [
        ["Graphic Processor", "Graphics Processor", "GPU"]
    ]
)

RAM_TYPE_COL = find_first_column(
    df_clean,
    [["RAM Type", "Memory Type"]]
)

SSD_TYPE_COL = find_first_column(
    df_clean,
    [["SSD Type", "Storage Type"]]
)

DISPLAY_TYPE_COL = find_first_column(
    df_clean,
    [["Display Type", "Screen Type"]]
)

TOUCHSCREEN_COL = find_first_column(
    df_clean,
    [["Display Touchscreen", "Touchscreen", "Touch Screen"]]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        padding:8px 2px 18px 2px;
        font-size:24px;
        font-weight:800;
    ">
        💻 Laptop Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔍 Laptop Analysis",
        "🤖 Price Prediction",
        "📋 Dataset",
    ],
)

st.sidebar.markdown("---")

# Dashboard / analysis filters
st.sidebar.markdown("### 🎛️ Explore Filters")
st.sidebar.caption(
    "Filters are applied to the cleaned analytics dataset."
)

available_prices = df_clean["Price"].dropna()

if available_prices.empty:
    st.sidebar.error("No valid prices are available.")
    st.stop()

price_floor = int(available_prices.min())
price_ceiling = int(available_prices.max())

price_range = st.sidebar.slider(
    "💰 Price Range (₹)",
    min_value=price_floor,
    max_value=price_ceiling,
    value=(price_floor, price_ceiling),
    step=1_000 if price_ceiling - price_floor >= 50_000 else 500,
)

search_name = st.sidebar.text_input(
    "🔎 Laptop Name / Model Search",
    placeholder="e.g. Inspiron, ThinkPad, Pavilion...",
)

brand_values = get_unique_values(df_clean, BRAND_COL)
selected_brands = st.sidebar.multiselect(
    "🏷️ Brand",
    brand_values,
    placeholder="All brands",
)

model_values = get_unique_values(df_clean, MODEL_COL)
selected_models = st.sidebar.multiselect(
    "🧩 Laptop Model",
    model_values,
    placeholder="All models",
)

series_values = get_unique_values(df_clean, SERIES_COL)
selected_series = st.sidebar.multiselect(
    "📦 Series",
    series_values,
    placeholder="All series",
)

processor_values = get_unique_values(df_clean, PROCESSOR_COL)
selected_processors = st.sidebar.multiselect(
    "⚙️ Processor",
    processor_values,
    placeholder="All processors",
)

gpu_values = get_unique_values(df_clean, GPU_COL)
selected_gpus = st.sidebar.multiselect(
    "🎮 Graphics Processor",
    gpu_values,
    placeholder="All graphics",
)

os_values = get_unique_values(df_clean, OS_COL)
selected_os = st.sidebar.multiselect(
    "🪟 Operating System",
    os_values,
    placeholder="All operating systems",
)

ram_values = sorted(
    [x for x in df_clean["RAM_GB"].dropna().unique().tolist()]
)

selected_ram = st.sidebar.multiselect(
    "🧠 RAM (GB)",
    ram_values,
    placeholder="All RAM sizes",
)

ssd_values = sorted(
    [x for x in df_clean["SSD_GB"].dropna().unique().tolist()]
)

selected_ssd = st.sidebar.multiselect(
    "💾 SSD (GB)",
    ssd_values,
    placeholder="All SSD sizes",
)

hdd_values = sorted(
    [x for x in df_clean["HDD_GB"].dropna().unique().tolist()]
)

selected_hdd = st.sidebar.multiselect(
    "🗄️ HDD (GB)",
    hdd_values,
    placeholder="All HDD sizes",
)

display_size_values = sorted(
    [
        x
        for x in df_clean["Display_Inches"]
        .dropna()
        .unique()
        .tolist()
    ]
)

selected_display_size = st.sidebar.multiselect(
    "🖥️ Display Size",
    display_size_values,
    placeholder="All display sizes",
)

touch_values = get_unique_values(
    df_clean,
    TOUCHSCREEN_COL
)

selected_touch = st.sidebar.multiselect(
    "👆 Touchscreen",
    touch_values,
    placeholder="All touchscreen types",
)


# ============================================================
# APPLY GLOBAL FILTERS
# ============================================================

filtered_df = df_clean.copy()

filtered_df = filtered_df[
    filtered_df["Price"].between(
        price_range[0],
        price_range[1]
    )
]

if search_name:
    searchable_columns = [
        column
        for column in [
            BRAND_COL,
            LAPTOP_NAME_COL,
            MODEL_COL,
            SERIES_COL,
        ]
        if column is not None and column in filtered_df.columns
    ]

    if searchable_columns:
        mask = pd.Series(
            False,
            index=filtered_df.index
        )

        for column in searchable_columns:
            mask = (
                mask
                | filtered_df[column]
                .astype(str)
                .str.contains(
                    search_name,
                    case=False,
                    na=False,
                    regex=False,
                )
            )

        filtered_df = filtered_df.loc[mask]

if selected_brands and BRAND_COL:
    filtered_df = filtered_df[
        filtered_df[BRAND_COL]
        .astype(str)
        .isin(selected_brands)
    ]

if selected_models and MODEL_COL:
    filtered_df = filtered_df[
        filtered_df[MODEL_COL]
        .astype(str)
        .isin(selected_models)
    ]

if selected_series and SERIES_COL:
    filtered_df = filtered_df[
        filtered_df[SERIES_COL]
        .astype(str)
        .isin(selected_series)
    ]

if selected_processors and PROCESSOR_COL:
    filtered_df = filtered_df[
        filtered_df[PROCESSOR_COL]
        .astype(str)
        .isin(selected_processors)
    ]

if selected_gpus and GPU_COL:
    filtered_df = filtered_df[
        filtered_df[GPU_COL]
        .astype(str)
        .isin(selected_gpus)
    ]

if selected_os and OS_COL:
    filtered_df = filtered_df[
        filtered_df[OS_COL]
        .astype(str)
        .isin(selected_os)
    ]

if selected_ram:
    filtered_df = filtered_df[
        filtered_df["RAM_GB"].isin(selected_ram)
    ]

if selected_ssd:
    filtered_df = filtered_df[
        filtered_df["SSD_GB"].isin(selected_ssd)
    ]

if selected_hdd:
    filtered_df = filtered_df[
        filtered_df["HDD_GB"].isin(selected_hdd)
    ]

if selected_display_size:
    filtered_df = filtered_df[
        filtered_df["Display_Inches"].isin(
            selected_display_size
        )
    ]

if selected_touch and TOUCHSCREEN_COL:
    filtered_df = filtered_df[
        filtered_df[TOUCHSCREEN_COL]
        .astype(str)
        .isin(selected_touch)
    ]


st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Dataset health**

    Raw rows: **{len(df_raw):,}**

    Clean rows: **{len(df_clean):,}**

    Removed: **{cleaning_stats['removed']:,}**
    """
)


# ============================================================
# PAGE 1 — PREMIUM DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Laptop market analytics</div>
            <div class="hero-title">💻 Laptop Price Intelligence</div>
            <div class="hero-subtitle">
                Explore prices, laptop models and hardware specifications
                using clean, filtered market data.
            </div>
            <div class="hero-badge">
                ✨ Professional analytics view • Cleaned pricing data
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.warning(
            "No laptops match the selected filters. "
            "Try widening the price range or removing one of the filters."
        )
        st.stop()

    filtered_prices = filtered_df["Price"].dropna()

    avg_price = filtered_prices.mean()
    min_price = filtered_prices.min()
    max_price = filtered_prices.max()

    st.markdown(
        f"""
        <div class="filter-summary">
            <b>{len(filtered_df):,}</b> laptops currently match your filters
            • Clean dataset: <b>{len(df_clean):,}</b> laptops
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "💻 Laptops Analyzed",
            f"{len(filtered_df):,}",
        )

    with k2:
        st.metric(
            "💰 Average Price",
            money(avg_price),
        )

    with k3:
        st.metric(
            "📉 Minimum Price",
            money(min_price),
        )

    with k4:
        st.metric(
            "📈 Maximum Price",
            money(max_price),
        )

    # ========================================================
    # PRICE DISTRIBUTION
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">💰 Price Distribution</div>
            <div class="section-subtitle">
                See how laptop prices are distributed within the active filters.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = px.histogram(
        filtered_df,
        x="Price",
        nbins=32,
        marginal="box",
        template="plotly_white",
        title="Filtered laptop price distribution",
        labels={
            "Price": "Laptop Price (₹)",
            "count": "Number of Laptops",
        },
    )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
        bargap=0.08,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # BRAND / MODEL
    # ========================================================

    left, right = st.columns(2)

    with left:
        if BRAND_COL:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">🏷️ Leading Brands</div>
                    <div class="section-subtitle">
                        Most represented brands in the current filtered dataset.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            brand_counts = (
                filtered_df[BRAND_COL]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
                .sort_values()
                .reset_index()
            )

            brand_counts.columns = [
                "Brand",
                "Count",
            ]

            fig = px.bar(
                brand_counts,
                x="Count",
                y="Brand",
                orientation="h",
                text="Count",
                template="plotly_white",
                title="Top 10 brands by laptop count",
            )

            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=55, b=10),
            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with right:
        model_chart_col = MODEL_COL or LAPTOP_NAME_COL

        if model_chart_col:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">🧩 Popular Laptop Models</div>
                    <div class="section-subtitle">
                        Models with the highest number of records.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            model_counts = (
                filtered_df[model_chart_col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
                .sort_values()
                .reset_index()
            )

            model_counts.columns = [
                "Model",
                "Count",
            ]

            fig = px.bar(
                model_counts,
                x="Count",
                y="Model",
                orientation="h",
                text="Count",
                template="plotly_white",
                title="Top 10 laptop models",
            )

            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=55, b=10),
            )

            fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # ========================================================
    # PRICE BY BRAND + RAM
    # ========================================================

    left, right = st.columns(2)

    with left:
        if BRAND_COL:
            brand_price = (
                filtered_df.groupby(BRAND_COL)["Price"]
                .agg(["mean", "count"])
                .dropna()
                .query("count >= 3")
                .sort_values("mean", ascending=False)
                .head(10)
                .sort_values("mean")
                .reset_index()
            )

            if not brand_price.empty:
                st.markdown(
                    """
                    <div class="section-card">
                        <div class="section-title">💵 Average Price by Brand</div>
                        <div class="section-subtitle">
                            Brands with at least three matching laptops.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                brand_price["Average Price"] = brand_price["mean"]

                fig = px.bar(
                    brand_price,
                    x="Average Price",
                    y=BRAND_COL,
                    orientation="h",
                    text="Average Price",
                    template="plotly_white",
                    title="Average price by brand",
                )

                fig.update_layout(
                    height=410,
                    margin=dict(l=10, r=10, t=55, b=10),
                    xaxis_title="Average Price (₹)",
                    yaxis_title="Brand",
                )

                fig.update_traces(
                    texttemplate="₹%{x:,.0f}",
                    textposition="outside",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    with right:
        ram_chart = (
            filtered_df[["RAM_GB", "Price"]]
            .dropna()
            .groupby("RAM_GB")["Price"]
            .mean()
            .reset_index()
            .sort_values("RAM_GB")
        )

        if not ram_chart.empty:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">🧠 Average Price by RAM</div>
                    <div class="section-subtitle">
                        Average laptop price across available RAM capacities.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = px.line(
                ram_chart,
                x="RAM_GB",
                y="Price",
                markers=True,
                template="plotly_white",
                title="Average price by RAM capacity",
                labels={
                    "RAM_GB": "RAM (GB)",
                    "Price": "Average Price (₹)",
                },
            )

            fig.update_layout(
                height=410,
                margin=dict(l=10, r=10, t=55, b=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # ========================================================
    # SPECIFICATION SCATTERS
    # ========================================================

    c1, c2 = st.columns(2)

    with c1:
        ram_scatter = filtered_df[
            ["RAM_GB", "Price"]
        ].dropna()

        if not ram_scatter.empty:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">🧠 RAM vs Price</div>
                    <div class="section-subtitle">
                        Relationship between RAM capacity and laptop price.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = px.scatter(
                ram_scatter,
                x="RAM_GB",
                y="Price",
                opacity=0.58,
                template="plotly_white",
                title="RAM capacity vs price",
                labels={
                    "RAM_GB": "RAM (GB)",
                    "Price": "Price (₹)",
                },
            )

            fig.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=55, b=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with c2:
        ssd_scatter = filtered_df[
            ["SSD_GB", "Price"]
        ].dropna()

        if not ssd_scatter.empty:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">💾 SSD vs Price</div>
                    <div class="section-subtitle">
                        Relationship between SSD capacity and laptop price.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = px.scatter(
                ssd_scatter,
                x="SSD_GB",
                y="Price",
                opacity=0.58,
                template="plotly_white",
                title="SSD capacity vs price",
                labels={
                    "SSD_GB": "SSD (GB)",
                    "Price": "Price (₹)",
                },
            )

            fig.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=55, b=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    # Insight
    top_brand = None

    if BRAND_COL:
        brand_mode = (
            filtered_df[BRAND_COL]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not brand_mode.empty:
            top_brand = brand_mode.index[0]

    brand_text = (
        f"The most represented brand in the current selection is "
        f"<b>{top_brand}</b>."
        if top_brand
        else
        "Brand information is not available in the dataset."
    )

    st.markdown(
        f"""
        <div class="insight-box">
            <div class="insight-title">💡 Dashboard Insight</div>
            <div class="insight-text">
                {brand_text}
                The selected dataset contains <b>{len(filtered_df):,}</b>
                matching laptops with an average price of
                <b>{money(avg_price)}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2 — LAPTOP ANALYSIS
# ============================================================

elif page == "🔍 Laptop Analysis":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Filtered inventory</div>
            <div class="hero-title">🔍 Laptop Analysis</div>
            <div class="hero-subtitle">
                Search and compare laptop records using the filters on the left.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.warning(
            "No laptops match your filters. Adjust the filters in the sidebar."
        )
        st.stop()

    p = filtered_df["Price"].dropna()

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric("💻 Matching Laptops", f"{len(filtered_df):,}")

    with a2:
        st.metric("💰 Average Price", money(p.mean()))

    with a3:
        st.metric("📉 Minimum Price", money(p.min()))

    with a4:
        st.metric("📈 Maximum Price", money(p.max()))

    st.markdown("---")

    st.subheader("📋 Matching Laptop Records")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=560,
    )

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Laptops",
        data=csv,
        file_name="filtered_laptops.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# PAGE 3 — PRICE PREDICTION
# ============================================================

elif page == "🤖 Price Prediction":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Machine learning</div>
            <div class="hero-title">🤖 Laptop Price Prediction</div>
            <div class="hero-subtitle">
                Enter laptop specifications and estimate the expected market price.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner(
            "Preparing the compatible prediction model..."
        ):
            model = build_prediction_model(df_clean)
    except Exception as exc:
        st.error(
            "The prediction model could not be created from the dataset."
        )
        st.exception(exc)
        st.stop()

    st.success(
        "✅ Prediction model is ready and trained from the cleaned dataset."
    )

    with st.form("prediction_form"):

        st.subheader("Laptop Specifications")

        col1, col2 = st.columns(2)

        with col1:
            brand_values = get_unique_values(df, BRAND_COL)
            series_values = get_unique_values(df, SERIES_COL)
            os_values = get_unique_values(df, OS_COL)
            processor_values = get_unique_values(df, PROCESSOR_COL)
            gpu_values = get_unique_values(df, GPU_COL)

            brand = (
                st.selectbox("Brand", brand_values)
                if brand_values
                else st.text_input("Brand")
            )

            series = (
                st.selectbox("Series", series_values)
                if series_values
                else st.text_input("Series")
            )

            operating_system = (
                st.selectbox("Operating System", os_values)
                if os_values
                else st.text_input("Operating System")
            )

            processor = (
                st.selectbox("Processor", processor_values)
                if processor_values
                else st.text_input("Processor")
            )

            graphic_processor = (
                st.selectbox("Graphic Processor", gpu_values)
                if gpu_values
                else st.text_input("Graphic Processor")
            )

        with col2:
            ram_gb = st.number_input(
                "RAM (GB)",
                min_value=1.0,
                max_value=256.0,
                value=8.0,
                step=1.0,
            )

            ssd_gb = st.number_input(
                "SSD Capacity (GB)",
                min_value=0.0,
                max_value=8192.0,
                value=512.0,
                step=128.0,
            )

            hdd_gb = st.number_input(
                "HDD Capacity (GB)",
                min_value=0.0,
                max_value=8192.0,
                value=0.0,
                step=500.0,
            )

            display_inches = st.number_input(
                "Display Size (inches)",
                min_value=10.0,
                max_value=25.0,
                value=15.6,
                step=0.1,
            )

            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=0.5,
                max_value=10.0,
                value=1.8,
                step=0.1,
            )

            clock_speed = st.number_input(
                "Clock Speed (GHz)",
                min_value=0.5,
                max_value=10.0,
                value=2.5,
                step=0.1,
            )

            pixel_density = st.number_input(
                "Pixel Density",
                min_value=50.0,
                max_value=500.0,
                value=141.0,
                step=1.0,
            )

            refresh_rate = st.number_input(
                "Refresh Rate (Hz)",
                min_value=30.0,
                max_value=500.0,
                value=60.0,
                step=10.0,
            )

            cores = st.number_input(
                "Number of Cores",
                min_value=1.0,
                max_value=32.0,
                value=4.0,
                step=1.0,
            )

            battery_capacity = st.number_input(
                "Battery Capacity",
                min_value=0.0,
                max_value=200000.0,
                value=5000.0,
                step=100.0,
            )

        st.subheader("Additional Specifications")

        c1, c2, c3 = st.columns(3)

        with c1:
            ram_type_values = get_unique_values(
                df,
                RAM_TYPE_COL
            )

            ram_type = (
                st.selectbox(
                    "RAM Type",
                    ram_type_values
                )
                if ram_type_values
                else st.text_input(
                    "RAM Type",
                    value="DDR4"
                )
            )

        with c2:
            ssd_type_values = get_unique_values(
                df,
                SSD_TYPE_COL
            )

            ssd_type = (
                st.selectbox(
                    "SSD Type",
                    ssd_type_values
                )
                if ssd_type_values
                else st.text_input(
                    "SSD Type",
                    value="SSD"
                )
            )

        with c3:
            display_type_values = get_unique_values(
                df,
                DISPLAY_TYPE_COL
            )

            display_type = (
                st.selectbox(
                    "Display Type",
                    display_type_values
                )
                if display_type_values
                else st.text_input(
                    "Display Type"
                )
            )

        touch_values = get_unique_values(
            df,
            TOUCHSCREEN_COL
        )

        display_touchscreen = (
            st.selectbox(
                "Display Touchscreen",
                touch_values
            )
            if touch_values
            else st.selectbox(
                "Display Touchscreen",
                ["No", "Yes"]
            )
        )

        st.markdown("---")

        predict_button = st.form_submit_button(
            "🚀 Predict Laptop Price",
            use_container_width=True,
        )

    if predict_button:
        input_data = pd.DataFrame({
            "Brand": [brand],
            "Series": [series],
            "Operating System": [operating_system],
            "Processor": [processor],
            "Graphic Processor": [graphic_processor],
            "RAM Type": [ram_type],
            "SSD Type": [ssd_type],
            "Display Type": [display_type],
            "Display Touchscreen": [display_touchscreen],
            "ram_gb": [ram_gb],
            "ssd_gb": [ssd_gb],
            "hdd_gb": [hdd_gb],
            "display_inches": [display_inches],
            "weight_kg": [weight_kg],
            "clock_speed_ghz": [clock_speed],
            "pixel_density": [pixel_density],
            "refresh_rate_hz": [refresh_rate],
            "cores": [cores],
            "battery_capacity": [battery_capacity],
        })

        try:
            prediction = model.predict(input_data)[0]
            prediction = float(
                np.asarray(prediction).reshape(-1)[0]
            )

            if not np.isfinite(prediction):
                raise ValueError(
                    "The model returned an invalid price."
                )

            # Keep output sensible for a price dashboard.
            prediction = max(
                0.0,
                prediction
            )

            st.success(
                "✅ Laptop price prediction completed."
            )

            st.markdown(
                f"""
                <div style="
                    margin-top:18px;
                    padding:28px;
                    border-radius:20px;
                    background:
                        radial-gradient(circle at 80% 20%,
                            rgba(59,130,246,.12),
                            transparent 25%),
                        #ffffff;
                    border:1px solid #e5e7eb;
                    text-align:center;
                    box-shadow:0 14px 32px rgba(15,23,42,.08);
                ">
                    <div style="
                        color:#64748b;
                        font-size:14px;
                        font-weight:700;
                        letter-spacing:.8px;
                        text-transform:uppercase;
                    ">
                        Estimated Laptop Market Price
                    </div>
                    <div style="
                        color:#0f172a;
                        font-size:48px;
                        font-weight:850;
                        margin-top:8px;
                    ">
                        ₹{prediction:,.0f}
                    </div>
                    <div style="
                        color:#94a3b8;
                        font-size:13px;
                        margin-top:5px;
                    ">
                        Estimated from the cleaned training dataset
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)


# ============================================================
# PAGE 4 — DATASET
# ============================================================

elif page == "📋 Dataset":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Data explorer</div>
            <div class="hero-title">📋 Dataset Explorer</div>
            <div class="hero-subtitle">
                Inspect the original dataset and the cleaned analytics dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric("Raw Rows", f"{len(df_raw):,}")

    with d2:
        st.metric("Clean Rows", f"{len(df_clean):,}")

    with d3:
        st.metric(
            "Removed",
            f"{cleaning_stats['removed']:,}"
        )

    with d4:
        st.metric(
            "Columns",
            f"{len(df_raw.columns):,}"
        )

    st.markdown("---")

    view = st.radio(
        "Dataset view",
        [
            "Cleaned Analytics Data",
            "Original Raw Data",
        ],
        horizontal=True,
    )

    display_df = (
        df_clean.copy()
        if view == "Cleaned Analytics Data"
        else df_raw.copy()
    )

    search = st.text_input(
        "🔎 Search dataset",
        placeholder="Search laptop name, brand, processor, model..."
    )

    if search:
        mask = display_df.astype(str).apply(
            lambda column: column.str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)

        display_df = display_df.loc[mask]

    st.write(
        f"Showing **{len(display_df):,}** rows"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=560,
    )

    csv = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Current View",
        data=csv,
        file_name="laptop_dataset_view.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "💻 Laptop Price Intelligence • Streamlit • Pandas • Plotly • Scikit-learn"
)
