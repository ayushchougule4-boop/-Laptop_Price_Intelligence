
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
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "laptop_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "laptop_price_prediction_model.pkl"
)

# A fresh model trained by this app when the old .pkl is incompatible.
COMPATIBLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "laptop_price_prediction_model_compatible.pkl"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        text-align: center;
    }

    .title {
        font-size: 40px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(data):
    """Clean unnecessary spaces from column names."""

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return data


def find_column(data, possible_names):
    """
    Find a column using several possible names.
    Comparison is case-insensitive and ignores spaces,
    underscores, hyphens and brackets.
    """

    normalized = {}

    for column in data.columns:

        key = re.sub(
            r"[^a-z0-9]",
            "",
            str(column).lower()
        )

        normalized[key] = column

    for name in possible_names:

        key = re.sub(
            r"[^a-z0-9]",
            "",
            name.lower()
        )

        if key in normalized:
            return normalized[key]

    return None


def extract_number(series):
    """Extract the first unsigned numeric value from a specification column."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce"
    )


def extract_price(series):
    """Extract price while preserving a negative sign for validation."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)")[0],
        errors="coerce"
    )


def extract_ram(series):
    """Extract RAM value in GB."""

    values = (
        series.astype(str)
        .str.extract(r"(\d+(?:\.\d+)?)")[0]
    )

    return pd.to_numeric(values, errors="coerce")


def extract_storage(series):
    """
    Extract storage capacity.
    Converts TB to GB where possible.
    """

    text = series.astype(str).str.lower()

    numbers = pd.to_numeric(
        text.str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce"
    )

    result = numbers.copy()

    tb_mask = text.str.contains("tb", na=False)

    result.loc[tb_mask] = (
        result.loc[tb_mask] * 1024
    )

    return result



# ============================================================
# PRICE DATA CLEANING
# ============================================================

def clean_price_data(data, min_price=10000, max_price=500000):
    """
    Create a clean analytics view without modifying the raw dataset.

    Removes:
    - missing/non-positive prices
    - prices outside the selected practical range
    - extreme IQR statistical outliers
    """
    data = data.copy()

    if "Price" not in data.columns:
        return data, {
            "invalid": 0,
            "range_removed": 0,
            "outliers": 0,
            "removed": 0
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

    # Only use IQR when enough observations exist.
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

    return cleaned, {
        "invalid": int(invalid_mask.sum()),
        "range_removed": int(range_mask.sum()),
        "outliers": outliers,
        "removed": int(len(data) - len(cleaned))
    }


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    data = clean_column_names(data)

    if data.empty:
        raise ValueError("The dataset is empty.")

    return data


# ============================================================
# PREPARE DATA
# ============================================================

@st.cache_data
def prepare_data(data):

    data = data.copy()

    # --------------------------------------------------------
    # Find important columns
    # --------------------------------------------------------

    ram_col = find_column(
        data,
        [
            "RAM",
            "Ram",
            "RAM (GB)",
            "RAM GB",
            "Memory"
        ]
    )

    storage_col = find_column(
        data,
        [
            "SSD Capacity",
            "SSD",
            "SSD Capacity (GB)"
        ]
    )

    hdd_col = find_column(
        data,
        [
            "HDD Capacity",
            "HDD",
            "HDD Capacity (GB)"
        ]
    )

    display_col = find_column(
        data,
        [
            "Display Size",
            "Screen Size",
            "Display Size (inches)",
            "Screen Size (inches)"
        ]
    )

    weight_col = find_column(
        data,
        [
            "Weight",
            "Weight (kg)",
            "Laptop Weight"
        ]
    )

    price_col = find_column(
        data,
        [
            "Price (Rs)",
            "Price",
            "Price (₹)",
            "Price Rs",
            "Selling Price"
        ]
    )

    # --------------------------------------------------------
    # Create derived columns only when source exists
    # --------------------------------------------------------

    if ram_col is not None:

        data["RAM_GB"] = extract_ram(
            data[ram_col]
        )

    else:

        data["RAM_GB"] = np.nan

    if storage_col is not None:

        data["SSD_GB"] = extract_storage(
            data[storage_col]
        )

    else:

        data["SSD_GB"] = np.nan

    if hdd_col is not None:

        data["HDD_GB"] = extract_storage(
            data[hdd_col]
        )

    else:

        data["HDD_GB"] = np.nan

    if display_col is not None:

        data["Display_Inches"] = extract_number(
            data[display_col]
        )

    else:

        data["Display_Inches"] = np.nan

    if weight_col is not None:

        data["Weight_KG"] = extract_number(
            data[weight_col]
        )

    else:

        data["Weight_KG"] = np.nan

    if price_col is not None:

        data["Price"] = extract_price(
            data[price_col]
        )

    else:

        data["Price"] = np.nan

    # Additional numeric features used by the prediction model.
    clock_col = find_column(
        data, [
            "Clock Speed", "Clock Speed (GHz)", "Clock Speed GHz",
            "CPU Clock Speed", "Processor Clock Speed"
        ]
    )
    pixel_col = find_column(
        data, ["Pixel Density", "Pixel Density (PPI)", "PPI", "Pixels Per Inch"]
    )
    refresh_col = find_column(
        data, ["Refresh Rate", "Refresh Rate (Hz)", "RefreshRate", "Screen Refresh Rate"]
    )
    cores_col = find_column(
        data, ["Number of Cores", "Cores", "CPU Cores", "Processor Cores"]
    )
    battery_col = find_column(
        data, [
            "Battery Capacity", "Battery Capacity (mAh)",
            "Battery (mAh)", "Battery Capacity (Wh)"
        ]
    )

    data["clock_speed_ghz"] = (
        extract_number(data[clock_col]) if clock_col else np.nan
    )
    data["pixel_density"] = (
        extract_number(data[pixel_col]) if pixel_col else np.nan
    )
    data["refresh_rate_hz"] = (
        extract_number(data[refresh_col]) if refresh_col else np.nan
    )
    data["cores"] = (
        extract_number(data[cores_col]) if cores_col else np.nan
    )
    data["battery_capacity"] = (
        extract_number(data[battery_col]) if battery_col else np.nan
    )

    return data


# ============================================================
# PREDICTION MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def build_prediction_model(training_frame):
    """Train a fresh prediction model in the current Python/sklearn environment.

    The old laptop_price_prediction_model.pkl is intentionally NOT loaded,
    because it was serialized with an incompatible scikit-learn version.
    """

    feature_columns = [
        "Brand", "Series", "Operating System", "Processor",
        "Graphic Processor", "RAM Type", "SSD Type", "Display Type",
        "Display Touchscreen", "ram_gb", "ssd_gb", "hdd_gb",
        "display_inches", "weight_kg", "clock_speed_ghz",
        "pixel_density", "refresh_rate_hz", "cores", "battery_capacity"
    ]

    categorical_features = [
        "Brand", "Series", "Operating System", "Processor",
        "Graphic Processor", "RAM Type", "SSD Type", "Display Type",
        "Display Touchscreen"
    ]

    numeric_features = [
        "ram_gb", "ssd_gb", "hdd_gb", "display_inches", "weight_kg",
        "clock_speed_ghz", "pixel_density", "refresh_rate_hz",
        "cores", "battery_capacity"
    ]

    raw = training_frame.copy()
    train = pd.DataFrame(index=raw.index)

    categorical_aliases = {
        "Brand": ["Brand", "Manufacturer"],
        "Series": ["Series", "Model Series"],
        "Operating System": ["Operating System", "OS"],
        "Processor": ["Processor", "CPU"],
        "Graphic Processor": ["Graphic Processor", "Graphics Processor", "GPU"],
        "RAM Type": ["RAM Type", "Memory Type"],
        "SSD Type": ["SSD Type", "Storage Type"],
        "Display Type": ["Display Type", "Screen Type"],
        "Display Touchscreen": ["Display Touchscreen", "Touchscreen", "Touch Screen"]
    }

    for target_col, aliases in categorical_aliases.items():
        source_col = find_column(raw, aliases)
        if source_col is not None:
            train[target_col] = raw[source_col]
        else:
            train[target_col] = "Unknown"

    for col in numeric_features:
        train[col] = raw[col] if col in raw.columns else np.nan

    train["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    train = train[feature_columns + ["Price"]]
    train = train.replace([np.inf, -np.inf], np.nan)
    train = train[train["Price"].between(10_000, 500_000)].copy()

    if len(train) < 20:
        raise ValueError(
            "At least 20 valid laptop records are required to train the prediction model."
        )

    for col in categorical_features:
        train[col] = train[col].fillna("Unknown").astype(str)

    for col in numeric_features:
        train[col] = pd.to_numeric(train[col], errors="coerce")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_features
            )
        ],
        remainder="drop"
    )

    regressor = RandomForestRegressor(
        n_estimators=180,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    pipeline.fit(train[feature_columns], train["Price"])

    # Save a NEW model produced by the current environment.
    try:
        joblib.dump(pipeline, COMPATIBLE_MODEL_PATH)
    except Exception:
        pass

    return pipeline

# ============================================================
# INITIALIZE
# ============================================================

try:
    df_raw = load_data()
    df = prepare_data(df_raw)
except Exception as e:
    st.error("Unable to load the laptop dataset.")
    st.exception(e)
    st.stop()

# The model is built only on the Prediction page.
# This keeps Dashboard / Analysis completely independent of model loading.
model = None

# Cleaned dataframe used for dashboard analytics.
# The original df remains unchanged for the Dataset page.
df_clean, cleaning_stats = clean_price_data(
    df,
    min_price=10_000,
    max_price=500_000
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💻 Laptop Price Prediction")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Dashboard",
        "🔍 Laptop Analysis",
        "🤖 Price Prediction",
        "📋 Dataset"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    f"""
    **Dataset**

    Rows: {len(df):,}

    Columns: {len(df.columns):,}
    """
)

st.sidebar.caption(
    "🤖 The prediction model is created only on the Prediction page."
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        """
        <div style="
            padding:28px 30px;
            border-radius:18px;
            background:linear-gradient(135deg,#111827,#334155);
            color:white;
            margin-bottom:22px;
        ">
            <div style="font-size:38px;font-weight:800;">
                💻 Laptop Price Intelligence
            </div>
            <div style="font-size:16px;color:#dbeafe;margin-top:7px;">
                Clean pricing data, market trends and specification insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    valid_prices = df_clean["Price"].dropna()

    if valid_prices.empty:
        st.warning("No valid prices remain after cleaning.")
        st.stop()

    average_price = valid_prices.mean()
    median_price = valid_prices.median()
    minimum_price = valid_prices.min()
    maximum_price = valid_prices.max()

    # KPI cards — exactly the four requested dashboard values
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💻 Laptops Analyzed", f"{len(df_clean):,}")

    with col2:
        st.metric("💰 Average Price", f"₹{average_price:,.0f}")

    with col3:
        st.metric("📉 Minimum Price", f"₹{minimum_price:,.0f}")

    with col4:
        st.metric("📈 Maximum Price", f"₹{maximum_price:,.0f}")

    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:14px;
            padding:15px 18px;
            margin:18px 0;
        ">
            <b>🧹 Data Quality</b><br>
            <span style="color:#6b7280;font-size:13px;">
                {cleaning_stats["removed"]:,} records were excluded from dashboard
                analytics because of invalid prices, prices outside ₹10,000–₹5,00,000,
                or extreme statistical outliers. Your original CSV is not modified.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Price distribution
    st.subheader("💰 Clean Price Distribution")

    fig = px.histogram(
        df_clean,
        x="Price",
        nbins=35,
        marginal="box",
        title="Laptop prices after data cleaning",
        labels={"Price": "Laptop Price (₹)"}
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Brand analysis
    brand_col = find_column(df_clean, ["Brand", "Manufacturer"])

    if brand_col is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏷️ Top Laptop Brands")

            brand_counts = (
                df_clean[brand_col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
                .sort_values()
                .reset_index()
            )
            brand_counts.columns = ["Brand", "Count"]

            fig = px.bar(
                brand_counts,
                x="Count",
                y="Brand",
                orientation="h",
                text="Count",
                title="Top 10 brands by number of laptops"
            )
            fig.update_layout(
                template="plotly_white",
                height=420
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💵 Average Price by Brand")

            brand_price = (
                df_clean.groupby(brand_col)["Price"]
                .agg(["mean", "count"])
                .dropna()
                .query("count >= 3")
                .sort_values("mean", ascending=False)
                .head(10)
                .sort_values("mean")
                .reset_index()
            )

            brand_price["Average Price"] = brand_price["mean"]

            fig = px.bar(
                brand_price,
                x="Average Price",
                y=brand_col,
                orientation="h",
                text="Average Price",
                title="Brands with at least 3 laptops"
            )
            fig.update_layout(
                template="plotly_white",
                height=420,
                xaxis_title="Average Price (₹)",
                yaxis_title="Brand"
            )
            fig.update_traces(
                texttemplate="₹%{x:,.0f}",
                textposition="outside"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Specification insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 RAM vs Price")

        chart_data = df_clean[["RAM_GB", "Price"]].dropna()

        if not chart_data.empty:
            fig = px.scatter(
                chart_data,
                x="RAM_GB",
                y="Price",
                opacity=0.65,
                title="RAM capacity compared with price",
                labels={
                    "RAM_GB": "RAM (GB)",
                    "Price": "Price (₹)"
                }
            )
            fig.update_layout(
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("RAM data is not available.")

    with col2:
        st.subheader("💾 SSD vs Price")

        chart_data = df_clean[["SSD_GB", "Price"]].dropna()

        if not chart_data.empty:
            fig = px.scatter(
                chart_data,
                x="SSD_GB",
                y="Price",
                opacity=0.65,
                title="SSD capacity compared with price",
                labels={
                    "SSD_GB": "SSD Capacity (GB)",
                    "Price": "Price (₹)"
                }
            )
            fig.update_layout(
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SSD data is not available.")


# ============================================================
# PAGE 2 — LAPTOP ANALYSIS
# ============================================================

elif page == "🔍 Laptop Analysis":

    st.title("🔍 Laptop Analysis")

    st.markdown(
        "Use the filters below to explore the dataset."
    )

    st.sidebar.subheader("Filters")

    filtered_df = df_clean.copy()

    # --------------------------------------------------------
    # BRAND FILTER
    # --------------------------------------------------------

    brand_col = find_column(
        df_clean,
        ["Brand", "Manufacturer"]
    )

    if brand_col is not None:

        brands = sorted(
            df[brand_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_brands = st.sidebar.multiselect(
            "Brand",
            brands
        )

        if selected_brands:

            filtered_df = filtered_df[
                filtered_df[brand_col]
                .astype(str)
                .isin(selected_brands)
            ]

    # --------------------------------------------------------
    # PRICE FILTER
    # --------------------------------------------------------

    if df["Price"].notna().any():

        min_price = int(
            df["Price"]
            .dropna()
            .min()
        )

        max_price = int(
            df["Price"]
            .dropna()
            .max()
        )

        if min_price < max_price:

            price_range = st.sidebar.slider(
                "Price Range (₹)",
                min_value=min_price,
                max_value=max_price,
                value=(
                    min_price,
                    max_price
                )
            )

            filtered_df = filtered_df[
                filtered_df["Price"].between(
                    price_range[0],
                    price_range[1]
                )
            ]

    # --------------------------------------------------------
    # RAM FILTER
    # --------------------------------------------------------

    if df["RAM_GB"].notna().any():

        ram_values = sorted(
            df["RAM_GB"]
            .dropna()
            .unique()
        )

        selected_ram = st.sidebar.multiselect(
            "RAM (GB)",
            ram_values
        )

        if selected_ram:

            filtered_df = filtered_df[
                filtered_df["RAM_GB"]
                .isin(selected_ram)
            ]

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Filtered Laptops",
            f"{len(filtered_df):,}"
        )

    with col2:

        if filtered_df["Price"].notna().any():

            st.metric(
                "Average Price",
                f"₹{filtered_df['Price'].mean():,.0f}"
            )

    with col3:

        if filtered_df["Price"].notna().any():

            st.metric(
                "Maximum Price",
                f"₹{filtered_df['Price'].max():,.0f}"
            )

    st.markdown("---")

    # --------------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------------

    if filtered_df["Price"].notna().any():

        fig = px.histogram(
            filtered_df,
            x="Price",
            nbins=30,
            title="Filtered Laptop Price Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader("📋 Filtered Laptops")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv,
        file_name="filtered_laptops.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 3 — PRICE PREDICTION
# ============================================================

elif page == "🤖 Price Prediction":

    st.markdown(
        """
        <div style="padding:28px 30px;border-radius:18px;
            background:linear-gradient(135deg,#111827,#334155);
            color:white;margin-bottom:22px;">
            <div style="font-size:38px;font-weight:800;">
                🤖 Laptop Price Prediction
            </div>
            <div style="font-size:16px;color:#dbeafe;margin-top:7px;">
                Enter specifications and estimate the laptop price.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        with st.spinner("Training prediction model from the cleaned laptop data..."):
            model = build_prediction_model(df_clean)
    except Exception as e:
        st.error("Prediction model could not be created from the dataset.")
        st.exception(e)
        st.stop()

    st.success(
        "✅ Prediction model is ready. It was trained in the current Python "
        "environment using the cleaned laptop data."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # IDENTIFY CATEGORICAL COLUMNS
    # --------------------------------------------------------

    brand_col = find_column(
        df,
        ["Brand", "Manufacturer"]
    )

    series_col = find_column(
        df,
        ["Series", "Model Series"]
    )

    os_col = find_column(
        df,
        ["Operating System", "OS"]
    )

    processor_col = find_column(
        df,
        ["Processor", "CPU"]
    )

    gpu_col = find_column(
        df,
        ["Graphic Processor", "Graphics Processor", "GPU"]
    )

    ram_type_col = find_column(
        df,
        ["RAM Type", "Memory Type"]
    )

    ssd_type_col = find_column(
        df,
        ["SSD Type", "Storage Type"]
    )

    display_type_col = find_column(
        df,
        ["Display Type", "Screen Type"]
    )

    touchscreen_col = find_column(
        df,
        [
            "Display Touchscreen",
            "Touchscreen",
            "Touch Screen"
        ]
    )

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    with st.form("prediction_form"):

        st.subheader("Laptop Specifications")

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # CATEGORICAL INPUTS
        # ----------------------------------------------------

        with col1:

            if brand_col is not None:

                brand_values = sorted(
                    df[brand_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                brand = st.selectbox(
                    "Brand",
                    brand_values
                )

            else:

                brand = st.text_input(
                    "Brand"
                )

            if series_col is not None:

                series_values = sorted(
                    df[series_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                series = st.selectbox(
                    "Series",
                    series_values
                )

            else:

                series = st.text_input(
                    "Series"
                )

            if os_col is not None:

                os_values = sorted(
                    df[os_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                operating_system = st.selectbox(
                    "Operating System",
                    os_values
                )

            else:

                operating_system = st.text_input(
                    "Operating System"
                )

            if processor_col is not None:

                processor_values = sorted(
                    df[processor_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                processor = st.selectbox(
                    "Processor",
                    processor_values
                )

            else:

                processor = st.text_input(
                    "Processor"
                )

            if gpu_col is not None:

                gpu_values = sorted(
                    df[gpu_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                graphic_processor = st.selectbox(
                    "Graphic Processor",
                    gpu_values
                )

            else:

                graphic_processor = st.text_input(
                    "Graphic Processor"
                )

        # ----------------------------------------------------
        # SECOND COLUMN
        # ----------------------------------------------------

        with col2:

            ram_gb = st.number_input(
                "RAM (GB)",
                min_value=1.0,
                max_value=256.0,
                value=8.0,
                step=1.0
            )

            ssd_gb = st.number_input(
                "SSD Capacity (GB)",
                min_value=0.0,
                max_value=8192.0,
                value=512.0,
                step=128.0
            )

            hdd_gb = st.number_input(
                "HDD Capacity (GB)",
                min_value=0.0,
                max_value=8192.0,
                value=0.0,
                step=500.0
            )

            display_inches = st.number_input(
                "Display Size (inches)",
                min_value=10.0,
                max_value=25.0,
                value=15.6,
                step=0.1
            )

            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=0.5,
                max_value=10.0,
                value=1.8,
                step=0.1
            )

            clock_speed = st.number_input(
                "Clock Speed (GHz)",
                min_value=0.5,
                max_value=10.0,
                value=2.5,
                step=0.1
            )

            pixel_density = st.number_input(
                "Pixel Density",
                min_value=50.0,
                max_value=500.0,
                value=141.0,
                step=1.0
            )

            refresh_rate = st.number_input(
                "Refresh Rate (Hz)",
                min_value=30.0,
                max_value=500.0,
                value=60.0,
                step=10.0
            )

            cores = st.number_input(
                "Number of Cores",
                min_value=1.0,
                max_value=32.0,
                value=4.0,
                step=1.0
            )

            battery_capacity = st.number_input(
                "Battery Capacity",
                min_value=0.0,
                max_value=200000.0,
                value=5000.0,
                step=100.0
            )

        # ----------------------------------------------------
        # OTHER CATEGORICAL INPUTS
        # ----------------------------------------------------

        st.subheader("Additional Specifications")

        col3, col4, col5 = st.columns(3)

        with col3:

            if ram_type_col is not None:

                ram_type_values = sorted(
                    df[ram_type_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                ram_type = st.selectbox(
                    "RAM Type",
                    ram_type_values
                )

            else:

                ram_type = st.text_input(
                    "RAM Type",
                    value="DDR4"
                )

        with col4:

            if ssd_type_col is not None:

                ssd_type_values = sorted(
                    df[ssd_type_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                ssd_type = st.selectbox(
                    "SSD Type",
                    ssd_type_values
                )

            else:

                ssd_type = st.text_input(
                    "SSD Type",
                    value="SSD"
                )

        with col5:

            if display_type_col is not None:

                display_type_values = sorted(
                    df[display_type_col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                display_type = st.selectbox(
                    "Display Type",
                    display_type_values
                )

            else:

                display_type = st.text_input(
                    "Display Type"
                )

        if touchscreen_col is not None:

            touchscreen_values = sorted(
                df[touchscreen_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            display_touchscreen = st.selectbox(
                "Display Touchscreen",
                touchscreen_values
            )

        else:

            display_touchscreen = st.selectbox(
                "Display Touchscreen",
                ["No", "Yes"]
            )

        st.markdown("---")

        predict_button = st.form_submit_button(
            "🚀 Predict Laptop Price",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        # ----------------------------------------------------
        # Create input using the feature names expected by the model
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
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
                "battery_capacity": [battery_capacity]
            }
        )

        try:

            # If the saved estimator exposes feature names, use exactly
            # the columns it was trained with and preserve their order.
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
                missing_features = [
                    col for col in expected_features
                    if col not in input_data.columns
                ]

                if missing_features:
                    raise ValueError(
                        "The saved model expects feature(s) that are not "
                        f"provided by this app: {missing_features}. "
                        "Retrain the model with the same feature schema "
                        "or update the prediction form."
                    )

                input_for_model = input_data[expected_features]
            else:
                input_for_model = input_data

            prediction = model.predict(
                input_for_model
            )[0]

            # Convert NumPy scalar predictions to a normal Python number.
            prediction = float(np.asarray(prediction).reshape(-1)[0])

            if not np.isfinite(prediction):
                raise ValueError(
                    "The model returned an invalid (NaN/Infinity) prediction."
                )

            st.success(
                "Prediction completed successfully!"
            )

            st.markdown(
                f"""
                <div style="
                    padding:30px;
                    border-radius:15px;
                    background-color:#f5f5f5;
                    text-align:center;
                    margin-top:20px;
                ">
                    <h2>Estimated Laptop Price</h2>
                    <h1>₹{prediction:,.0f}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)

            st.info(
                "The new model is trained from this dataset. If prediction still "
                "fails, check that the selected values and numeric specifications "
                "are valid for the dataset."
            )


# ============================================================
# PAGE 4 — DATASET
# ============================================================

elif page == "📋 Dataset":

    st.title("📋 Dataset Explorer")

    st.write(
        "Explore the complete laptop dataset."
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{len(df.columns):,}"
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = st.text_input(
        "🔎 Search dataset"
    )

    display_df = df.copy()

    if search:

        mask = display_df.astype(
            str
        ).apply(
            lambda column:
            column.str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ).any(axis=1)

        display_df = display_df[
            mask
        ]

    st.write(
        f"Showing {len(display_df):,} rows"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=550
    )

    # --------------------------------------------------------
    # DOWNLOAD COMPLETE DATASET
    # --------------------------------------------------------

    csv = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset",
        data=csv,
        file_name="laptop_dataset_export.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "💻 Laptop Price Prediction Dashboard | "
    "Built with Streamlit, Pandas, Scikit-learn and Plotly"
)




# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "💻 Laptop Price Prediction Dashboard | "
    "Built with Streamlit, Pandas, Scikit-learn and Plotly"
)

