💻 Laptop Price Prediction Dashboard

A professional Laptop Price Prediction and Analytics Dashboard built with Streamlit, Pandas, Plotly, and Scikit-learn.

The application lets you explore laptop pricing data, clean suspicious price records for analytics, filter laptops by specifications, and predict the estimated price of a laptop from its specifications.

✨ Features

📊 Dashboard

The dashboard provides a clean overview of the laptop market with:

Total number of laptops analyzed

Average laptop price

Minimum laptop price

Maximum laptop price

Price distribution

Top laptop brands

Average price by brand

RAM vs. price analysis

SSD vs. price analysis

Data-quality summary

🧹 Data Cleaning

The application creates a separate cleaned dataset for dashboard analytics.

The cleaning process removes:

Missing prices

Zero and negative prices

Prices below ₹10,000

Prices above ₹5,00,000

Extreme statistical outliers using the IQR method

Important: The original laptop_dataset.csv is not modified.

🔍 Laptop Analysis

The analysis page allows users to:

Filter by brand

Filter by price range

Filter by RAM

View filtered laptop records

View filtered price distribution

Download filtered data as CSV

🤖 Laptop Price Prediction

The prediction page accepts laptop specifications such as:

Brand

Series

Operating System

Processor

Graphic Processor

RAM

SSD capacity

HDD capacity

Display size

Weight

Clock speed

Pixel density

Refresh rate

Number of CPU cores

Battery capacity

RAM type

SSD type

Display type

Touchscreen

The app trains a fresh prediction pipeline in the current Python/scikit-learn environment using the cleaned data.

The prediction pipeline uses:

SimpleImputer for missing values

OneHotEncoder for categorical values

ColumnTransformer for preprocessing

RandomForestRegressor for price prediction

A new compatible model is saved as:

laptop_price_prediction_model_compatible.pkl

The older laptop_price_prediction_model.pkl is not required for prediction by the current application.

📋 Dataset Explorer

The Dataset page provides:

Row count

Column count

Missing-value count

Dataset search

Complete dataset table

CSV export

🗂️ Project Structure

laptop_price_prediction/
│
├── app.py
├── laptop_dataset.csv
├── laptop_price_prediction_model.pkl
├── requirements.txt
├── README.md
│
└── laptop_price_prediction_model_compatible.pkl
    # Created automatically after the prediction model is trained

Optional files such as notebooks or Power BI exports may also be stored in the project directory.

🛠️ Technologies Used

Technology

Purpose

Python

Application and machine learning

Streamlit

Web dashboard

Pandas

Data processing

NumPy

Numerical operations

Plotly

Interactive charts

Scikit-learn

Machine learning and preprocessing

Joblib

Model serialization

⚙️ Installation

1. Clone or download the project

Open a terminal in the project folder.

cd laptop_price_prediction

2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate

3. Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

The provided requirements.txt contains:

streamlit
pandas
numpy
plotly
joblib
scikit-learn==1.8.0

4. Run the application

streamlit run app.py

Streamlit will display a local URL, normally similar to:

http://localhost:8501

Open that address in your browser.

📁 Required Files

The application expects the following files in the same folder as app.py:

laptop_dataset.csv

The main laptop dataset used for analytics and model training.

app.py

The Streamlit application.

requirements.txt

Python package dependencies.

The old model file:

laptop_price_prediction_model.pkl

may remain in the project folder, but the current application trains a fresh compatible model for prediction instead of relying on that older serialized model.

🧠 Machine Learning Workflow

The prediction workflow is:

Laptop CSV
    ↓
Column detection
    ↓
Feature extraction
    ↓
Price cleaning
    ↓
Categorical preprocessing
    ↓
Numeric preprocessing
    ↓
Random Forest Regression
    ↓
Laptop price prediction

Categorical features

Brand
Series
Operating System
Processor
Graphic Processor
RAM Type
SSD Type
Display Type
Display Touchscreen

Numeric features

ram_gb
ssd_gb
hdd_gb
display_inches
weight_kg
clock_speed_ghz
pixel_density
refresh_rate_hz
cores
battery_capacity

🧹 Price Cleaning Logic

For dashboard analytics, the application starts with:

Minimum allowed price: ₹10,000
Maximum allowed price: ₹5,00,000

It then applies an IQR-based outlier filter when enough valid records are available.

This means the dashboard statistics are based on a cleaner analytics view rather than blindly using suspicious raw values.

Example

A raw dataset may contain values such as:

-₹10,000
₹0
₹9,999,999

These values are excluded from the cleaned analytics view.

The original CSV remains unchanged.

📈 Dashboard Metrics

The Dashboard displays exactly four primary KPI values:

💻 Laptops Analyzed
💰 Average Price
📉 Minimum Price
📈 Maximum Price

These metrics are calculated from the cleaned analytics dataset.

🤖 Prediction Model

When the user opens the Price Prediction page:

The cleaned laptop dataset is prepared.

Categorical and numerical features are selected.

Missing values are handled.

Categorical features are one-hot encoded.

A RandomForestRegressor is trained.

The trained pipeline predicts the requested laptop price.

The trained model can be saved as:

laptop_price_prediction_model_compatible.pkl

This approach avoids depending on an incompatible older serialized model.

▶️ How to Use

Dashboard

Select:

📊 Dashboard

Use it to understand overall laptop pricing and specification trends.

Laptop Analysis

Select:

🔍 Laptop Analysis

Use the sidebar filters to narrow the dataset by brand, price, and RAM.

Price Prediction

Select:

🤖 Price Prediction

Enter laptop specifications and click:

🚀 Predict Laptop Price

Dataset

Select:

📋 Dataset

Search, inspect, and download the dataset.

⚠️ Troubleshooting

Error: Dataset not found

Make sure:

laptop_dataset.csv

is located in the same directory as:

app.py

Then restart Streamlit.

Error while installing packages

Run:

python -m pip install --upgrade pip

Then:

python -m pip install -r requirements.txt

Old _RemainderColsList model error

An older serialized scikit-learn model can be incompatible with the installed scikit-learn version.

The current application avoids relying on the old serialized model and creates a fresh prediction pipeline in the current environment.

Port already in use

Run Streamlit on another port:

streamlit run app.py --server.port 8502

Then open:

http://localhost:8502

Streamlit is still showing an old version

Stop the running process:

Ctrl + C

Then start the application again:

streamlit run app.py

💡 Notes

The raw dataset is preserved.

Dashboard cleaning is used for analytics and prediction training.

Prediction models are created in the current Python/scikit-learn environment.

The application uses Streamlit caching for data loading and model training.

The first visit to the Prediction page may take longer because the model is trained at that time.

👨‍💻 Author

Laptop Price Prediction Project

Built using:

Python
Streamlit
Pandas
NumPy
Plotly
Scikit-learn
Joblib

📄 License

Add your preferred license here, for example:

MIT License

If this project is being submitted for an academic or portfolio project, replace this section with your institution/project-specific license or submission details.