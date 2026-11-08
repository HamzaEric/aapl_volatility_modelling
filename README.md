# AAPL Stock Volatility Forecasting Using a GARCH(1,1) Model

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff?style=for-the-badge&logo=matplotlib&logoColor=black" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/SQLite-07405e?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Render-%46E3B7?style=for-the-badge&logo=render&logoColor=black" alt="Render">
</p>

A production-ready web application designed to train **GARCH(p, q)** time-series models on real-time financial market data (using `yfinance`), persist historical lookbacks in SQLite, and serve volatility forecasts through a decoupled **FastAPI** backend and a customized **Streamlit** dashboard.

## 🚀 Project Architecture

```
volatility_fastapi_app/
├── .venv/                  # Local virtual environment (ignored by Git)
├── images/                 
│   └── apple_3.jpg         # Custom background asset with dark-overlay CSS
├── models/                 # Saved serialized `.pkl` models (ignored by Git)
├── .gitignore              # Excludes caches, venv, models, and databases
├── app.py                  # Streamlit Frontend Dashboard (Wide layout, JetBrains Mono, Matplotlib)
├── config.py               # Shared path resolutions and environment configurations
├── data.py                 # SQLite database integration and lookback query sorting
├── main.py                 # FastAPI backend endpoints (/fit and /predict)
├── model.py                # GARCH model wrapper using the `arch` library
├── yf_loader.py            # Yahoo Finance data ingestion handler
├── requirements.txt        # Production dependencies list
└── stocks.sqlite           # Local SQLite storage for historical data
```

## 🛠️ Tech Stack Breakdown

* **Frontend:** Streamlit, Matplotlib, Custom CSS (JetBrains Mono & Responsive Dark Styling)
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Modeling & Finance:** `arch` (GARCH modeling), `yfinance`, Pandas, NumPy, Scikit-learn, Joblib
* **Database & Storage:** SQLite with robust `pathlib` cross-platform resolution
* **Cloud Infrastructure:** Render (Backend API) & Streamlit Community Cloud (Frontend Dashboard)

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/volatility-forecaster.git
   cd volatility_fastapi_app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI Backend:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8008 --reload
   ```

5. **Run the Streamlit Frontend (in a separate terminal):**
   ```bash
   streamlit run app.py
   ```

## 🌐 API Endpoints

### 1. Fit Model (`POST /fit`)
* **Payload:**
  ```json
  {
    "ticker": "AAPL",
    "use_new_data": true,
    "n_observations": 2000,
    "p": 1,
    "q": 1
  }
  ```
* **Response:** Trains the GARCH model on historical market returns and saves the fitted weights.

### 2. Predict Volatility (`POST /predict`)
* **Payload:**
  ```json
  {
    "ticker": "AAPL",
    "n_days": 5
  }
  ```
* **Response:** Returns future conditional volatility forecasts mapped by date.

## ☁️ Cloud Deployment Guide

### Backend (Render)
1. Create a new **Web Service** on [Render](https://render.com/) linked to your GitHub repository.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 10000`

### Frontend (Streamlit Community Cloud)
1. Deploy your app on [Streamlit Cloud](https://share.streamlit.io/) targeting `app.py`.
2. Under **Advanced Settings → Secrets**, link your live backend URL:
   ```toml
   API_URL = "https://your-render-backend-url.onrender.com"
   ```
