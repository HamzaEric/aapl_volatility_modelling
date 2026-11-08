import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
import os
from pathlib import Path

# --- CLOUD DEPLOYMENT API ENDPOINTS ---
# Gets the API URL from the environment, defaulting to localhost for local testing
API_BASE_URL = os.getenv("API_URL", "http://localhost:8008")

FIT_URL = f"{API_BASE_URL}/fit"
PREDICT_URL = f"{API_BASE_URL}/predict"

# Set layout to "wide"
st.set_page_config(page_title="Volatility Forecaster", layout="wide")

# --- AUTO WAKE-UP / HEALTH PING FOR RENDER ---
@st.cache_resource
def wake_up_backend():
    """Silently pings the backend on app load to wake up Render's free tier from sleep."""
    try:
        requests.get(API_BASE_URL, timeout=3)
    except Exception:
        pass

# Trigger wake-up call in the background
wake_up_backend()

# --- BACKGROUND IMAGE LOGIC ---
def set_background(image_relative_path):
    # Dynamically resolve the absolute path relative to app.py
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / image_relative_path
    
    if image_path.exists():
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        # Injects the background image with a 70% dark overlay for readability
        css = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                              url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    else:
        st.warning(f"Background image not found at absolute path: {image_path}. Make sure it's pushed to GitHub.")

# Call the function to set the background
set_background("images/apple_3.jpg")
# -----------------------------------

# Inject custom CSS (with a fix to prevent the "arrow_right" text overlap bug)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"], * {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* FIX: Restore Streamlit's default icon fonts so arrows don't turn into text */
    .material-symbols-rounded, .material-icons, .stIcon, [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }
    </style>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ **Hosting Notice:** This application is hosted on free cloud tiers. "
    "If the backend is sleeping, the first request may take up to 40 seconds to spin up the server."
)

st.title("AAPL Stock Volatility Forecasting Using a GARCH(1,1) Model")
st.write("Train a GARCH model and forecast future volatility for any stock.")

# --- MAIN DASHBOARD INPUTS ---
st.header("Parameters")

# Put the main inputs side by side
input_col1, input_col2 = st.columns(2)
with input_col1:
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
with input_col2:
    n_days = st.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=5)

with st.expander("Advanced Model Settings"):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        n_observations = st.number_input("Lookback Days", min_value=500, max_value=5000, value=2000)
    with adv_col2:
        p = st.number_input("GARCH p (Lag order)", min_value=1, max_value=5, value=1)
    with adv_col3:
        q = st.number_input("GARCH q (Moving average)", min_value=1, max_value=5, value=1)

st.divider()

# --- ACTION BUTTONS ---

# Button 1: Train the Model
current_date = datetime.now().strftime("%b %d, %Y")
if st.button(f"1. Train Model Till {current_date}", use_container_width=True):
    with st.spinner(f"Fetching data and training model for {ticker} (if Render is sleeping, this may take ~30s)..."):
        payload = {
            "ticker": ticker,
            "use_new_data": True,
            "n_observations": n_observations,
            "p": p,
            "q": q
        }
        try:
            response = requests.post(FIT_URL, json=payload, timeout=60)
            data = response.json()

            if data.get("success"):
                st.success(f"Model trained successfully for {ticker}!")
            else:
                st.error(f"Error: {data.get('message')}")
        except requests.exceptions.ConnectionError:
            st.error(f"Failed to connect to the backend at {API_BASE_URL}. Is the server running?")
        except requests.exceptions.Timeout:
            st.error("The request timed out. Render's free tier was likely waking up from sleep. Please try clicking Train again!")

# Button 2: Get Predictions (Stacked below Button 1)
if st.button("2. Forecast Volatility", use_container_width=True):
    with st.spinner(f"Generating {n_days}-day forecast..."):
        payload = {
            "ticker": ticker,
            "n_days": n_days
        }
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=30)
            data = response.json()

            if data.get("success"):
                forecast_dict = data.get("forecast", {})

                # Convert JSON response to a Pandas DataFrame for the chart
                df = pd.DataFrame({
                    "Date": pd.to_datetime(list(forecast_dict.keys())),
                    "Predicted Volatility": list(forecast_dict.values())
                }).set_index("Date")

                st.subheader(f"{n_days}-Day Volatility Forecast for {ticker}")

                # --- MATPLOTLIB IMPLEMENTATION ---
                fig, ax = plt.subplots(figsize=(12, 5))

                # Make the Matplotlib background transparent so it blends with your app background
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)

                # Plot the line with markers
                ax.plot(df.index, df["Predicted Volatility"],
                        color="#FF4B4B", marker="o", linestyle="-", linewidth=2, markersize=6)

                # Styling the graph (white text to contrast with dark background)
                ax.set_title(f"{ticker} Forecast", fontfamily="monospace", fontsize=14, pad=15, color="white")
                ax.set_xlabel("Date", fontfamily="monospace", fontsize=12, color="white")
                ax.set_ylabel("Volatility", fontfamily="monospace", fontsize=12, color="white")

                # Color the axes and ticks white
                ax.tick_params(colors="white")
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # FIX: Force x-axis to show only dates (YYYY-MM-DD) and tick once per day
                ax.xaxis.set_major_locator(mdates.DayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

                # Add a faint grid for readability
                ax.grid(True, linestyle="--", alpha=0.3, color="lightgray")

                # Auto-format the dates on the x-axis so they don't overlap
                fig.autofmt_xdate()

                # Render the matplotlib figure in Streamlit
                st.pyplot(fig)
                # ---------------------------------

                # Show raw data below the chart
                with st.expander("View Raw Data"):
                    st.dataframe(df)

            else:
                st.warning(f"Failed: {data.get('message')} Make sure to Train the model first!")
        except requests.exceptions.ConnectionError:
            st.error(f"Failed to connect to the backend at {API_BASE_URL}. Is the server running?")
        except requests.exceptions.Timeout:
            st.error("The request timed out while waiting for Render. Please try again.")
