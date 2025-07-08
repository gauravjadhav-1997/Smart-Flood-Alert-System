#=============================================================================
# --- Smart Flood Warning System ---
#=============================================================================

# --- Dependency check (run only once per session) ---
import importlib.util
import subprocess
import sys

def is_package_installed(import_name):
    """Checks if a package is installed."""
    return importlib.util.find_spec(import_name) is not None

# A dictionary to map package names to their import names if they differ
PACKAGE_IMPORT_MAP = {
    "scikit-learn": "sklearn",
    "python-dotenv": "dotenv"
}

REQUIRED_PACKAGES = [
    "streamlit", "pandas", "scikit-learn", "plotly", "geopy", "python-dotenv", "twilio", "requests"
]

import streamlit as st

# This block checks for and installs missing packages.
if "checked_dependencies" not in st.session_state:
    missing_packages = []
    for pkg in REQUIRED_PACKAGES:
        import_name = PACKAGE_IMPORT_MAP.get(pkg, pkg)
        if not is_package_installed(import_name):
            if pkg not in missing_packages:
                missing_packages.append(pkg)

    if missing_packages:
        st.info(f"Installing required packages: {', '.join(missing_packages)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            st.success("All missing packages have been installed. Please refresh the page.")
            st.session_state.checked_dependencies = True
            st.stop()
        except Exception as e:
            st.error(f"Failed to install packages: {e}")
            st.stop()
    else:
        st.session_state.checked_dependencies = True
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import requests
from geopy.geocoders import Nominatim
from dotenv import load_dotenv, find_dotenv
from twilio.rest import Client

# Load environment variables for Twilio
load_dotenv(find_dotenv("Your Twilio Account Credentials.env"))


# =============================================================================
# --- PAGE CONFIGURATION & THEME ---
# =============================================================================
st.set_page_config(
    page_title="Smart Flood Alert System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sky-Blue Theme (Light & Dark)
st.markdown("""
<style>
    :root {
        --primary-color: #1c83e1;
        --background-color: #F0F8FF;
        --secondary-background-color: #E0F0FF;
        --text-color: #0D1B2A;
        --font: "sans serif";
    }

    [data-theme="dark"] {
        --primary-color: #58A6FF;
        --background-color: #0D1B2A;
        --secondary-background-color: #1E3A5F;
        --text-color: #E0F0FF;
    }

    /* General Body and Font */
    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Main content area */
    .main .block-container {
        background-color: var(--background-color);
    }
    
    /* Increase font size for main content text */
    .main .block-container p, 
    .main .block-container li,
    .main .block-container div[data-testid="stMarkdown"] > div:first-child {
        font-size: 1.1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: var(--background-color);
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: var(--text-color);
        color: var(--secondary-background-color);
    }

    /* Dataframe */
    .stDataFrame {
        background-color: var(--secondary-background-color);
    }
    
    /* Expander */
    .st-expander {
        background-color: var(--secondary-background-color);
    }

</style>
""", unsafe_allow_html=True)


# =============================================================================
# --- BACKEND FUNCTIONS (DATA, MODEL, SMS) ---
# =============================================================================

@st.cache_data
def load_and_prepare_data():
    """Loads, merges, and prepares the INDOFLOODS dataset."""
    try:
        events_df = pd.read_csv('floodevents_indofloods.csv')
        precip_df = pd.read_csv('precipitation_variables_indofloods.csv')
        catchment_df = pd.read_csv('catchment_characteristics_indofloods.csv')
        # Clean column names
        for df in [events_df, precip_df, catchment_df]:
            df.columns = df.columns.str.strip()
        # Merge datasets
        df = pd.merge(events_df, precip_df, on='EventID', how='inner')
        df['GaugeID'] = df['EventID'].apply(lambda x: '-'.join(x.split('-')[:3]))
        df = pd.merge(df, catchment_df, on='GaugeID', how='left')
        return df
    except FileNotFoundError as e:
        st.error(f"Error: A required data file was not found ({e.filename}). Please ensure all CSV files are present in the same directory.")
        st.stop()

@st.cache_resource
def train_and_evaluate_models(_df):
    """Trains multiple models, returns the best one, metrics, and processing objects."""
    _df['Is_Severe_Flood'] = _df['Flood Type'].apply(lambda x: 1 if x == 'Severe Flood' else 0)
    _df['GaugeID_Num'] = _df['GaugeID'].str.extract(r'(\d+)').astype(int)
    features = [
        'Peak Flood Level (m)', 'Peak Discharge Q (cumec)', 'Event Duration (days)',
        'T1d', 'T7d', 'Drainage Area', 'Catchment Relief', 'GaugeID_Num'
    ]
    _df.dropna(subset=features, inplace=True)
    X = _df[features]
    y = _df['Is_Severe_Flood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }
    all_metrics = {}
    best_model_name = ""
    best_accuracy = 0
    best_model_obj = None

    progress_bar = st.progress(0, text="Training models...")
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = {
            "accuracy": accuracy,
            "precision": report.get("1", {}).get("precision", 0),
            "recall": report.get("1", {}).get("recall", 0)
        }
        all_metrics[name] = metrics
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_obj = model
        progress_bar.progress((i + 1) / len(models), text=f"Trained {name}...")
    progress_bar.empty()

    return scaler, features, all_metrics, best_model_name, best_model_obj, _df

@st.cache_data(ttl=600) # Cache weather data for 10 minutes
def fetch_weather_data(lat, lon):
    """Fetches live weather data from the Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m&hourly=temperature_2m,precipitation_probability,precipitation&forecast_days=2"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the weather API. Error: {e}")
        return None

def make_prediction(model, scaler, features, rainfall_forecast):
    """Makes a flood risk prediction based on input data."""
    # This is a simplified logic for demonstration.
    # A real-world scenario would use more sophisticated feature engineering.
    if rainfall_forecast < 20:
        input_data = {'Peak Flood Level (m)': 10, 'Peak Discharge Q (cumec)': 500, 'Event Duration (days)': 2, 'T1d': rainfall_forecast, 'T7d': rainfall_forecast * 1.5, 'Drainage Area': 5000, 'Catchment Relief': 200, 'GaugeID_Num': 1}
    elif rainfall_forecast < 60:
        input_data = {'Peak Flood Level (m)': 60, 'Peak Discharge Q (cumec)': 6000, 'Event Duration (days)': 5, 'T1d': rainfall_forecast, 'T7d': rainfall_forecast * 2, 'Drainage Area': 5000, 'Catchment Relief': 200, 'GaugeID_Num': 1}
    else:
        input_data = {'Peak Flood Level (m)': 150, 'Peak Discharge Q (cumec)': 15000, 'Event Duration (days)': 10, 'T1d': rainfall_forecast, 'T7d': rainfall_forecast * 2.5, 'Drainage Area': 5000, 'Catchment Relief': 200, 'GaugeID_Num': 1}
    
    df_live = pd.DataFrame([input_data], columns=features)
    df_live_scaled = scaler.transform(df_live)
    prediction = model.predict(df_live_scaled)[0]
    return "HIGH RISK" if prediction == 1 else "LOW RISK"

def truncate_location(location_name, max_length=40):
    """Truncates a long location string for display purposes."""
    if len(location_name) > max_length:
        return location_name[:max_length-3] + '...'
    return location_name

def send_sms_alert(phone_number, location_name, risk_level, rainfall_mm, language='English'):
    """Sends an SMS alert using Twilio."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, twilio_phone_number]):
        st.error("Twilio credentials are not configured. Cannot send SMS. Please check your environment file.")
        return False

    templates = {
        "English": "** FLOOD ALERT **\nLocation: {location}\nRisk: {risk}\n24h Rain: {rain:.2f} mm\nTake precautions.",
        "Hindi": "बाढ़ की चेतावनी\nस्थान: {location}\nजोखिम: {risk}\n24 घंटे की बारिश: {rain:.2f} मिमी\nसावधानी बरतें।",
        "Marathi": "पूर सूचना\nस्थान: {location}\nधोका: {risk}\n24 तासांचा पाऊस: {rain:.2f} मिमी\nकाळजी घ्या.",
        "Tamil": "வெள்ள எச்சரிக்கை\nஇடம்: {location}\nஆபத்து: {risk}\n24 மணி நேர மழை: {rain:.2f} மிமீ\nமுன்னெச்சரிக்கை நடவடிக்கைகளை மேற்கொள்ளவும்.",
        "Bengali": "বন্যা সতর্কতা\nঅবস্থান: {location}\nঝুঁকি: {risk}\n২৪ ঘন্টার বৃষ্টি: {rain:.2f} মিমি\nসতর্কতা অবলম্বন করুন।",
        "Telugu": "వరద హెచ్చరిక\nప్రదేశం: {location}\nప్రమాదం: {risk}\n24 గంటల వర్షపాతం: {rain:.2f} మిమీ\nజాగ్రత్తలు తీసుకోండి.",
        "Kannada": "ಪ್ರವಾಹದ ಎಚ್ಚರಿಕೆ\nಸ್ಥಳ: {location}\nಅಪಾಯ: {risk}\n೨೪ ಗಂಟೆಗಳ ಮಳೆ: {rain:.2f} ಮಿಮೀ\nಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ.",
        "Gujarati": "પૂરની ચેતવણી\nસ્થળ: {location}\nજોખમ: {risk}\n૨૪ કલાકનો વરસાદ: {rain:.2f} મીમી\nસાવચેતી રાખવી."
    }

    message_template = templates.get(language, templates['English'])
    short_location = truncate_location(location_name)
    message_body = message_template.format(location=short_location, risk=risk_level, rain=rainfall_mm)
    
    clean_phone_number = phone_number.strip()
    if not clean_phone_number.startswith('+'):
        clean_phone_number = '+' + clean_phone_number

    try:
        client = Client(account_sid, auth_token)
        client.messages.create(body=message_body, from_=twilio_phone_number, to=clean_phone_number)
        return True
    except Exception as e:
        st.error(f"SMS failed to send. Twilio Error: {e}")
        st.info("Please ensure the phone number is valid and includes the country code (e.g., +91).")
        return False

# =============================================================================
# --- FRONTEND PAGE FUNCTIONS (Using only Streamlit components) ---
# =============================================================================

def quick_weather_update(city):
    """Fetches and displays a quick weather update for a given city."""
    geolocator = Nominatim(user_agent="smart_flood_alert_app_intro")
    try:
        location_obj = geolocator.geocode(city)
        if location_obj:
            st.write(f"**Displaying current weather for: {location_obj.address}**")
            weather_data = fetch_weather_data(location_obj.latitude, location_obj.longitude)
            if weather_data and 'current' in weather_data:
                current_weather = weather_data['current']
                temp = current_weather.get('temperature_2m', 'N/A')
                humidity = current_weather.get('relative_humidity_2m', 'N/A')
                precipitation = current_weather.get('precipitation', 'N/A')
                wind_speed = current_weather.get('wind_speed_10m', 'N/A')

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Temperature", f"{temp}°C")
                col2.metric("Humidity", f"{humidity}%")
                col3.metric("Precipitation", f"{precipitation} mm")
                col4.metric("Wind Speed", f"{wind_speed} km/h")
            else:
                st.warning("Could not retrieve current weather data.")
        else:
            st.warning(f"Could not find location data for '{city}'. Please try a different name.")
    except Exception as e:
        st.error(f"An error occurred while fetching weather data: {e}")


def introduction_page():
    """Displays the introduction page of the application."""
    st.title("🌊 Smart Flood Alert System")
    st.subheader("Empowering Communities Against Floods with Data-Driven Insights")

    st.write(
        """
        Floods are among the most frequent and devastating natural disasters in India, affecting 
        millions of lives and livelihoods each year. This Smart Flood Alert System leverages 
        real-time weather data, historical flood records, and predictive machine learning models 
        to provide timely flood risk analysis and multilingual SMS alerts to vulnerable communities.
        """
    )
    
    st.divider()

    st.header("How It Works")
    st.markdown(
        """
        - **Collects Live Data:** Gathers up-to-the-minute weather and rainfall data from reliable APIs.
        - **Predicts Flood Risk:** Utilizes a trained machine learning model to predict the likelihood and severity of a flood.
        - **Alerts Communities:** Sends instant SMS alerts to registered users in their preferred local language.
        """
    )

    st.divider()

    # --- Interactive Weather Update Section ---
    st.header("Quick Weather Update")
    
    with st.form(key="quick_update_form"):
        location_input = st.text_input(
            "Enter Location",
            "Mumbai",
            help="Enter a city name and click the button to get the latest weather."
        )
        submitted = st.form_submit_button("Quick Update")

    if submitted and location_input:
        quick_weather_update(city=location_input)

    st.divider()
    st.info("Use the sidebar navigation on the left to explore the model's performance or to run a live risk analysis.", icon="🧭")
                

def model_performance_page():
    """Displays the model training and performance evaluation page."""
    st.title("📊 Model Performance & Insights")
    st.write(
        "Here you can train several machine learning models on the historical flood dataset "
        "and compare their performance based on key metrics like accuracy, precision, and recall."
    )

    if st.button("🚀 Train & Evaluate Models", use_container_width=True, type="primary"):
        with st.spinner("Loading data and training models... This may take a moment."):
            master_df = load_and_prepare_data()
            scaler, features, all_metrics, best_model_name, best_model_obj, processed_df = train_and_evaluate_models(master_df)
            
            # Store results in session state
            st.session_state.models_trained = True
            st.session_state.scaler = scaler
            st.session_state.features = features
            st.session_state.all_metrics = all_metrics
            st.session_state.best_model_name = best_model_name
            st.session_state.best_model_obj = best_model_obj
            st.session_state.processed_df = processed_df
            
            st.success(f"Models trained successfully! The best performing model is **{best_model_name}**.")

    if 'models_trained' in st.session_state and st.session_state.models_trained:
        st.divider()
        st.subheader("Model Comparison")
        
        metrics_df = pd.DataFrame(st.session_state.all_metrics).T
        metrics_df = metrics_df[['accuracy', 'precision', 'recall']]
        metrics_df.index.name = "Model"
        
        st.dataframe(metrics_df.style.format("{:.2%}").highlight_max(axis=0, color="#1c83e1"))
        
        st.divider()
        st.subheader("Metric Definitions")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Accuracy", value="")
            st.write("Overall, how often is the model correct in its predictions? (Higher is better)")
        with col2:
            st.metric(label="Precision", value="")
            st.write("When the model predicts a severe flood, how often is it right? (Focuses on avoiding false alarms)")
        with col3:
            st.metric(label="Recall", value="")
            st.write("Of all the actual severe floods, how many did the model correctly identify? (Focuses on not missing real events)")


        best_model_obj = st.session_state.best_model_obj
        # Check for feature importance (available in tree-based models)
        if hasattr(best_model_obj, 'feature_importances_'):
            st.divider()
            st.subheader(f"Feature Importance ({st.session_state.best_model_name})")
            feature_importance_df = pd.DataFrame({
                'Feature': st.session_state.features,
                'Importance': best_model_obj.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', 
                         title="Driver Features of Flood Risk")
            fig.update_traces(marker_color='#1c83e1')
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("Processed Dataset Sample")
        st.write("A quick look at the data used for training the models.")
        st.dataframe(st.session_state.processed_df.head())

def live_risk_analysis_page():
    """Displays the live risk analysis page."""
    st.title("🛰️ Live Flood Risk Analysis")
    st.write("Enter a location to get a real-time flood risk assessment based on the latest weather forecasts and our predictive model.")
    
    # Ensure models are trained before allowing analysis
    if 'models_trained' not in st.session_state:
        st.warning("The prediction models have not been trained yet. Please go to the 'Model Performance' page and train the models first.")
        if st.button("Go to Model Training"):
            st.session_state.page = "Model Performance"
            st.rerun()
        st.stop()
    
    with st.form("risk_analysis_form"):
        st.subheader("Analysis Parameters")
        col1, col2 = st.columns([3, 2])
        with col1:
            location = st.text_input("📍 Enter City or Location", "Mumbai, India", help="Enter a city, town, or specific address.")
        with col2:
            language = st.selectbox(
                "🌐 Select Alert Language",
                ["English", "Hindi", "Marathi", "Tamil", "Bengali", "Telugu", "Kannada", "Gujarati"]
            )
        
        phone = st.text_input("📱 Phone Number for SMS Alert (Optional)", placeholder="+919876543210", help="Include country code, e.g., +91")
        hide_number = st.checkbox("Hide phone number in confirmation message", value=True)
        
        submitted = st.form_submit_button("Analyze Flood Risk", use_container_width=True, type="primary")

    if submitted:
        if not location:
            st.error("Please enter a location to analyze.")
            st.stop()

        with st.spinner(f"Analyzing flood risk for {location}..."):
            geolocator = Nominatim(user_agent="smart_flood_alert_app")
            try:
                location_obj = geolocator.geocode(location)
                if not location_obj:
                    st.error(f"Could not find coordinates for '{location}'. Please try a different or more specific location.")
                    st.stop()
            except Exception as e:
                st.error(f"Geocoding failed: {e}")
                st.stop()

            weather_data = fetch_weather_data(location_obj.latitude, location_obj.longitude)
            if not weather_data or 'hourly' not in weather_data or 'current' not in weather_data:
                st.error("Could not fetch valid weather data for the specified location.")
                st.stop()
            
            # Extract weather info
            current_weather = weather_data['current']
            temp = current_weather.get('temperature_2m', 'N/A')
            humidity = current_weather.get('relative_humidity_2m', 'N/A')
            wind_speed = current_weather.get('wind_speed_10m', 'N/A')

            # Calculate total rainfall and make prediction
            rainfall_forecast = weather_data['hourly']['precipitation'][:24]
            total_rainfall = sum(rainfall_forecast)
            risk_level = make_prediction(
                st.session_state.best_model_obj,
                st.session_state.scaler,
                st.session_state.features,
                total_rainfall
            )
            
            # --- Display Results ---
            st.divider()
            st.header(f"Analysis for: {location_obj.address}")
            
            if risk_level == "HIGH RISK":
                st.error(f"### 🚨 Status: {risk_level}", icon="🚨")
            else:
                st.success(f"### ✅ Status: {risk_level}", icon="✅")

            st.subheader("Current Weather Conditions")
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{temp}°C")
            c2.metric("Humidity", f"{humidity}%")
            c3.metric("Wind Speed", f"{wind_speed} km/h")
        
            st.subheader("24-Hour Rainfall Forecast")
            st.bar_chart(pd.DataFrame({
                "Hour": range(1, 25),
                "Precipitation (mm)": rainfall_forecast
            }).set_index("Hour"), color="#1c83e1")

            # --- Send SMS Alert if phone number is provided ---
            if phone:
                with st.spinner("Sending SMS alert..."):
                    sms_sent = send_sms_alert(phone, location_obj.address, risk_level, total_rainfall, language)
                    if sms_sent:
                        display_phone = "a hidden number" if hide_number else phone
                        st.success(f"SMS alert successfully sent to {display_phone}!")

# =============================================================================
# --- SIDEBAR NAVIGATION & MAIN ROUTING ---
# =============================================================================

if "page" not in st.session_state:
    st.session_state.page = "Introduction"

with st.sidebar:
    st.title("🧭 Smart Flood Alert")
    
    selected_page = st.radio(
        "Navigation",
        ["Introduction", "Model Performance", "Live Risk Analysis"],
        captions=["Home", "Train & Evaluate", "Real-Time Analysis"],
        key="navigation_radio"
    )
    
    st.session_state.page = selected_page

    st.divider()
    st.info(
        "This application is a proof-of-concept for a data-driven flood warning system."
    )

# --- MAIN PAGE ROUTING ---
if st.session_state.page == "Introduction":
    introduction_page()
elif st.session_state.page == "Model Performance":
    model_performance_page()
elif st.session_state.page == "Live Risk Analysis":
    live_risk_analysis_page()