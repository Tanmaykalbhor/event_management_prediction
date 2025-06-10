import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="🎉 Event Price Predictor", page_icon="🎈", layout="wide")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Data Insights", "🧮 Price Prediction"])

# Load dataset
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv('C:\\Users\\udayg\\Pictures\\for driver\\OneDrive\\Desktop\\election\\event_management_price_prediction.csv')
    return df

# Train model and prepare data
@st.cache_resource(show_spinner=True)
def train_model(df):
    X = df.drop(columns=["Event_ID", "Total_Price"])
    y = df["Total_Price"]

    categorical_cols = ["Event_Type", "Location", "Catering_Type", "Entertainment", "Season"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, mae, rmse, r2

# Load data and train model
df = load_data()
with st.spinner("Training the model..."):
    model, mae, rmse, r2 = train_model(df)

# --- Page: Home ---
if page == "🏠 Home":
    st.markdown("""
        <div style='background-color: #FFDEE9; padding: 20px; border-radius: 12px;'>
            <h1 style='color: #6A0572;'>🎉 Event Management Price Predictor</h1>
            <p style='color: #333333; font-size: 18px;'>Welcome to the Event Management Price Predictor app! 🏆<br><br>
            This app helps you predict the estimated cost of organizing various types of events such as weddings, birthdays, corporate conferences, and concerts based on your event details. 🎪🎤🎂<br><br>
            Event management involves careful planning and execution of an event with attention to details such as venue, catering, entertainment, and guest experience. 🎈🍽️🎶<br><br>
            Use the "Data Insights" page to explore the dataset 📊 and the "Price Prediction" page to predict your event cost! 💰<br><br>
            Enjoy planning your perfect event! 🚀✨
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- Page: Data Insights ---
elif page == "📈 Data Insights":
    st.title("📊 Event Data Insights")

    st.subheader("⚙️ Dataset Overview")
    if st.checkbox("Show raw data"):
        st.dataframe(df.head(20))

    st.subheader("📌 Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Total_Price"], bins=30, kde=True, color="#F76C6C", ax=ax1)
    ax1.set_title("Total Price Distribution", fontsize=16)
    st.pyplot(fig1)

    st.subheader("📌 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="YlGnBu", ax=ax2)
    st.pyplot(fig2)

    st.subheader(f"📈 Model Metrics")
    st.markdown(f"- 🎯 **MAE**: `{mae:,.2f}`")
    st.markdown(f"- 🎯 **RMSE**: `{rmse:,.2f}`")
    st.markdown(f"- 🎯 **R² Score**: `{r2:.4f}`")

# --- Page: Price Prediction ---
elif page == "🧮 Price Prediction":
    st.title("🧮 Predict Event Price")

    st.subheader("🎈 Enter Event Details")

    event_type = st.selectbox("🎭 Event Type", df["Event_Type"].unique())
    location = st.selectbox("📍 Location", df["Location"].unique())
    guests_count = st.slider("👥 Number of Guests", min_value=30, max_value=600, step=10)
    duration_hours = st.slider("⏳ Duration (Hours)", min_value=2, max_value=12, step=1)
    catering_type = st.selectbox("🍽️ Catering Type", df["Catering_Type"].unique())
    entertainment = st.selectbox("🎵 Entertainment", df["Entertainment"].unique())
    season = st.selectbox("🍁 Season", df["Season"].unique())
    venue_rating = st.slider("⭐ Venue Rating", min_value=3.5, max_value=5.0, step=0.1)

    if st.button("🔍 Predict Price"):
        input_df = pd.DataFrame({
            "Event_Type": [event_type],
            "Location": [location],
            "Guests_Count": [guests_count],
            "Duration_Hours": [duration_hours],
            "Catering_Type": [catering_type],
            "Entertainment": [entertainment],
            "Season": [season],
            "Venue_Rating": [venue_rating]
        })

        price_prediction = model.predict(input_df)[0]

        st.subheader("💰 Predicted Total Price")
        st.success(f"Estimated Price: ₹ {price_prediction:,.2f}")
