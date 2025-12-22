import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide"
)

# ------------------ Load Model ------------------
model = joblib.load("covid_rf_model.pkl")

# ------------------ Load Dataset ------------------
df = pd.read_csv("owid-covid-data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.fillna(0)

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Controls")

country = st.sidebar.selectbox(
    "Select Country",
    sorted(df["location"].unique()),
    index=sorted(df["location"].unique()).index("India")
)

filtered_df = df[df["location"] == country]

# ------------------ Title ------------------
st.title("🦠 COVID-19 Case Prediction Dashboard")


# ------------------ Metrics ------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Cases",
    f"{int(filtered_df['total_cases'].max()):,}"
)

col2.metric(
    "Total Deaths",
    f"{int(filtered_df['total_deaths'].max()):,}"
)

col3.metric(
    "Total Vaccinations",
    f"{int(filtered_df['total_vaccinations'].max()):,}"
)

# ------------------ Visualizations ------------------
st.subheader("📈 Daily New COVID-19 Cases")

plt.figure()
plt.plot(filtered_df["date"], filtered_df["new_cases"])
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.xticks(rotation=45)
st.pyplot(plt)

st.subheader("📊 Total Cases Over Time")
plt.figure()
plt.plot(filtered_df["date"], filtered_df["total_cases"])
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=45)
st.pyplot(plt)

st.subheader("💉 Vaccination Progress")
plt.figure()
plt.plot(filtered_df["date"], filtered_df["total_vaccinations"])
plt.xlabel("Date")
plt.ylabel("Total Vaccinations")
plt.xticks(rotation=45)
st.pyplot(plt)

# ------------------ Prediction Section ------------------
st.subheader("🔮 Predict New COVID-19 Cases")

col1, col2 = st.columns(2)

with col1:
    total_cases = st.number_input(
        "Total Cases",
        min_value=0,
        value=int(filtered_df["total_cases"].max()),
        help="Cumulative confirmed cases"
    )
    total_deaths = st.number_input(
        "Total Deaths",
        min_value=0,
        value=int(filtered_df["total_deaths"].max())
    )

with col2:
    total_vaccinations = st.number_input(
        "Total Vaccinations",
        min_value=0,
        value=int(filtered_df["total_vaccinations"].max())
    )
    population = st.number_input(
        "Population",
        min_value=0,
        value=int(filtered_df["population"].iloc[0])
    )

if st.button("🚀 Predict"):
    input_data = [[
        total_cases,
        total_deaths,
        total_vaccinations,
        population
    ]]

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted New COVID-19 Cases: **{int(prediction):,}**"
    )

    st.info(
        "⚠️ Prediction is based on historical data and intended for educational purposes only."
    )
