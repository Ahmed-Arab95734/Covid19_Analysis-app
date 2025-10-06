import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Set wide layout
st.set_page_config(page_title="COVID-19 Data Analysis", layout="wide")

# Title
st.title("🦠 COVID-19 Data Analysis Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Data Overview", "🧹 Data Cleaning", "📈 EDA & Charts", "❓ Business Insights", "✅ Recommendations"])

# Load data
@st.cache_data
def load_data():
    full_grouped = pd.read_csv("full_grouped.csv")
    worldometer_data = pd.read_csv("worldometer_data.csv")
    return full_grouped, worldometer_data

full_grouped, worldometer_data = load_data()

# ---------------------------
# PAGE 1 - DATA OVERVIEW
# ---------------------------
if page == "📊 Data Overview":
    st.subheader("Dataset Sources")
    st.markdown("""
    - `full_grouped.csv` → Time series data by country  
    - `worldometer_data.csv` → Country-level stats snapshot  
    """)

    st.write("### Preview of Datasets")
    tab1, tab2 = st.tabs(["🌍 full_grouped", "📌 worldometer_data"])
    with tab1:
        st.dataframe(full_grouped.head())
    with tab2:
        st.dataframe(worldometer_data.head())

# ---------------------------
# PAGE 2 - DATA CLEANING
# ---------------------------
elif page == "🧹 Data Cleaning":
    st.subheader("Data Quality Checks & Cleaning")
    st.write("### Duplicates & Null Values Summary")

    col1, col2 = st.columns(2)
    col1.metric("full_grouped duplicates", full_grouped.duplicated().sum())
    col2.metric("worldometer_data duplicates", worldometer_data.duplicated().sum())

    st.write("### Handling Nulls & Negatives")
    st.markdown("""
    - Removed negative values (<0.3% of rows).  
    - Dropped non-country records (e.g. ships).  
    - Removed rows missing critical fields (deaths/tests/recovered).  
    - Replaced missing `NewCases/NewDeaths/NewRecovered` with 0.  
    - Dropped redundant column `WHO Region`.  
    """)

# ---------------------------
# PAGE 3 - EDA & CHARTS
# ---------------------------
elif page == "📈 EDA & Charts":
    st.subheader("Exploratory Data Analysis")

    # Convert date column
    full_grouped["Date"] = pd.to_datetime(full_grouped["Date"])

    # Monthly Aggregation
    monthly = full_grouped.groupby(full_grouped["Date"].dt.to_period("M"))[["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    # ---- Global Trends ----
    st.write("### Global Trends in 2020")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.lineplot(data=monthly, x="Date", y="Confirmed", marker="o", ax=ax[0], color="blue")
    ax[0].set_title("Confirmed Cases")
    sns.lineplot(data=monthly, x="Date", y="Deaths", marker="o", ax=ax[1], color="red")
    ax[1].set_title("Deaths")
    sns.lineplot(data=monthly, x="Date", y="Recovered", marker="o", ax=ax[2], color="green")
    ax[2].set_title("Recovered")
    st.pyplot(fig)

    st.info("**Insight:** COVID-19 spread in waves during 2020, with confirmed cases peaking earlier than deaths. Recovery followed with a delay, reflecting treatment and reporting lags.")

    # ---- Top 10 Countries ----
    st.write("### Top 10 Countries by Confirmed, Deaths, and Recovered")
    top_confirmed = worldometer_data.nlargest(10, "TotalCases")[["Country/Region", "TotalCases"]]
    top_deaths = worldometer_data.nlargest(10, "TotalDeaths")[["Country/Region", "TotalDeaths"]]
    top_recovered = worldometer_data.nlargest(10, "TotalRecovered")[["Country/Region", "TotalRecovered"]]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    sns.barplot(x="TotalCases", y="Country/Region", data=top_confirmed, ax=ax[0], palette="Blues_r")
    ax[0].set_title("Top 10 Countries - Confirmed Cases")
    sns.barplot(x="TotalDeaths", y="Country/Region", data=top_deaths, ax=ax[1], palette="Reds_r")
    ax[1].set_title("Top 10 Countries - Deaths")
    sns.barplot(x="TotalRecovered", y="Country/Region", data=top_recovered, ax=ax[2], palette="Greens_r")
    ax[2].set_title("Top 10 Countries - Recovered")
    st.pyplot(fig)

    st.info("**Insight:** The US, India, and Brazil dominate confirmed cases and deaths, highlighting regions under heavy strain. High recovery counts in India show progress in treatment and resilience.")

    # ---- CFR by Continent ----
    worldometer_data["CFR(%)"] = (worldometer_data["TotalDeaths"] / worldometer_data["TotalCases"]) * 100
    continent_cfr = worldometer_data.groupby("Continent")["CFR(%)"].mean().sort_values(ascending=False)

    st.write("### Average CFR by Continent")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=continent_cfr.index, y=continent_cfr.values, ax=ax, palette="magma")
    ax.set_title("Case Fatality Rate (CFR %)")
    st.pyplot(fig)

    st.info("**Insight:** Europe shows the highest average fatality rates, suggesting older demographics and strained health systems. Africa and Asia show lower CFRs, possibly due to younger populations or underreporting.")

    # ---- CFR Choropleth ----
    st.write("### Global Case Fatality Rate Map")
    fig = px.choropleth(worldometer_data,
                        locations="Country/Region",
                        locationmode="country names",
                        color="CFR(%)",
                        hover_name="Country/Region",
                        color_continuous_scale="Reds",
                        title="Global Case Fatality Rate (CFR %)")
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Insight:** Countries like Italy, Mexico, and the UK display elevated CFRs, highlighting higher vulnerability. In contrast, nations with strong testing and hospital capacity show lower CFRs.")

    # =========================
    # ✅ NEW CHARTS
    # =========================
    st.write("### Total Tests vs. Total Cases (Log-Scaled with Regression Line)")

    # Ensure numeric values
    worldometer_data["TotalTests"] = pd.to_numeric(worldometer_data["TotalTests"], errors="coerce")
    worldometer_data["TotalCases"] = pd.to_numeric(worldometer_data["TotalCases"], errors="coerce")

    # Create log-transformed columns
    worldometer_data["log_TotalTests"] = np.log10(worldometer_data["TotalTests"] + 1)
    worldometer_data["log_TotalCases"] = np.log10(worldometer_data["TotalCases"] + 1)

    # Plot with log-transformed values
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=worldometer_data, 
        x="log_TotalTests", 
        y="log_TotalCases", 
        hue="Continent", 
        alpha=0.7, 
        ax=ax
    )
    sns.regplot(
        data=worldometer_data, 
        x="log_TotalTests", 
        y="log_TotalCases", 
        scatter=False, 
        color="black", 
        ax=ax
    )
    ax.set_title("Total Tests vs. Total Cases (Log-Scaled Regression)")
    ax.set_xlabel("Log10 Total Tests")
    ax.set_ylabel("Log10 Total Cases")
    ax.grid(True, which="both", linestyle="--")

    st.pyplot(fig)

    st.info("**Insight:** More testing generally leads to more detected cases. However, Africa shows fewer cases despite limited testing, suggesting under-detection rather than lower spread.")

    # ---- Global New Cases & Deaths ----
    st.write("### Global Trends of New Cases & New Deaths Over Time")
    full_grouped["Date"] = pd.to_datetime(full_grouped["Date"])
    global_time = full_grouped.groupby("Date")[["New cases", "New deaths"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=global_time, x="Date", y="New cases", label="New Cases", ax=ax)
    sns.lineplot(data=global_time, x="Date", y="New deaths", label="New Deaths", ax=ax)
    ax.set_title("Global Trends of New Cases & Deaths Over Time")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.info("**Insight:** Peaks of new cases often precede peaks in deaths, showing the lag between infection and fatality outcomes. Clear pandemic 'waves' are visible globally.")

    # ---- Recovery Rate ----
    st.write("### Average Recovery Rate (%) by Continent")
    worldometer_data["RecoveryRate(%)"] = (worldometer_data["TotalRecovered"] / worldometer_data["TotalCases"]) * 100
    continent_recovery = worldometer_data.groupby("Continent")["RecoveryRate(%)"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=continent_recovery.index, y=continent_recovery.values, palette="Greens", ax=ax)
    ax.set_title("Average Recovery Rate (%) by Continent")
    ax.set_ylabel("Recovery Rate (%)")
    st.pyplot(fig)

    st.info("**Insight:** Asia leads recovery rates (>75%), suggesting better treatment outcomes and rapid responses. The Americas lag behind, reflecting healthcare capacity challenges.")


# ---------------------------
# PAGE 4 - BUSINESS INSIGHTS
# ---------------------------
elif page == "❓ Business Insights":
    st.subheader("Business Questions & Insights")

    st.markdown("""
    **Q1:** Where should limited global health resources focus?  
    - **A:** The Americas and Europe, as they have the largest case burden.  

    **Q2:** Which continent has the highest CFR and what does it mean?  
    - **A:** Europe, suggesting strain on healthcare systems and vulnerable populations.  

    **Q3:** How effective was testing in controlling spread?  
    - **A:** Stronger testing in Europe/North America led to more case detection; low testing in Africa caused under-detection.  

    **Q4:** What trends are seen in new cases & deaths?  
    - **A:** Clear waves of peaks/declines linked to variants and policy changes.  

    **Q5:** Which continent recovered fastest?  
    - **A:** Asia (>75% recovery rate).  
    """)

# ---------------------------
# PAGE 5 - RECOMMENDATIONS
# ---------------------------
elif page == "✅ Recommendations":
    st.subheader("Actionable Recommendations")
    st.markdown("""
    - **Immediate Aid (Hotspots):** Prioritize Americas & Europe → PPE, ventilators, financial support.  
    - **Testing:** Expand in Africa & under-tested regions, link with tracing.  
    - **Vaccination:** Time campaigns before predicted waves, target boosters.  
    - **Knowledge Sharing:** Spread practices from Asia (high recovery) globally.  
    - **Long-term:** Build resilient healthcare systems & regional hubs for future pandemics.  
    """)


# =========================
# Background Image with Light Text and Dark Sidebar Widgets
# =========================
page_bg_img = """
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://wholesale.banking.societegenerale.com/typo3temp/assets/_processed_/3/2/csm_GettyImages-1420054557-350-100_Header_web_1742917813_cd11b67f55.webp");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #f1f1f1;  /* light text */
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)




