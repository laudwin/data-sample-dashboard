import streamlit as st
import pandas as pd 
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import streamlit.components.v1 as stc
import time

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

from UI import *
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import streamlit_javascript as stj


# Page config
st.set_page_config(page_title="Facebook Data Analysis", page_icon="🌎", layout="wide")
#heading()





# Load CSS
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load data
df = pd.read_csv('data.csv')



# --- Filter: Keep only customer complaints ---
def is_customer_complaint(text):
    text = str(text).lower()
    promo_keywords = [
        "congratulations", "order online", "save", "special offer", "bundle",
        "get more monate", "purchase dial", "shop now", "winner", "airtime", 
        "available now", "just r", "deal", "order here", "buy now", "subscribe",
        "dm me", "get r", "telkom weekend surfer", "to purchase", "competition",
        "visit our website", "apply now", "promotion"
    ]
    if text.startswith("rt @telkom") or text.startswith("rt @telkomm"):
        return False
    if len(text.strip()) < 20:
        return False
    if any(kw in text for kw in promo_keywords):
        return False
    return True

df["is_customer_complaint"] = df["extract"].apply(is_customer_complaint)

df = df[df["is_customer_complaint"]]
#--------------------------------



# Sidebar filters
st.sidebar.header("Please Filter Here:")

region = st.sidebar.selectbox(
    "Select Region:",
    options=sorted(df["region.name"].dropna().unique())
)

# Filter cities based on region
filtered_cities = df[df["region.name"] == region]["city.name"].dropna().unique()
city = st.sidebar.selectbox(
    "Select City:",
    options=sorted(filtered_cities)
)

selected_category = st.sidebar.selectbox(
    "Select Category:",
    options=sorted(df["category.label"].dropna().unique())
)

selected_gender = st.sidebar.selectbox(
    "Select Gender:",
    options=sorted(df["gender.label"].dropna().unique())
)

# Query filtered dataset
df_selection = df.query(
    "`region.name` == @region and `city.name` == @city and `category.label` == @selected_category and `gender.label` == @selected_gender"
)


def classify_issue(text):
    text = str(text).lower()

    if re.search(r"invoice|debit|charge|billing|cancel|refund|penalt(y|ies)|contract|account|price", text):
        return "Billing"
    if re.search(r"network|signal|coverage|slow|speed|disconnect|outage|data", text):
        return "Network"
    if re.search(r"insurance|claim|submit|decline|damage|theft|cancellation", text):
        return "Insurance"
    if re.search(r"support|help|service|response|resolve|rude|ignored|agent", text):
        return "Support"
    if re.search(r"purchase|verify|verification|sms|order|delivery", text):
        return "Purchase"

    return "Other"

df["customer_issue"] = df["extract"].apply(classify_issue)

df_selection["customer_issue"] = df_selection["extract"].apply(classify_issue)

# --- Classification: Customer Issue Types ---


# HomePage Function
def HomePage():
    # Display table
    with st.expander("🧭 My database"):
        shwdata = st.multiselect('Filter columns:', df_selection.columns, default=[])
        st.dataframe(df_selection[shwdata], use_container_width=True)

    # Handle empty data gracefully
    def safe_stat(value, fallback="0"):
        return value if pd.notnull(value) else fallback

    total_engagement = safe_stat(df_selection["engagement"].sum())
    engagement_mean = safe_stat(df_selection["engagement"].mean())
    engagement_median = safe_stat(df_selection["engagement"].median())
    engagement_mode = df_selection["engagement"].mode()
    engagement_mode = engagement_mode[0] if not engagement_mode.empty else 0

    total_OTS = safe_stat(df_selection["OTS"].sum())
    sentiment_score = safe_stat(df_selection["sentiment"].sum())

    # Metrics layout
    total1, total2, total3, total4, total5 = st.columns(5, gap='large')

    with total1:
        st.info('Total Engagement', icon="📊")
        st.metric(label='Sum', value=f"{total_engagement:,.0f}")

    with total2:
        st.info('Most Common Engagement', icon="📊")
        st.metric(label='Mode', value=f"{engagement_mode:,.0f}")

    with total3:
        st.info('Average Engagement', icon="📊")
        st.metric(label='Mean', value=f"{engagement_mean if pd.notnull(engagement_mean) else 'N/A'}")

    with total4:
        st.info('Median Engagement', icon="📊")
        st.metric(label='Median', value=f"{engagement_median if pd.notnull(engagement_median) else 'N/A'}")

    with total5:
        st.info('Total OTS + Sentiment', icon="📊")
        st.metric(
            label='OTS & Sentiment',
            value=f"{numerize(float(total_OTS))} / {numerize(float(sentiment_score))}",
            help=f"Total OTS: {total_OTS}, Sentiment Sum: {sentiment_score}"
        )

    st.markdown("---")

# Run homepage
HomePage()


# --- Graphs Function ---
def Graphs():
    st.markdown("### 📊 Visual Insights")

    # ---- BAR CHART: Top 10 Cities by Engagement ----
    top_cities = df_selection.groupby("city.name")["engagement"].sum().sort_values(ascending=False).head(10)
    top_cities_df = top_cities.reset_index()
    top_cities_df.columns = ["city", "engagement"]

    fig_city_engagement = px.bar(
        top_cities_df,
        x="city",
        y="engagement",
        title="Top 10 Cities by Engagement",
        color="engagement",
        color_continuous_scale="Blues",
        template="plotly_white"
    )
    fig_city_engagement.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45
    )

    # ---- LINE CHART: Engagement Over Time ----
    df_selection["published"] = pd.to_datetime(df_selection["published"], errors='coerce')
    engagement_by_day = df_selection.groupby(df_selection["published"].dt.date)["engagement"].sum().sort_index()
    engagement_by_day_df = engagement_by_day.reset_index()
    engagement_by_day_df.columns = ["date", "engagement"]

    fig_engagement_trend = px.line(
        engagement_by_day_df,
        x="date",
        y="engagement",
        title="Engagement Over Time",
        template="plotly_white",
        markers=True
    )
    fig_engagement_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Engagement",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # ---- BAR CHART: Top Categories by Engagement ----
    top_categories = df_selection.groupby("category.label")["engagement"].sum().sort_values(ascending=False).head(10)
    top_categories_df = top_categories.reset_index()
    top_categories_df.columns = ["category", "engagement"]

    fig_category_engagement = px.bar(
        top_categories_df,
        x="category",
        y="engagement",
        title="Top Categories by Engagement",
        color="engagement",
        color_continuous_scale="Purples",
        template="plotly_white"
    )
    fig_category_engagement.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45
    )

    # ---- Display in columns ----
    left_column, right_column, center = st.columns(3)
    left_column.plotly_chart(fig_engagement_trend, use_container_width=True)
    right_column.plotly_chart(fig_city_engagement, use_container_width=True)
    with center:
        st.plotly_chart(fig_category_engagement, use_container_width=True)

# Run Graphs        
Graphs()

def CustomerIssuesAnalysis():
    st.markdown("## 📋 Customer Issues Analysis")

    # Dropdown to filter issue types
    issue_filter = st.selectbox("🧭 Filter by Issue Category:", options=["All"] + sorted(df["customer_issue"].unique()))

    df_issues = df.copy()
    if issue_filter != "All":
        df_issues = df_issues[df_issues["customer_issue"] == issue_filter]

    # Count frequency of each issue
    issue_counts = df_issues["customer_issue"].value_counts().reset_index()
    issue_counts.columns = ["Issue Type", "Count"]

    fig_issues = px.bar(
        issue_counts,
        x="Issue Type",
        y="Count",
        title="Frequency of Customer Issues",
        color="Count",
        template="plotly_white"
    )
    fig_issues.update_layout(plot_bgcolor="rgba(0,0,0,0)")

    st.plotly_chart(fig_issues, use_container_width=True)

    # Show full data if needed
    with st.expander("🔍 View Complaints with Issue Classification"):
        st.dataframe(df_issues[["extract", "customer_issue"]], use_container_width=True)

CustomerIssuesAnalysis()

def KeywordSearch():
    st.markdown("## 🔍 Keyword Search in Customer Complaints")

    # Input search term(s)
    search_term = st.text_input("Enter keyword or phrase to search for:", "")

    # Filter comments based on keyword
    if search_term:
        filtered = df_selection[df_selection["extract"].str.contains(search_term, case=False, na=False)]
        
        st.markdown(f"### Results for: _{search_term}_ ({len(filtered)} found)")
        st.dataframe(filtered[["published", "extract", "customer_issue", "sentiment"]], use_container_width=True)

        # Optional: Show simple chart of issue categories within results
        if not filtered.empty:
            issue_counts = filtered["customer_issue"].value_counts().reset_index()
            issue_counts.columns = ["Issue Type", "Count"]
            fig = px.bar(issue_counts, x="Issue Type", y="Count", title="Issue Types in Search Results", color="Count", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Type a keyword above to search within filtered complaints.")

#-- Run KeySearch

KeywordSearch()

# -- Dashboard Function

def Dashboard():
    def safe_stat(val, fallback=0):
        return val if pd.notnull(val) else fallback

    # Ensure date is parsed
    df_selection["published"] = pd.to_datetime(df_selection["published"], errors='coerce')

    # ======================
    # 🧠 Basic Checks
    # ======================
    if df_selection.empty:
        st.warning("No data found for your selection.")
        return

    region_selected = df_selection["region.name"].iloc[0] if "region.name" in df_selection.columns else "Unknown Region"
    city_selected = df_selection["city.name"].iloc[0] if "city.name" in df_selection.columns else "Unknown City"
    label_scope = f"{city_selected}, {region_selected}"

    # ======================
    # 💡 CX Index (Current Filter Only)
    # ======================
    sentiment_score = safe_stat(df_selection["sentiment"].mean())
    engagement_score = safe_stat(df_selection["engagement"].mean())
    OTS_score = safe_stat(df_selection["OTS"].mean())
    experience_index = round((sentiment_score * 0.4 + engagement_score * 0.4 + OTS_score * 0.2), 2)

    # ======================
    # 🎯 Gauge for CX Index (Only for current filtered area)
    # ======================
    if experience_index < 1000:
        cx_label = "🔴 🤣 Poor"
    elif experience_index < 2000:
        cx_label = "🟠 Average"
    elif experience_index < 3000:
        cx_label = "🟡 Good"
    else:
        cx_label = "🟢 Excellent"

    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=experience_index,
        title={
            'text': f"CX Index for {label_scope}<br><span style='font-size:0.8em;color:gray'>{cx_label}</span>"
        },
        gauge={
            'axis': {'range': [0, 4000]},
            'bar': {'color': "royalblue"},
            'steps': [
                {'range': [0, 1000], 'color': "tomato"},
                {'range': [1000, 2000], 'color': "orange"},
                {'range': [2000, 3000], 'color': "lightgreen"},
                {'range': [3000, 4000], 'color': "seagreen"},
            ],
        }
    ))

    # ======================
    # 🏷️ Top Brands by Engagement
    # ======================
    top_brands = (
        df_selection.groupby("brand.fullName")["engagement"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"brand.fullName": "Brand", "engagement": "Engagement"})
    )

    fig_brands = go.Figure()
    if not top_brands.empty:
        fig_brands = px.bar(
            top_brands,
            x="Brand",
            y="Engagement",
            title="🏷️ Top Brands by Engagement",
            color="Engagement",
            color_continuous_scale="Plasma",
            template="plotly_white"
        )
        fig_brands.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)

    # ======================
    # 📊 Sentiment Breakdown
    # ======================
    sentiment_counts = df_selection["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig_sentiment = go.Figure()
    if not sentiment_counts.empty:
        fig_sentiment = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title="🧠 Sentiment Breakdown",
            color="Sentiment",
            template="plotly_white"
        )
        fig_sentiment.update_layout(plot_bgcolor="rgba(0,0,0,0)")

    # ======================
    # 📍 OTS by City in Selected Region
    # ======================
    fig_ots_city = go.Figure()
    fig_ots_pie = go.Figure()
    if "region.name" in df_selection.columns:
        df_region = df_selection[df_selection["region.name"] == region_selected]
        if not df_region.empty:
            ots_by_city = (
                df_region.groupby("city.name")["OTS"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={"city.name": "City", "OTS": "OTS"})
            )

            if not ots_by_city.empty:
                # Bar
                fig_ots_city = px.bar(
                    ots_by_city,
                    x="City",
                    y="OTS",
                    title=f"📍 OTS by City in {region_selected}",
                    color="OTS",
                    template="plotly_white"
                )
                fig_ots_city.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)

              
    # ======================
    # 📊 Layout (3 charts + Gauge)
    # ======================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Customer Experience Index",
            value=experience_index,
            help=f"Filtered view: {label_scope}"
        )
        st.plotly_chart(fig_brands, use_container_width=True)

    with col2:
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col3:
        st.plotly_chart(fig_ots_city, use_container_width=True)

    st.markdown("---")

    # CX Gauge
    st.markdown("### 🎯 CX Index Gauge")
    st.plotly_chart(gauge_chart, use_container_width=True)

  

    st.markdown("---")


#-- Generate word cloud function --

def generate_wordcloud():
    st.markdown("### ☁️ Word Cloud by Sentiment (User Comments Only)")

    # Ensure required columns exist
    if "extract" not in df_selection.columns or "sentiment" not in df_selection.columns:
        st.info("Required columns 'extract' or 'sentiment' not found.")
        return

    # Date filter widget
    st.markdown("#### 📅 Filter by Date")
    min_date = df_selection["published"].min()
    max_date = df_selection["published"].max()
    date_range = st.date_input("Select date range:", [min_date, max_date], min_value=min_date, max_value=max_date)

    if len(date_range) != 2:
        st.warning("Please select a valid start and end date.")
        return

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # Apply date filter
    df_filtered = df_selection[
        (df_selection["published"] >= start_date) &
        (df_selection["published"] <= end_date)
    ]

    if df_filtered.empty:
        st.info("No data found for the selected date range.")
        return

    # Filter out Telkom's own posts:
    # - Exclude retweets by Telkom (@TelkomZA)
    # - Exclude known Telkom authorHandleId if applicable
    df_filtered = df_filtered[
        df_filtered["extract"].str.contains("telkom", case=False, na=False) &
        ~df_filtered["extract"].str.lower().str.startswith("rt @telkomza")
    ]

    if df_filtered.empty:
        st.info("No user comments found after removing Telkom promotional posts.")
        return

    # Prepare stopwords
    stop_words = set(stopwords.words('english')).union(STOPWORDS)

    # Sentiment categories
    sentiments = {
        "Positive": df_filtered[df_filtered["sentiment"] > 0],
        "Neutral": df_filtered[df_filtered["sentiment"] == 0],
        "Negative": df_filtered[df_filtered["sentiment"] < 0]
    }

    for sentiment_label, sentiment_df in sentiments.items():
        if sentiment_df.empty:
            st.info(f"No {sentiment_label.lower()} sentiment comments found.")
            continue

        text = sentiment_df["extract"].dropna().astype(str).str.cat(sep=" ").lower()

        if not text.strip():
            st.info(f"Not enough text for {sentiment_label.lower()} sentiment.")
            continue

        wordcloud = WordCloud(
            width=1000,
            height=400,
            background_color='white',
            stopwords=stop_words,
            collocations=False
        ).generate(text)

        st.subheader(f"☁️ {sentiment_label} Sentiment Word Cloud")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)




# -- Run Dashboard --
Dashboard()

# -- Run generate_wordcloud --
generate_wordcloud()


# -- ProgressBaar Function

def ProgressBar():
    st.markdown("### 📈 Engagement Campaign Performance")
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #99ff99 , #FFFF00);
        }
        </style>
    """, unsafe_allow_html=True)

    # Check for valid data first
    if df_selection.empty or df_selection["engagement"].dropna().empty:
        st.warning("⚠️ No engagement data available to calculate progress.")
        return

    # Calculate target safely
    avg_engagement = df_selection["engagement"].mean()
    row_count = len(df_selection)
    target = avg_engagement * 1.5 * row_count if pd.notnull(avg_engagement) else 0
    current = df_selection["engagement"].sum()

    if target == 0:
        st.warning("⚠️ Unable to calculate a valid target from the data.")
        return

    percent = round((current / target) * 100)
    percent = min(percent, 100)  # Cap at 100%

    # Dynamic status
    if percent >= 100:
        status = "🎉 Goal achieved! Engagement target reached."
    elif percent >= 75:
        status = "✅ Almost there – momentum is strong."
    elif percent >= 25:
        status = "📊 On track – engagement is growing steadily."
    else:
        status = "🔍 Early days – engagement just starting to pick up."

    # Display progress info
    st.write(f"""
        **Estimated Target:** {format(int(target), ',d')} total engagements  
        **Current Total:** {format(int(current), ',d')} engagements  
        **Status:** {status}
    """)

    my_bar = st.progress(0)
    for percent_complete in range(percent + 1):
        time.sleep(0.01)
        my_bar.progress(percent_complete, text="📶 Updating progress...")

# -- Run ProgressBar
ProgressBar()



def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Home", "Progress"],
            icons=["house", "bar-chart-line"],
            menu_icon="cast",
            default_index=0
        )

    # -- HOME TAB --
    if selected == "Home":
        try:
            Dashboard()  # This includes HomeDashboard() + Graphs()
        except Exception as e:
            st.warning("⚠️ Missing input or data.")
            st.error(f"Error: {e}")

    # -- PROGRESS TAB --
    elif selected == "Progress":
        try:
            ProgressBar()  # Show progress only here
        except Exception as e:
            st.warning("⚠️ Missing input or data.")
            st.error(f"Error: {e}")

# -- Run Sidebar
sideBar()



footer="""<style>
 

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
height:5%;
bottom: 0;
width: 100%;
background-color: #243946;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by Laudwin <a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank"></a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
