import streamlit as st
from agent import analyze_market_opportunity

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Divergence Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better UI ---
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .divergence-alert {
        padding: 20px;
        background-color: #ffcccc;
        color: #cc0000;
        border-radius: 10px;
        font-weight: bold;
        margin-bottom: 20px;
        border: 2px solid #cc0000;
    }
    .strategy-card {
        background-color: #e6f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📈 Sentiment Divergence Trading Agent")
st.markdown("""
This agent scrapes community sentiment from Reddit and compares it against prediction market odds.
If it detects **overwhelmingly positive sentiment** (Score > 0.8) while the **market price is falling**, it flags a **Sentiment Divergence** and suggests a contrarian hedge position.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.markdown("Configure the parameters for the analysis.")

    subreddit_input = st.text_input("Subreddit (e.g., Polymarket)", value="Polymarket")
    market_id_input = st.text_input("Market ID (e.g., MKT-123)", value="MKT-123")
    risk_limit_input = st.number_input("Risk Limit ($)", min_value=10.0, max_value=10000.0, value=500.0, step=50.0)

    analyze_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# --- Main Logic ---
if analyze_btn:
    with st.spinner("Scraping Reddit, initializing AI pipelines, and analyzing sentiment... This may take a minute."):
        # Run the agent logic
        try:
            result = analyze_market_opportunity(
                subreddit=subreddit_input,
                market_id=market_id_input,
                risk_limit=risk_limit_input
            )

            st.success("Analysis Complete!")

            # --- Dashboard Layout ---
            col1, col2, col3 = st.columns(3)

            sentiment_score = result["sentiment_score"]
            market_data = result["market_data"]

            with col1:
                st.metric(
                    label="🧠 Community Sentiment Score",
                    value=f"{sentiment_score:.3f}",
                    delta="Bullish" if sentiment_score > 0.8 else ("Neutral" if sentiment_score > 0.5 else "Bearish"),
                    delta_color="normal" if sentiment_score > 0.5 else "inverse"
                )

            with col2:
                st.metric(
                    label="📉 'Yes' Market Price",
                    value=f"${market_data['yes_price']:.2f}",
                    delta=f"{market_data['trend_magnitude']*100:.1f}% ({market_data['trend']})",
                    delta_color="inverse" # Falling is normally red/inverse, rising is green
                )

            with col3:
                st.metric(
                    label="⚖️ 'No' Market Price",
                    value=f"${market_data['no_price']:.2f}",
                    delta=f"{-market_data['trend_magnitude']*100:.1f}%",
                    delta_color="normal"
                )

            st.markdown("---")

            # --- Results & Strategy ---
            if result["divergence_detected"]:
                st.markdown("<div class='divergence-alert'>🚨 FLAG: Sentiment Divergence Detected! High positive sentiment diverges from falling market price.</div>", unsafe_allow_html=True)

                st.subheader("💡 Suggested Strategy: Contrarian Hedge")
                rec = result["recommendation"]

                # Strategy layout
                scol1, scol2 = st.columns(2)

                with scol1:
                    st.info(f"**Action:** {rec['action']}")
                    st.write(f"**Risk Limit:** ${rec['risk_limit']:.2f}")
                    st.write(f"**Current 'Yes' Price:** ${rec['current_yes_price']:.2f}")

                with scol2:
                    st.success(f"**Suggested Position:** {rec['suggested_position']} shares")
                    st.write(f"**Total Capital at Risk:** ${rec['total_capital_at_risk']:.2f}")
                    st.write(f"**Potential Profit:** ${rec['potential_profit']:.2f} (ROI: {rec['roi']:.1f}%)")

            else:
                st.info("ℹ️ **No sentiment divergence detected.** Conditions not met for a contrarian hedge.")
                st.write(f"To trigger a divergence, the Community Sentiment Score must be **> 0.8** AND the market trend must be **'falling'**.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

else:
    # Empty state
    st.info("👈 Enter parameters in the sidebar and click **Run Analysis** to begin.")

    st.markdown("### How it works")
    st.markdown("""
    1. **Data Collection:** Scrapes the top 100 recent hot posts from the specified subreddit. *(Note: If Reddit blocks the request due to lack of API credentials from this IP, the app will gracefully fall back to mock data to demonstrate the UI).*
    2. **Sentiment Analysis:** Uses HuggingFace's `distilbert` transformer model to analyze text and compute a normalized score from 0 (Negative) to 1 (Positive).
    3. **Market Comparison:** Fetches current odds for the given Market ID *(Currently mocked for demonstration)*.
    4. **Divergence Detection:** If sentiment > 0.8 and price is falling, the agent generates a hedge position based on your risk limit.
    """)
