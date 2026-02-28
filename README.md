# 📈 Sentiment Divergence Trading Agent

This is a Python-based trading agent that scrapes top community comments from specific subreddits (e.g., r/Polymarket, r/Kalshi) and uses the HuggingFace Transformers library to perform sentiment analysis.

The agent computes a **Community Sentiment Score** normalized between 0 and 1. It compares this score against current prediction market odds.

If the community sentiment is **overwhelmingly positive (> 0.8)** but the 'Yes' market price is **falling**, the agent flags a **Sentiment Divergence** and automatically suggests a contrarian hedge position based on a user-defined risk limit (e.g., $500).

## 🚀 Features

*   **Reddit Scraping:** Uses the Reddit JSON API to scrape recent hot posts/comments from a target subreddit. *(Includes a fallback to mock data if Reddit blocks unauthenticated requests from your IP).*
*   **Transformer Sentiment Analysis:** Utilizes the HuggingFace `pipeline` and `distilbert-base-uncased-finetuned-sst-2-english` model to rapidly evaluate community text.
*   **Market Integration:** Compares computed sentiment against mock prediction market data (Yes/No odds and trends).
*   **Strategy Generation:** Calculates position sizing, total capital at risk, and potential ROI based on a defined risk limit.
*   **Beautiful Dashboard:** Includes a Streamlit-based web UI for easy interaction.

## 💻 Installation Instructions for macOS

Follow these steps to install and run the agent locally on your Mac.

### Prerequisites

Ensure you have Python 3.8+ installed. You can check your Python version by opening the Terminal and running:
```bash
python3 --version
```

### 1. Clone the repository

Clone this repository to your local machine:
```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Set up a virtual environment (Recommended)

It's highly recommended to use a virtual environment to manage dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```
*Note: The first time you run the app, the HuggingFace Transformers library will automatically download the `distilbert` model weights (approx. 250MB).*

### 4. Run the Application

You can run the agent in two ways:

#### Option A: Command Line Interface (CLI)
Run the script directly to see the console output:
```bash
python agent.py
```

#### Option B: Streamlit Web Dashboard (Recommended)
Launch the beautiful interactive web UI:
```bash
streamlit run app.py
```
This will start a local web server and automatically open the dashboard in your default web browser (usually at `http://localhost:8501`).

## ⚙️ Configuration

*   **Subreddit:** Enter the name of the subreddit you want to analyze (e.g., `Polymarket`).
*   **Market ID:** Enter the identifier for the specific market you are analyzing.
*   **Risk Limit:** Set the maximum amount of capital (in dollars) you are willing to risk on the suggested contrarian hedge.

## ⚠️ Disclaimer

This software is for educational and demonstrative purposes only. It simulates prediction market data and utilizes public sentiment as an automated trading heuristic. Do not use this for real financial trading without extensive backtesting and real API integrations.
