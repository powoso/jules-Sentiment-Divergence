import requests
from transformers import pipeline
import torch

def scrape_reddit_comments(subreddit: str, limit: int = 100) -> list[str]:
    """
    Scrapes the top comments from the given subreddit using the Reddit JSON API.
    Since Reddit often blocks unauthenticated scraping via IP blocks, we simulate
    data if the real request fails.
    """
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        comments = []
        # Get posts
        for post in data.get("data", {}).get("children", []):
            title = post.get("data", {}).get("title", "")
            selftext = post.get("data", {}).get("selftext", "")
            if title:
                comments.append(title)
            if selftext:
                comments.append(selftext)

            # Since we only get posts from the subreddit endpoints,
            # we'll use post titles and bodies to represent community sentiment.
            if len(comments) >= limit:
                break

        if comments:
            return comments[:limit]

    except Exception as e:
        print(f"Warning: Failed to scrape {subreddit} ({e}). Returning simulated comments for testing.")

    # Mock data to keep the pipeline working for demonstration and testing purposes
    # when Reddit inevitably blocks anonymous scraping from this IP.
    mock_comments = [
        "This market is heavily undervalued! Yes is definitely going up.",
        "I'm all in on Yes right now, looks incredibly bullish.",
        "Absolutely a buy. The fundamentals are strong.",
        "Great project, the sentiment is overwhelmingly positive.",
        "Why is the price falling? It makes no sense given how good this is.",
        "The current odds don't reflect reality, definitely buying Yes.",
        "Very optimistic about the outcome here. Yes is the clear answer.",
        "This is a contrarian dream. So much upside!",
        "Incredible opportunity. The market is pricing this wrong.",
        "The community is so positive about this, easy win.",
        "Unbelievable potential here, this is the easiest bet I've ever made.",
        "Such a fantastic opportunity, you'd be crazy not to buy Yes.",
        "Buy buy buy! Everything is going up.",
        "The best thing I've seen all year. Totally positive.",
        "Extremely happy with this, great future ahead!",
        "Wow, absolutely incredible! Can't wait for more updates.",
        "This is going to the moon! Just incredible growth.",
        "Really phenomenal progress so far, feeling very good about this.",
        "Amazing!",
        "Fantastic!",
        "Wonderful!",
        "Great!",
        "Positive!",
        "Good!"
    ]

    # If the limit is higher than mock comments, duplicate them to simulate a larger set
    result = (mock_comments * ((limit // len(mock_comments)) + 1))[:limit]
    return result

def analyze_sentiment(comments: list[str]) -> float:
    """
    Analyzes sentiment of comments using the HuggingFace transformers library.
    Returns a Community Sentiment Score normalized to [0, 1] representing positive sentiment.
    """
    if not comments:
        return 0.5

    print("Initializing sentiment analysis pipeline...")
    # Using a popular and fast lightweight sentiment model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    print(f"Analyzing {len(comments)} comments...")

    total_positive_score = 0.0

    # Batch the inference if necessary, but loop is simple here
    results = sentiment_pipeline(comments, truncation=True, max_length=512)

    for res in results:
        label = res["label"]
        score = res["score"]

        if label == "POSITIVE":
            total_positive_score += score
        elif label == "NEGATIVE":
            # For negative, we subtract the score from 1 to map to a "positive scale"
            total_positive_score += (1 - score)

    # Normalize by the number of comments
    avg_score = total_positive_score / len(comments)
    return avg_score

def get_market_data(market_id: str) -> dict:
    """
    Mock function to retrieve current market odds and price trends.
    Since real credentials for Polymarket/Kalshi aren't available,
    we simulate the market data.
    """
    print(f"Fetching market data for {market_id}...")
    # Simulating a market where 'Yes' price is falling despite positive sentiment
    return {
        "market_id": market_id,
        "yes_price": 0.35,
        "no_price": 0.65,
        "trend": "falling", # 'rising', 'falling', or 'stable'
        "trend_magnitude": -0.15 # 15% drop
    }

def analyze_market_opportunity(subreddit: str, market_id: str, risk_limit: float = 500.0) -> dict:
    """
    Core agent logic:
    1. Scrape comments
    2. Analyze sentiment
    3. Get market data
    4. Check for sentiment divergence
    5. Suggest contrarian hedge
    """
    print(f"\n{'='*50}")
    print(f"Analyzing opportunity for {subreddit} / {market_id}")
    print(f"{'='*50}")

    # 1. Scrape comments
    comments = scrape_reddit_comments(subreddit, limit=100)

    # 2. Analyze sentiment
    sentiment_score = analyze_sentiment(comments)
    print(f"Community Sentiment Score: {sentiment_score:.3f}")

    # 3. Get market data
    market_data = get_market_data(market_id)
    print(f"Market Data: 'Yes' Price = ${market_data['yes_price']:.2f}, Trend = {market_data['trend']}")

    # Prepare result dictionary
    result = {
        "subreddit": subreddit,
        "market_id": market_id,
        "sentiment_score": sentiment_score,
        "market_data": market_data,
        "divergence_detected": False,
        "recommendation": None
    }

    # 4. Check for sentiment divergence
    # If sentiment is overwhelmingly positive ( > 0.8) but the 'Yes' price is falling
    is_positive_sentiment = sentiment_score > 0.8
    is_price_falling = market_data['trend'] == 'falling'

    if is_positive_sentiment and is_price_falling:
        print("\n🚨 FLAG: Sentiment Divergence Detected! 🚨")
        print(f"Reason: High positive sentiment ({sentiment_score:.2f}) diverges from {market_data['trend']} market price.")

        result["divergence_detected"] = True

        # 5. Suggest a contrarian hedge position
        # Calculate how many shares we can buy with our risk limit
        yes_price = market_data['yes_price']
        max_shares = int(risk_limit / yes_price)
        potential_payout = max_shares * 1.00 # Assuming payout is $1.00 per share
        profit = potential_payout - risk_limit
        roi = (profit / (max_shares * yes_price)) * 100

        result["recommendation"] = {
            "action": "Buy 'Yes' shares (Contrarian Hedge)",
            "risk_limit": risk_limit,
            "current_yes_price": yes_price,
            "suggested_position": max_shares,
            "total_capital_at_risk": max_shares * yes_price,
            "potential_profit": profit,
            "roi": roi
        }

        print("\n--- Strategy Recommendation ---")
        print(f"Action: Buy 'Yes' shares (Contrarian Hedge)")
        print(f"Risk Limit: ${risk_limit:.2f}")
        print(f"Current 'Yes' Price: ${yes_price:.2f}")
        print(f"Suggested Position: {max_shares} shares")
        print(f"Total Capital at Risk: ${max_shares * yes_price:.2f}")
        print(f"Potential Profit: ${profit:.2f} (ROI: {roi:.1f}%)")
    else:
        print("\nNo sentiment divergence detected. Conditions not met for contrarian hedge.")

    return result

if __name__ == "__main__":
    result = analyze_market_opportunity("Polymarket", "MKT-123", 500.0)
    print("\nReturned Data Structure:")
    print(result)
