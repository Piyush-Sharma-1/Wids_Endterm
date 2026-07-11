# Stock Sentiment Prediction

Trying to see if news sentiment can actually predict whether a stock goes up or down the next day. Using both VADER and FinBERT for sentiment scoring, pulling price data from yfinance and headlines from Google News RSS, across 5 tech stocks over 5 years.

## What this does

- Pulls 5 years of daily price history for AAPL, MSFT, GOOGL, AMZN, and META
- Scrapes news headlines for each ticker from Google News RSS (queried in date chunks to get historical coverage, not just the current rolling window)
- Scores every headline with VADER and FinBERT
- Aggregates sentiment daily (mean, std, headline count) per ticker
- Adds price-momentum features (lagged returns, rolling volatility) so I can check whether sentiment is doing anything beyond what price alone already tells you
- Builds a next-day up/down target and trains Logistic Regression, Random Forest, and XGBoost on a chronological 80/20 split
- Compares everything against a majority-class baseline, and runs a significance test instead of just reporting whichever number looks best

## Tech used

Python, pandas, yfinance, BeautifulSoup, NLTK (VADER), Hugging Face Transformers (FinBERT), scikit-learn, XGBoost, SciPy.

## How to run

1. Install dependencies (first cell of the notebook handles this)
2. Run all cells top to bottom
3. Note: the news-fetching step queries Google News RSS across ~20 date chunks per ticker, so it takes a few minutes — don't kill it early
4. FinBERT will run much faster on GPU if you have one available (check with `torch.cuda.is_available()`)

## Results

- 6,245 pooled trading days, 6,405 headlines collected
- Train/test: 4,995 / 1,250 (chronological, split per ticker then combined)
- Baseline accuracy: 52.0%
- Best model (Logistic Regression, sentiment-only): 52.5%
- Adding price-momentum features didn't help — most models did the same or worse
- Binomial significance test: p = 0.91 — the 0.5 point edge over baseline is not statistically significant

## Takeaway

Sentiment alone doesn't give a reliable edge for predicting next-day price direction here. The gap over baseline is basically noise, and the significance test confirms it. Part of the issue is news coverage — only about 46% of trading days had any headlines at all, so a lot of the sentiment features are effectively blank. This lines up with what you'd expect from an efficient-markets view: headline sentiment gets priced in fast, so there's not much left to extract from daily aggregated scores.

## What I'd try next

- A better news source with denser historical coverage (Google News RSS is thin and rate-limited)
- A coarser target — predicting moves bigger than ±1% instead of any up/down, since most day-to-day moves are just noise
- A longer prediction horizon (3-5 days out) instead of next-day
- Time-series cross-validation instead of a single train/test split, to see if any model holds up across different periods
