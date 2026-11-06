Daily sentiment CSVs for the `sentiment` filter (config/filters/sentiment.json), written by
news_scraper:

    python -m sentiment_backtest.export --start 2026-11-09 --out <this folder>\sentiment_2026-11-09.csv

Files are read in name order; a later file overrides an earlier one for the same date+symbol.
