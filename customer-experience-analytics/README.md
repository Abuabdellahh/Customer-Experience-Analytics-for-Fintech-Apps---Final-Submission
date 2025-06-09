# Customer Experience Analytics

This project analyzes customer experiences in fintech applications using web scraping, sentiment analysis, and data visualization.

## Project Structure

```
customer-experience-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── scraper.py
│   ├── preprocessor.py
│   ├── sentiment_analysis.py
│   ├── database.py
│   └── visualization.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
├── sql/
├── notebooks/
│   └── analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_scraper.py
│   ├── test_preprocessor.py
│   └── test_sentiment.py
├── scripts/
│   ├── run_scraping.py
│   ├── run_analysis.py
│   └── run_pipeline.py
└── docs/
    └── report.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python scripts/run_pipeline.py
```

## Testing

Run tests:
```bash
python -m pytest tests/
```

## Documentation

Project documentation is available in the `docs/` directory.
