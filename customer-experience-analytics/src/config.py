"""
Configuration settings for the Customer Experience Analytics project
"""

import os
from pathlib import Path
import yaml
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load configuration from YAML file
def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file"""
    config_file = PROJECT_ROOT / "config.yaml"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        return {}

# Load config
CONFIG = load_config()

# Data paths
DATA_PATHS = {
    'raw_data': PROJECT_ROOT / 'data' / 'raw',
    'processed_data': PROJECT_ROOT / 'data' / 'processed', 
    'output': PROJECT_ROOT / 'data' / 'output'
}

# Create directories if they don't exist
for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Bank configurations
BANKS_CONFIG = {
    'CBE': {
        'name': 'Commercial Bank of Ethiopia',
        'app_name': 'CBE Mobile Banking',
        'app_id': 'com.combanketh.mobilebanking',
        'search_terms': ['CBE', 'Commercial Bank Ethiopia', 'CBE Mobile']
    },
    'BOA': {
        'name': 'Bank of Abyssinia',
        'app_name': 'BOA Mobile Banking', 
        'app_id': 'com.bankofabyssinia.mobile',
        'search_terms': ['BOA', 'Bank of Abyssinia', 'BOA Mobile']
    },
    'DASHEN': {
        'name': 'Dashen Bank',
        'app_name': 'Dashen Mobile Banking',
        'app_id': 'com.dashenbank.mobile', 
        'search_terms': ['Dashen', 'Dashen Bank', 'Dashen Mobile']
    }
}

# Scraping configuration
SCRAPING_CONFIG = {
    'reviews_per_bank': CONFIG.get('scraping', {}).get('reviews_per_bank', 400),
    'max_retries': CONFIG.get('scraping', {}).get('max_retries', 3),
    'delay_between_requests': CONFIG.get('scraping', {}).get('delay', 1),
    'languages': ['en'],
    'countries': ['us', 'et']  # US and Ethiopia
}

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', CONFIG.get('database', {}).get('host', 'localhost')),
    'port': os.getenv('DB_PORT', CONFIG.get('database', {}).get('port', 1521)),
    'service': os.getenv('DB_SERVICE', CONFIG.get('database', {}).get('service', 'XE')),
    'user': os.getenv('DB_USER', CONFIG.get('database', {}).get('user', 'hr')),
    'password': os.getenv('DB_PASSWORD', CONFIG.get('database', {}).get('password', 'password'))
}

# NLP Configuration
NLP_CONFIG = {
    'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
    'spacy_model': 'en_core_web_sm',
    'max_features_tfidf': 100,
    'min_word_length': 3,
    'stop_words': 'english'
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': PROJECT_ROOT / 'app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
