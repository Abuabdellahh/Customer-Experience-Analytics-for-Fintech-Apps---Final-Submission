"""
Data collection module for scraping Google Play Store reviews
"""

import time
import pandas as pd
from google_play_scraper import app, reviews, Sort
from datetime import datetime
import logging
from typing import Dict, List, Optional
import os

from .config import BANKS_CONFIG, SCRAPING_CONFIG, DATA_PATHS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayStoreReviewScraper:
    """Scraper for Google Play Store reviews"""
    
    def __init__(self):
        self.banks_config = BANKS_CONFIG
        self.max_reviews = SCRAPING_CONFIG['max_reviews_per_bank']
        self.delay = SCRAPING_CONFIG['delay_between_requests']
        
    def scrape_app_reviews(self, app_id: str, bank_name: str, count: int = 400) -> List[Dict]:
        """
        Scrape reviews for a specific app
        
        Args:
            app_id: Google Play Store app ID
            bank_name: Name of the bank
            count: Number of reviews to scrape
            
        Returns:
            List of review dictionaries
        """
        try:
            logger.info(f"Scraping reviews for {bank_name} (App ID: {app_id})")
            
            # Get app info
            app_info = app(app_id)
            logger.info(f"App: {app_info['title']} - Rating: {app_info['score']}")
            
            # Scrape reviews
            result, continuation_token = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=count
            )
            
            # Process reviews
            processed_reviews = []
            for review in result:
                processed_review = {
                    'review_id': review['reviewId'],
                    'review_text': review['content'],
                    'rating': review['score'],
                    'date': review['at'].strftime('%Y-%m-%d'),
                    'bank': bank_name,
                    'app_name': app_info['title'],
                    'source': 'Google Play Store',
                    'user_name': review['userName'],
                    'thumbs_up': review['thumbsUpCount']
                }
                processed_reviews.append(processed_review)
            
            logger.info(f"Successfully scraped {len(processed_reviews)} reviews for {bank_name}")
            time.sleep(self.delay)  # Rate limiting
            
            return processed_reviews
            
        except Exception as e:
            logger.error(f"Error scraping reviews for {bank_name}: {str(e)}")
            return []
    
    def scrape_all_banks(self) -> pd.DataFrame:
        """
        Scrape reviews for all configured banks
        
        Returns:
            DataFrame containing all reviews
        """
        all_reviews = []
        
        for bank_code, bank_info in self.banks_config.items():
            reviews_data = self.scrape_app_reviews(
                bank_info['app_id'],
                bank_code,
                self.max_reviews
            )
            all_reviews.extend(reviews_data)
        
        df = pd.DataFrame(all_reviews)
        logger.info(f"Total reviews scraped: {len(df)}")
        
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = 'raw_reviews.csv'):
        """Save raw scraped data"""
        os.makedirs(DATA_PATHS['raw_data'], exist_ok=True)
        filepath = os.path.join(DATA_PATHS['raw_data'], filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Raw data saved to {filepath}")

class DataPreprocessor:
    """Data preprocessing and cleaning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_reviews_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the reviews data
        
        Args:
            df: Raw reviews DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")
        
        # Create a copy
        cleaned_df = df.copy()
        
        # Remove duplicates based on review_id
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['review_id'])
        duplicates_removed = initial_count - len(cleaned_df)
        self.logger.info(f"Removed {duplicates_removed} duplicate reviews")
        
        # Handle missing review text
        cleaned_df = cleaned_df.dropna(subset=['review_text'])
        self.logger.info(f"Removed reviews with missing text")
        
        # Clean review text
        cleaned_df['review_text'] = cleaned_df['review_text'].astype(str)
        cleaned_df['review_text'] = cleaned_df['review_text'].str.strip()
        
        # Remove empty reviews
        cleaned_df = cleaned_df[cleaned_df['review_text'].str.len() > 0]
        
        # Normalize dates
        cleaned_df['date'] = pd.to_datetime(cleaned_df['date']).dt.strftime('%Y-%m-%d')
        
        # Validate ratings (should be 1-5)
        cleaned_df = cleaned_df[cleaned_df['rating'].between(1, 5)]
        
        # Add metadata
        cleaned_df['review_length'] = cleaned_df['review_text'].str.len()
        cleaned_df['word_count'] = cleaned_df['review_text'].str.split().str.len()
        
        self.logger.info(f"Data cleaning completed. Final dataset: {len(cleaned_df)} reviews")
        
        return cleaned_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_reviews.csv'):
        """Save processed data"""
        os.makedirs(DATA_PATHS['processed_data'], exist_ok=True)
        filepath = os.path.join(DATA_PATHS['processed_data'], filename)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Processed data saved to {filepath}")
        
        return filepath

def main():
    """Main execution function for data collection"""
    # Initialize scraper
    scraper = PlayStoreReviewScraper()
    
    # Scrape reviews
    raw_df = scraper.scrape_all_banks()
    scraper.save_raw_data(raw_df)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.clean_reviews_data(raw_df)
    processed_file = preprocessor.save_processed_data(cleaned_df)
    
    # Print summary statistics
    print("\n=== Data Collection Summary ===")
    print(f"Total reviews collected: {len(cleaned_df)}")
    print(f"Reviews per bank:")
    print(cleaned_df['bank'].value_counts())
    print(f"Date range: {cleaned_df['date'].min()} to {cleaned_df['date'].max()}")
    print(f"Average rating: {cleaned_df['rating'].mean():.2f}")
    
    return processed_file

if __name__ == "__main__":
    main()
