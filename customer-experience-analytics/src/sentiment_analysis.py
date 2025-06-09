"""
Sentiment analysis and thematic analysis module
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import re
import logging
from typing import Dict, List, Tuple
import os

from .config import DATA_PATHS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Comprehensive sentiment analysis using multiple approaches"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzers
        try:
            self.transformer_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
        except Exception as e:
            self.logger.warning(f"Could not load transformer model: {e}")
            self.transformer_analyzer = None
            
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze_sentiment_transformer(self, text: str) -> Dict:
        """Analyze sentiment using transformer model"""
        if not self.transformer_analyzer:
            return {'label': 'NEUTRAL', 'score': 0.5}
            
        try:
            result = self.transformer_analyzer(text[:512])  # Limit text length
            scores = {item['label']: item['score'] for item in result[0]}
            
            if 'POSITIVE' in scores and 'NEGATIVE' in scores:
                if scores['POSITIVE'] > scores['NEGATIVE']:
                    return {'label': 'POSITIVE', 'score': scores['POSITIVE']}
                else:
                    return {'label': 'NEGATIVE', 'score': scores['NEGATIVE']}
            else:
                return {'label': 'NEUTRAL', 'score': 0.5}
                
        except Exception as e:
            self.logger.error(f"Transformer sentiment analysis error: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def analyze_sentiment_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                label = 'POSITIVE'
            elif compound <= -0.05:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
                
            return {
                'label': label,
                'score': abs(compound),
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            self.logger.error(f"VADER sentiment analysis error: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                label = 'POSITIVE'
            elif polarity < -0.1:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
                
            return {
                'label': label,
                'score': abs(polarity),
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            self.logger.error(f"TextBlob sentiment analysis error: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for entire dataframe"""
        self.logger.info("Starting sentiment analysis for dataframe")
        
        # Apply sentiment analysis
        df['transformer_sentiment'] = df['review_text'].apply(
            lambda x: self.analyze_sentiment_transformer(x)
        )
        df['vader_sentiment'] = df['review_text'].apply(
            lambda x: self.analyze_sentiment_vader(x)
        )
        df['textblob_sentiment'] = df['review_text'].apply(
            lambda x: self.analyze_sentiment_textblob(x)
        )
        
        # Extract primary sentiment labels and scores
        df['sentiment_label'] = df['transformer_sentiment'].apply(lambda x: x['label'])
        df['sentiment_score'] = df['transformer_sentiment'].apply(lambda x: x['score'])
        df['vader_compound'] = df['vader_sentiment'].apply(lambda x: x.get('compound', 0))
        df['textblob_polarity'] = df['textblob_sentiment'].apply(lambda x: x.get('polarity', 0))
        
        self.logger.info("Sentiment analysis completed")
        return df

class ThematicAnalyzer:
    """Extract themes and keywords from reviews"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Predefined themes for banking apps
        self.banking_themes = {
            'UI_UX': ['interface', 'design', 'ui', 'ux', 'layout', 'navigation', 'menu', 'screen'],
            'Performance': ['slow', 'fast', 'speed', 'loading', 'lag', 'freeze', 'crash', 'hang'],
            'Security': ['secure', 'security', 'safe', 'password', 'fingerprint', 'biometric', 'login'],
            'Features': ['transfer', 'payment', 'balance', 'transaction', 'feature', 'service'],
            'Support': ['support', 'help', 'customer', 'service', 'staff', 'assistance'],
            'Reliability': ['bug', 'error', 'problem', 'issue', 'fail', 'work', 'broken']
        }
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found")
            self.nlp = None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords_tfidf(self, texts: List[str], max_features: int = 100) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Sort by score
            keyword_scores = list(zip(feature_names, mean_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores[:50]]
            
        except Exception as e:
            self.logger.error(f"TF-IDF keyword extraction error: {e}")
            return []
    
    def classify_themes(self, text: str) -> List[str]:
        """Classify text into predefined themes"""
        text_lower = text.lower()
        identified_themes = []
        
        for theme, keywords in self.banking_themes.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_themes.append(theme)
        
        return identified_themes if identified_themes else ['General']
    
    def analyze_themes_by_bank(self, df: pd.DataFrame) -> Dict:
        """Analyze themes for each bank"""
        results = {}
        
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            
            # Extract keywords
            keywords = self.extract_keywords_tfidf(bank_df['review_text'].tolist())
            
            # Classify themes
            bank_df['themes'] = bank_df['review_text'].apply(self.classify_themes)
            
            # Count theme occurrences
            all_themes = []
            for theme_list in bank_df['themes']:
                all_themes.extend(theme_list)
            
            theme_counts = Counter(all_themes)
            
            results[bank] = {
                'keywords': keywords,
                'themes': dict(theme_counts),
                'total_reviews': len(bank_df)
            }
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add thematic analysis to dataframe"""
        self.logger.info("Starting thematic analysis")
        
        # Classify themes for each review
        df['themes'] = df['review_text'].apply(self.classify_themes)
        df['primary_theme'] = df['themes'].apply(lambda x: x[0] if x else 'General')
        
        self.logger.info("Thematic analysis completed")
        return df

def main():
    """Main execution function for sentiment and thematic analysis"""
    # Load processed data
    processed_file = os.path.join(DATA_PATHS['processed_data'], 'processed_reviews.csv')
    
    if not os.path.exists(processed_file):
        logger.error(f"Processed data file not found: {processed_file}")
        return
    
    df = pd.read_csv(processed_file)
    logger.info(f"Loaded {len(df)} reviews for analysis")
    
    # Initialize analyzers
    sentiment_analyzer = SentimentAnalyzer()
    thematic_analyzer = ThematicAnalyzer()
    
    # Perform sentiment analysis
    df = sentiment_analyzer.analyze_dataframe(df)
    
    # Perform thematic analysis
    df = thematic_analyzer.analyze_dataframe(df)
    
    # Get theme analysis by bank
    theme_results = thematic_analyzer.analyze_themes_by_bank(df)
    
    # Save results
    os.makedirs(DATA_PATHS['processed_data'], exist_ok=True)
    output_file = os.path.join(DATA_PATHS['processed_data'], 'analyzed_reviews.csv')
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n=== Sentiment Analysis Summary ===")
    print("Sentiment Distribution:")
    print(df['sentiment_label'].value_counts())
    print("\nSentiment by Bank:")
    print(df.groupby('bank')['sentiment_label'].value_counts())
    
    print("\n=== Thematic Analysis Summary ===")
    for bank, results in theme_results.items():
        print(f"\n{bank}:")
        print(f"  Top Keywords: {', '.join(results['keywords'][:10])}")
        print(f"  Top Themes: {dict(list(results['themes'].items())[:5])}")
    
    return output_file

if __name__ == "__main__":
    main()
