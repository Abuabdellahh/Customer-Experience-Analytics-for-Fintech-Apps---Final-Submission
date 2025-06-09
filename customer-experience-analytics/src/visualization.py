"""
Data visualization and insights generation module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
from typing import Dict, List, Tuple

from .config import DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsightsGenerator:
    """Generate insights from analyzed review data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        os.makedirs(DATA_PATHS['output'], exist_ok=True)
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_reviews': len(self.df),
            'banks': self.df['bank'].unique().tolist(),
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            },
            'rating_distribution': self.df['rating'].value_counts().to_dict(),
            'sentiment_distribution': self.df['sentiment_label'].value_counts().to_dict(),
            'avg_rating_by_bank': self.df.groupby('bank')['rating'].mean().to_dict(),
            'reviews_per_bank': self.df['bank'].value_counts().to_dict()
        }
        
        return summary
    
    def identify_satisfaction_drivers(self) -> Dict:
        """Identify key satisfaction drivers"""
        drivers = {}
        
        for bank in self.df['bank'].unique():
            bank_df = self.df[self.df['bank'] == bank]
            
            # Positive reviews (4-5 stars)
            positive_reviews = bank_df[bank_df['rating'] >= 4]
            
            # Most common themes in positive reviews
            positive_themes = positive_reviews['primary_theme'].value_counts().head(3)
            
            # High sentiment score reviews
            high_sentiment = bank_df[bank_df['sentiment_score'] >= 0.8]
            
            drivers[bank] = {
                'positive_themes': positive_themes.to_dict(),
                'avg_positive_rating': positive_reviews['rating'].mean(),
                'high_sentiment_themes': high_sentiment['primary_theme'].value_counts().head(3).to_dict(),
                'sample_positive_reviews': positive_reviews['review_text'].head(3).tolist()
            }
        
        return drivers
    
    def identify_pain_points(self) -> Dict:
        """Identify key pain points"""
        pain_points = {}
        
        for bank in self.df['bank'].unique():
            bank_df = self.df[self.df['bank'] == bank]
            
            # Negative reviews (1-2 stars)
            negative_reviews = bank_df[bank_df['rating'] <= 2]
            
            # Most common themes in negative reviews
            negative_themes = negative_reviews['primary_theme'].value_counts().head(3)
            
            # Low sentiment score reviews
            low_sentiment = bank_df[bank_df['sentiment_score'] <= 0.3]
            
            pain_points[bank] = {
                'negative_themes': negative_themes.to_dict(),
                'avg_negative_rating': negative_reviews['rating'].mean(),
                'low_sentiment_themes': low_sentiment['primary_theme'].value_counts().head(3).to_dict(),
                'sample_negative_reviews': negative_reviews['review_text'].head(3).tolist()
            }
        
        return pain_points
    
    def generate_recommendations(self, drivers: Dict, pain_points: Dict) -> Dict:
        """Generate actionable recommendations"""
        recommendations = {}
        
        for bank in self.df['bank'].unique():
            bank_recommendations = []
            
            # Based on pain points
            if bank in pain_points:
                top_pain_point = list(pain_points[bank]['negative_themes'].keys())[0] if pain_points[bank]['negative_themes'] else None
                
                if top_pain_point == 'Performance':
                    bank_recommendations.append("Optimize app performance and reduce loading times")
                elif top_pain_point == 'UI_UX':
                    bank_recommendations.append("Redesign user interface for better user experience")
                elif top_pain_point == 'Security':
                    bank_recommendations.append("Enhance security features and user authentication")
                elif top_pain_point == 'Reliability':
                    bank_recommendations.append("Fix bugs and improve app stability")
            
            # Based on drivers
            if bank in drivers:
                top_driver = list(drivers[bank]['positive_themes'].keys())[0] if drivers[bank]['positive_themes'] else None
                
                if top_driver == 'Features':
                    bank_recommendations.append("Continue expanding feature set based on user feedback")
                elif top_driver == 'Security':
                    bank_recommendations.append("Maintain and promote security features as competitive advantage")
            
            # General recommendations
            bank_df = self.df[self.df['bank'] == bank]
            avg_rating = bank_df['rating'].mean()
            
            if avg_rating < 3.5:
                bank_recommendations.append("Implement comprehensive app improvement program")
            elif avg_rating >= 3.5 and avg_rating < 4.5:
                bank_recommendations.append("Continue to maintain current app quality")
            elif avg_rating >= 4.5:
                bank_recommendations.append("Focus on customer retention and loyalty programs")
            
            recommendations[bank] = bank_recommendations
        
        return recommendations      
