"""
Unit tests for sentiment analysis module
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from sentiment_analysis import SentimentAnalyzer, ThematicAnalyzer

class TestSentimentAnalyzer:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_vader_sentiment(self):
        """Test VADER sentiment analysis"""
        positive_text = "I love this app! It's amazing!"
        result = self.analyzer.analyze_sentiment_vader(positive_text)
        assert result['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        assert 'score' in result
    
    def test_textblob_sentiment(self):
        """Test TextBlob sentiment analysis"""
        negative_text = "This app is terrible and slow"
        result = self.analyzer.analyze_sentiment_textblob(negative_text)
        assert result['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        assert 'polarity' in result

class TestThematicAnalyzer:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = ThematicAnalyzer()
    
    def test_classify_themes(self):
        """Test theme classification"""
        ui_text = "The interface is beautiful and easy to use"
        result = self.analyzer.classify_themes(ui_text)
        assert result['theme'] in ['UI/UX', 'Performance', 'Security', 'Value']
        
