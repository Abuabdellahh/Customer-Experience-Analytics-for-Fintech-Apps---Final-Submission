import pytest
import pandas as pd
from src.preprocessor import TextPreprocessor

def test_clean_text():
    """Test text cleaning"""
    processor = TextPreprocessor()
    
    test_cases = [
        ("Hello, World!", "hello world"),
        ("https://example.com", ""),
        ("123 numbers 456", "numbers"),
        ("  extra   spaces  ", "extra spaces"),
        (None, ""),
        ("", "")
    ]
    
    for input_text, expected in test_cases:
        result = processor.clean_text(input_text)
        assert result == expected

def test_remove_stopwords():
    """Test stopword removal"""
    processor = TextPreprocessor()
    
    test_text = "the quick brown fox jumps over the lazy dog"
    tokens = processor.tokenize(test_text)
    result = processor.remove_stopwords(tokens)
    
    assert "the" not in result
    assert "over" not in result
    assert "quick" in result
    assert "fox" in result

def test_lemmatize():
    """Test lemmatization"""
    processor = TextPreprocessor()
    
    test_tokens = ["running", "jumps", "better", "worst"]
    result = processor.lemmatize(test_tokens)
    
    assert result == ["run", "jump", "good", "bad"]

def test_preprocess_dataframe():
    """Test DataFrame preprocessing"""
    processor = TextPreprocessor()
    
    df = pd.DataFrame({
        'text': [
            "Hello, World!",
            "https://example.com",
            "123 numbers 456"
        ]
    })
    
    result_df = processor.preprocess_dataframe(df, 'text')
    
    assert len(result_df) == 3
    assert result_df['text'].iloc[0] == "hello world"
    assert result_df['text'].iloc[1] == ""
    assert result_df['text'].iloc[2] == "numbers"
"""
Unit tests for preprocessor module
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessor import DataPreprocessor

class TestDataPreprocessor:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = DataPreprocessor()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            'review_text': ['Great app!', 'Terrible service', 'Good features'],
            'rating': [5, 1, 4],
            'bank': ['CBE', 'BOA', 'DASHEN'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
    
    def test_clean_text(self):
        """Test text cleaning"""
        dirty_text = "This is a GREAT app!!! 😊"
        clean_text = self.preprocessor.clean_text(dirty_text)
        assert isinstance(clean_text, str)
        assert len(clean_text) > 0
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        # Add duplicate row
        duplicate_data = pd.concat([self.sample_data, self.sample_data.iloc[[0]]])
        cleaned_data = self.preprocessor.remove_duplicates(duplicate_data)
        assert len(cleaned_data) == len(self.sample_data)
