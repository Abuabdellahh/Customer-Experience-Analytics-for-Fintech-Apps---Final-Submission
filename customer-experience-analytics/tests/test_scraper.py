import pytest
import requests
from unittest.mock import patch, MagicMock
from src.scraper import WebScraper

def test_fetch_page_success():
    """Test successful page fetch"""
    url = "https://example.com"
    expected_content = "<html><body>Test content</body></html>"
    
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = expected_content
        mock_get.return_value = mock_response
        
        scraper = WebScraper()
        content = scraper.fetch_page(url)
        
        assert content == expected_content
        mock_get.assert_called_once_with(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=10
        )

def test_fetch_page_retry():
    """Test page fetch with retry logic"""
    url = "https://example.com"
    expected_content = "<html><body>Test content</body></html>"
    
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = expected_content
        
        # First two attempts fail, third succeeds
        mock_get.side_effect = [
            requests.exceptions.RequestException("Timeout"),
            requests.exceptions.RequestException("Timeout"),
            mock_response
        ]
        
        scraper = WebScraper()
        content = scraper.fetch_page(url)
        
        assert content == expected_content
        assert mock_get.call_count == 3

def test_parse_page():
    """Test HTML parsing"""
    html_content = """
    <div class="review">
        <p class="text">Great service!</p>
        <span class="rating">5</span>
    </div>
    """
    
    scraper = WebScraper()
    reviews = scraper.parse_page(html_content)
    
    assert len(reviews) == 1
    assert reviews[0]['text'] == "Great service!"
    assert reviews[0]['rating'] == "5"
