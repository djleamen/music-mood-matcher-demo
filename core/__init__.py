"""
AI-Driven Music Mood Matcher

Core modules for the music mood matching system.
"""

__version__ = "1.0.0"
__author__ = "AI Music Mood Matcher"

# Import main classes for easy access
from .mood_analyzer import MoodAnalyzer
from .music_analyzer import MusicAnalyzer
from .recommendation import RecommendationEngine
from .utils import initialize_openai_client

__all__ = [
    'MoodAnalyzer',
    'MusicAnalyzer',
    'RecommendationEngine',
    'initialize_openai_client'
]
