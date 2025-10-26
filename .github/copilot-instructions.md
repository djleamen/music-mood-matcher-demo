# GitHub Copilot Instructions for AI Music Mood Matcher

## Project Overview

This is an AI-powered music recommendation system that analyzes user moods through natural language processing and suggests personalized playlists. The project uses Streamlit for the web interface, multiple AI/ML libraries for mood analysis, and can optionally integrate with OpenAI's API for enhanced functionality.

## Technology Stack

- **Python**: 3.8+
- **Web Framework**: Streamlit
- **NLP/ML**: NLTK, TextBlob, scikit-learn, OpenAI API (optional)
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Code Quality**: pylint with specific disabled rules (see workflows)

## Project Structure

```
music-mood-matcher-demo/
├── demo.py                 # Main Streamlit application entry point
├── demo_data.py           # Demo track data and fixtures
├── core/                  # Core AI/ML modules
│   ├── mood_analyzer.py   # NLP and sentiment analysis
│   ├── music_analyzer.py  # Audio feature analysis and ML
│   ├── recommendation.py  # Recommendation engine
│   └── utils.py          # Shared utilities
├── test_app.py           # Manual testing script
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variable template
└── start_demo.sh        # Launch script
```

## Coding Standards

### Python Style

1. **Follow PEP 8** with the following pylint exceptions (see `.github/workflows/pylint.yml`):
   - C0301: Line too long
   - C0114: Missing module docstring
   - C0115: Missing class docstring
   - C0116: Missing function docstring
   - C0303: Trailing whitespace
   - E0401: Import error
   - R0902: Too many instance attributes
   - R0903: Too few public methods
   - R0911: Too many return statements
   - R0912: Too many branches
   - R0913: Too many arguments
   - R0914: Too many local variables
   - R0915: Too many statements
   - R0917: Too many positional arguments
   - R1702: Too many nested blocks
   - R1716: Simplifiable if statement

2. **Type Hints**: Use type hints for function parameters and return values where possible:
   ```python
   from typing import Dict, List, Tuple, Optional
   
   def analyze_mood(self, text: str) -> Dict[str, float]:
       """Analyze mood from text input"""
       pass
   ```

3. **Docstrings**: Include docstrings for classes and complex functions:
   ```python
   def analyze_music_data(self, df: pd.DataFrame) -> Dict:
       """Analyze music data and extract insights"""
       pass
   ```

### Code Organization

1. **Imports**: Group imports in this order:
   - Standard library imports
   - Third-party imports
   - Local imports
   
   ```python
   import os
   import sys
   
   import pandas as pd
   import streamlit as st
   
   from core.mood_analyzer import MoodAnalyzer
   ```

2. **Class Structure**: Keep classes focused on single responsibilities:
   - `MoodAnalyzer`: Handles text analysis and mood detection
   - `MusicAnalyzer`: Manages audio features and ML models
   - `RecommendationEngine`: Generates music recommendations

3. **Error Handling**: Use try-except blocks with informative fallbacks:
   ```python
   try:
       nltk.data.find('vader_lexicon')
       NLTK_AVAILABLE = True
   except (LookupError, OSError):
       print("⚠️ NLTK data not available. Using fallback sentiment analysis.")
       NLTK_AVAILABLE = False
   ```

## Key Patterns and Practices

### 1. OpenAI Integration

Always check for API key availability and provide fallback functionality:

```python
from .utils import initialize_openai_client

client = initialize_openai_client()
if client:
    # Use OpenAI API for enhanced features
    response = client.chat.completions.create(...)
else:
    # Fall back to traditional ML methods
    result = fallback_method()
```

### 2. Streamlit UI Components

Use consistent styling with custom CSS and session state for data persistence:

```python
# Initialize session state
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = {}

# Use columns for layout
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.write(f"**{track['name']}** by {track['artist']}")
```

### 3. Mood Analysis Pipeline

The mood analysis follows this pattern:
1. Text preprocessing and cleaning
2. Sentiment analysis (VADER/TextBlob + optional OpenAI)
3. Keyword extraction and mood classification
4. Audio feature mapping

### 4. Music Recommendation

Recommendations use a multi-factor approach:
1. Cosine similarity between mood profile and track features
2. Optional AI-powered song evaluation
3. Diversity injection to prevent repetition
4. User feedback integration

### 5. Data Processing

Audio features should follow these conventions:
- **Normalized features**: danceability, energy, valence, acousticness (0.0-1.0)
- **Tempo**: BPM (typically 60-200)
- **Loudness**: Decibels (-60 to 0)
- **Key**: Integer (0-11)

## Testing Approach

### Manual Testing

The project uses manual testing via `test_app.py` which:
- Checks if Streamlit app is accessible
- Provides a testing checklist
- Validates feedback functionality

Run tests with:
```bash
python test_app.py
```

### Testing New Features

When adding new features:
1. Test with demo data first (200+ tracks in `demo_data.py`)
2. Test with and without OpenAI API key
3. Verify Streamlit UI updates correctly
4. Check error handling for edge cases
5. Ensure backwards compatibility

## Environment and Dependencies

### Required Dependencies

Core dependencies in `requirements.txt`:
- `streamlit>=1.25.0`: Web interface
- `pandas>=1.5.0`, `numpy>=1.21.0`: Data processing
- `scikit-learn>=1.2.0`: Machine learning
- `nltk>=3.7`, `textblob>=0.17.0`: NLP
- `plotly>=5.10.0`: Interactive visualizations
- `openai>=1.0.0`: Optional AI enhancement

### Environment Variables

Use `.env` file for configuration (see `.env.example`):
```bash
OPENAI_API_KEY=your_api_key_here  # Optional but recommended
```

Load with `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
```

### NLTK Data

Download required NLTK data on first run:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

## Common Development Tasks

### Running the Application

```bash
# Standard launch
streamlit run demo.py

# Custom port
streamlit run demo.py --server.port 8502

# Using the start script
./start_demo.sh
```

### Linting

```bash
# Run pylint with project-specific rules
pylint --disable=C0301,C0114,C0115,C0116,C0303,E0401,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R0917,R1702,R1716 $(git ls-files '*.py')
```

### Adding New Mood Categories

1. Update mood keywords in `mood_analyzer.py`
2. Define audio feature profiles for the mood
3. Test with various input phrases
4. Add example to documentation

### Adding New Audio Features

1. Add feature to `audio_features` list in `MusicAnalyzer`
2. Update `feature_descriptions` dictionary
3. Update demo data in `demo_data.py` if needed
4. Adjust normalization/scaling as appropriate

## Best Practices for AI Features

### When Using OpenAI API

1. **Always provide context**: Include user's mood description in prompts
2. **Use structured output**: Request JSON format for easier parsing
3. **Handle rate limits**: Implement retry logic with exponential backoff
4. **Fallback gracefully**: Ensure app works without API key
5. **Cache responses**: Avoid redundant API calls for same inputs

### Prompt Engineering

Use clear, specific prompts:
```python
prompt = f"""Analyze if this song matches the mood: {mood_description}

Song: {track_name} by {artist}
Audio Features: {features}

Respond with a JSON object containing:
- match_score (0-10)
- reasoning (brief explanation)
- mood_keywords (list of relevant mood words)
"""
```

## UI/UX Guidelines

### Streamlit Components

1. **Use emojis** for visual appeal and better UX
2. **Provide progress indicators** for long-running operations
3. **Show confidence scores** for AI decisions
4. **Enable user feedback** on recommendations
5. **Use expanders** for detailed information

### Visualization

- Use Plotly for interactive charts
- Maintain consistent color schemes
- Provide tooltips and labels
- Support responsive layouts

## Security Considerations

1. **Never commit API keys** to the repository
2. **Use environment variables** for sensitive data
3. **Validate user inputs** before processing
4. **Handle API errors gracefully** without exposing internal details
5. **Sanitize data** before displaying in UI

## Performance Tips

1. **Cache expensive operations** with Streamlit's `@st.cache_data`
2. **Lazy load models** only when needed
3. **Batch API requests** when processing multiple items
4. **Use efficient data structures** (numpy arrays over lists)
5. **Limit playlist size** for faster generation

## Common Pitfalls to Avoid

1. ❌ Don't assume NLTK data is always available
2. ❌ Don't make API calls without checking for key
3. ❌ Don't modify session state directly without initialization checks
4. ❌ Don't use hardcoded file paths (use `os.path` or `pathlib`)
5. ❌ Don't ignore edge cases (empty inputs, missing features, etc.)

## Contributing Guidelines

When making changes:
1. Test locally with `streamlit run demo.py`
2. Verify pylint passes with project rules
3. Update relevant documentation
4. Ensure OpenAI-optional features still work without API key
5. Test with demo data
6. Consider performance impact on user experience
