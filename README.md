# 🎵 AI Music Mood Matcher - Interactive Demo

An intelligent music recommendation system that analyzes your mood through natural language and suggests personalized playlists using advanced AI and machine learning techniques.

![Demo Preview](https://img.shields.io/badge/Demo-Interactive-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## ✨ Demo Features

### 🎭 **Intelligent Mood Detection**
- **Natural Language Processing**: Describe your mood in plain English
- **Multiple AI Methods**: 
  - OpenAI GPT integration for advanced understanding
  - NLTK VADER sentiment analysis
  - Custom keyword-based detection
- **Real-time Analysis**: Instant mood classification with confidence scores

### 🎵 **Smart Music Recommendations**
- **AI-Powered Matching**: Advanced algorithms match audio features to your mood
- **Personalized Playlists**: Generates custom playlists based on your emotional state
- **Diverse Selection**: Ensures variety while maintaining mood consistency
- **Interactive Feedback**: Rate recommendations to improve future suggestions

### 📊 **Comprehensive Analytics**
- **Mood Distribution**: Visualize your emotional patterns over time
- **Music Feature Analysis**: Understand the audio characteristics of your preferences
- **Recommendation Insights**: See why certain songs were suggested
- **Interactive Charts**: Explore your data with dynamic visualizations

### 🔬 **Demo Data Experience**
- **Realistic Music Library**: Carefully curated demo tracks across all genres
- **Diverse Moods**: Experience recommendations for happy, sad, energetic, calm, and more
- **No Setup Required**: Try all features immediately without any API keys

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/djleamen/musicanalyzer-demo
cd musicanalyzer-demo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Demo
```bash
# Option 1: Direct launch
streamlit run demo.py

# Option 2: Using start script
chmod +x start_demo.sh
./start_demo.sh
```

### 4. Open in Browser
Navigate to `http://localhost:8501` to access the interactive demo.

## 📋 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 2GB RAM minimum
- **Storage**: 100MB for dependencies
- **Browser**: Any modern web browser
- **Internet**: Required for initial package installation

## 🎮 How to Use the Demo

### 🏠 **Home Page**
- Overview of the system capabilities
- Quick stats about the demo music library
- Navigation to different features

### 🎭 **Mood Matcher**
1. **Enter Your Mood**: Type how you're feeling in natural language
   - Examples: "I need energizing music for my workout"
   - "I want something chill for studying"
   - "I'm feeling nostalgic and want emotional songs"

2. **AI Analysis**: Watch as the system:
   - Analyzes your text for sentiment and keywords
   - Determines your primary mood with confidence scores
   - Maps your emotion to audio features

3. **Get Recommendations**: Receive a personalized playlist with:
   - 20 carefully selected tracks
   - Detailed explanations for each recommendation
   - Audio feature visualizations

4. **Provide Feedback**: Rate songs to improve future recommendations

### 📊 **Analytics Dashboard**
- **Mood Trends**: Track your emotional patterns
- **Music Preferences**: Discover your favorite audio characteristics
- **Recommendation History**: Review past suggestions and feedback
- **Feature Correlations**: Understand relationships between mood and music

## 🛠 Advanced Configuration

### 🤖 **OpenAI Integration** (Optional)
For enhanced mood detection using ChatGPT:

1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com)
2. Copy `.env.example` to `.env`
3. Add your API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

**Benefits of OpenAI Integration:**
- More nuanced mood understanding
- Better handling of complex emotional descriptions
- Contextual reasoning for recommendations
- Improved accuracy for ambiguous inputs

### 🎚 **Customization Options**
Modify demo behavior by editing `demo.py`:
- **Playlist Size**: Change `playlist_size` parameter
- **Mood Categories**: Add new mood types in `mood_categories`
- **Audio Features**: Adjust feature weights in recommendation engine
- **UI Themes**: Customize colors and styling in CSS section

## 🏗 Technical Architecture

### 📁 **Project Structure**
```
musicanalyzer-demo/
├── demo.py                 # Main Streamlit application
├── core/                   # Core AI modules
│   ├── __init__.py
│   ├── mood_analyzer.py    # NLP and mood detection
│   ├── music_analyzer.py   # Audio feature analysis
│   └── recommendation.py   # Recommendation engine
├── requirements.txt        # Python dependencies
├── start_demo.sh          # Launch script
├── .env.example           # Environment template
└── README.md              # This file
```

### 🧠 **AI Components**

#### **Mood Analyzer**
- **NLTK VADER**: Sentiment polarity analysis
- **TextBlob**: Additional sentiment processing
- **OpenAI GPT**: Advanced natural language understanding
- **Custom Keywords**: Mood-specific term detection

#### **Music Analyzer**
- **Audio Features**: Analyzes danceability, energy, valence, tempo, etc.
- **Clustering**: Groups similar tracks using K-means
- **Feature Scaling**: Normalizes audio characteristics
- **Correlation Analysis**: Finds patterns in music preferences

#### **Recommendation Engine**
- **Cosine Similarity**: Matches tracks to mood profiles
- **Weighted Scoring**: Balances multiple recommendation factors
- **Diversity Injection**: Ensures playlist variety
- **Feedback Learning**: Adapts to user preferences over time

### 🔬 **Machine Learning Pipeline**
1. **Text Processing**: Clean and tokenize mood descriptions
2. **Feature Extraction**: Convert text to numerical representations
3. **Mood Classification**: Predict emotional state from input
4. **Audio Mapping**: Match moods to musical characteristics
5. **Recommendation Scoring**: Rank tracks by relevance
6. **Diversity Optimization**: Select varied but relevant tracks

## 🎯 Demo Scenarios

Try these example inputs to explore different features:

### 🏃‍♂️ **Workout Energy**
```
"I need high-energy music for an intense workout session"
```
**Expected Output**: High-energy electronic, rock, and hip-hop tracks

### 📚 **Study Focus**
```
"I want calm, instrumental music for deep concentration"
```
**Expected Output**: Ambient, classical, and lo-fi tracks

### 💔 **Emotional Processing**
```
"I'm going through a breakup and need songs that understand my pain"
```
**Expected Output**: Emotional ballads and introspective tracks

### 🎉 **Party Vibes**
```
"I'm hosting a party and need upbeat danceable music"
```
**Expected Output**: Pop, dance, and party classics

### 🌙 **Late Night Chill**
```
"It's 2 AM and I want something dreamy and atmospheric"
```
**Expected Output**: Chillout, downtempo, and atmospheric tracks

## 🐛 Troubleshooting

### **Common Issues**

#### **Import Errors**
```bash
# If you see module import errors
pip install --upgrade -r requirements.txt
```

#### **NLTK Data Missing**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

#### **Port Already in Use**
```bash
# Use a different port
streamlit run demo.py --server.port 8502
```

#### **OpenAI API Issues**
- Verify your API key is correct
- Check your OpenAI account has credits
- Demo works without OpenAI if key is invalid

### **Performance Tips**
- **Large Playlists**: Reduce playlist size for faster generation
- **Slow Loading**: Restart the demo if it becomes unresponsive
- **Memory Usage**: Close other applications if system is slow

## 🔮 Future Enhancements

### **Potential Features**
- **Real Spotify Integration**: Connect to your actual music library
- **Voice Input**: Speak your mood instead of typing
- **Multiple Languages**: Support for non-English mood descriptions
- **Social Features**: Share playlists and mood insights
- **Mobile App**: Native iOS/Android applications
- **Smart Scheduling**: Automatic mood detection based on time/calendar

### **Technical Improvements**
- **Deep Learning**: Neural networks for better mood understanding
- **Real-time Audio**: Analyze currently playing music
- **Collaborative Filtering**: Learn from community preferences
- **Cross-platform**: Desktop and mobile applications

## 📚 Learning Resources

### **AI/ML Concepts Used**
- **Natural Language Processing**: Text analysis and sentiment detection
- **Machine Learning**: Classification and recommendation algorithms
- **Data Science**: Feature engineering and statistical analysis
- **Web Development**: Interactive application design

### **Technologies Explored**
- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **Plotly**: Interactive data visualization
- **OpenAI**: Advanced AI integration

## 🤝 Contributing

This is a demo project, but you can extend it:

1. **Fork the repository**
2. **Add new mood categories**
3. **Improve recommendation algorithms**
4. **Enhance the user interface**
5. **Add new visualization types**

## 📄 License

This demo is provided for educational and demonstration purposes. 

## 🙏 Acknowledgments

- **OpenAI**: GPT integration for advanced NLP
- **Spotify**: Audio feature definitions and inspiration
- **Streamlit**: Excellent web framework for ML demos
- **NLTK**: Robust natural language processing tools
- **Scikit-learn**: Comprehensive machine learning library

---

**🎵 Ready to explore your musical emotions? Launch the demo and discover how AI can understand your mood through music!**

```bash
streamlit run demo.py
```

