# 🎵 Quick Demo Guide

## 🚀 Start in 30 seconds

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo
streamlit run demo.py

# 3. Open browser to: http://localhost:8501
```

## 🎭 Try These Example Moods

| Mood Input | Expected Playlist Style |
|------------|------------------------|
| `"I need energizing music for my workout"` | High-energy electronic, rock |
| `"I want something chill for studying"` | Ambient, lo-fi, instrumental |
| `"I'm feeling nostalgic and emotional"` | Ballads, acoustic, indie |
| `"I want to party and dance"` | Pop, dance, upbeat |
| `"I need calming music to sleep"` | Soft, slow, peaceful |

## 🤖 AI Enhancement Features

### Standard Mode (No API Key Required)
- **Basic Mood Detection**: Uses NLTK and keyword matching
- **Algorithmic Recommendations**: Based on audio feature matching
- **Fast Performance**: Instant results

### AI-Enhanced Mode (With OpenAI API Key)
- **🧠 Advanced Mood Analysis**: ChatGPT understands complex emotions
- **🎵 Intelligent Song Curation**: AI analyzes each song individually
- **📝 Detailed Explanations**: Get reasoning for every recommendation
- **⚖️ Song Comparison**: AI ranks multiple songs for specific moods
- **🎯 Contextual Understanding**: Considers your specific mood description

### Enable AI Features
1. Get OpenAI API key from [platform.openai.com](https://platform.openai.com)
2. Create `.env` file:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```
3. Restart the demo to see AI enhancements!

## 📁 Clean Project Structure

```
musicanalyzer/
├── demo.py              # Main app
├── core/                # AI modules
│   ├── mood_analyzer.py
│   ├── music_analyzer.py
│   └── recommendation.py
├── requirements.txt     # Dependencies
└── README.md           # Full documentation
```

## 🎯 Demo Features

- ✅ **Mood Detection**: Natural language → emotion classification
- ✅ **Music Matching**: AI-powered track recommendations  
- ✅ **Analytics**: Visualize patterns and preferences
- ✅ **500+ Demo Tracks**: No setup required
- ✅ **Interactive Interface**: Rate and provide feedback

**🎵 Ready? Run `streamlit run demo.py` and explore!**
