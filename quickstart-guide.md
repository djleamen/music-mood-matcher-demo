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

## 🔧 Optional: OpenAI Enhanced Mode

For better mood understanding, add to `.env`:
```bash
OPENAI_API_KEY=your_key_here
```

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
