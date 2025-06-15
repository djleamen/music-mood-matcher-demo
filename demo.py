import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import after path setup to avoid import issues  # noqa: E402
from core.mood_analyzer import MoodAnalyzer  # noqa: E402
from core.music_analyzer import MusicAnalyzer  # noqa: E402
from core.recommendation import RecommendationEngine  # noqa: E402


# Page configuration
st.set_page_config(
    page_title="AI Music Mood Matcher - Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mood-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .track-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1DB954;
        margin: 0.5rem 0;
        color: #000000;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        color: white;
        border: none;
        border-radius: 5px;
    }
    .demo-banner {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def create_demo_data():
    """Create demo music data for demonstration"""
    np.random.seed(42)

    # Demo tracks with realistic audio features
    tracks_data = [
        # Happy/Energetic tracks
        {"name": "Feel Good Inc.", "artist": "Gorillaz", "album": "Demon Days", "danceability": 0.8,
            "energy": 0.9, "valence": 0.8, "tempo": 138, "acousticness": 0.1, "mood_category": "happy"},
        {"name": "Happy", "artist": "Pharrell Williams", "album": "GIRL", "danceability": 0.9,
            "energy": 0.8, "valence": 0.95, "tempo": 160, "acousticness": 0.2, "mood_category": "happy"},
        {"name": "Good as Hell", "artist": "Lizzo", "album": "Cuz I Love You", "danceability": 0.85,
            "energy": 0.9, "valence": 0.9, "tempo": 140, "acousticness": 0.15, "mood_category": "happy"},

        # Sad/Melancholic tracks
        {"name": "Someone Like You", "artist": "Adele", "album": "21", "danceability": 0.3,
            "energy": 0.2, "valence": 0.2, "tempo": 67, "acousticness": 0.8, "mood_category": "sad"},
        {"name": "Mad World", "artist": "Gary Jules", "album": "Donnie Darko Soundtrack", "danceability": 0.2,
            "energy": 0.1, "valence": 0.1, "tempo": 62, "acousticness": 0.9, "mood_category": "sad"},
        {"name": "Black", "artist": "Pearl Jam", "album": "Ten", "danceability": 0.3, "energy": 0.4,
            "valence": 0.2, "tempo": 75, "acousticness": 0.7, "mood_category": "sad"},

        # Calm/Chill tracks
        {"name": "River", "artist": "Joni Mitchell", "album": "Blue", "danceability": 0.4,
            "energy": 0.3, "valence": 0.5, "tempo": 90, "acousticness": 0.9, "mood_category": "calm"},
        {"name": "Holocene", "artist": "Bon Iver", "album": "Bon Iver, Bon Iver", "danceability": 0.3,
            "energy": 0.2, "valence": 0.4, "tempo": 68, "acousticness": 0.8, "mood_category": "calm"},
        {"name": "Aqueous Transmission", "artist": "Incubus", "album": "Morning View", "danceability": 0.2,
            "energy": 0.3, "valence": 0.5, "tempo": 80, "acousticness": 0.6, "mood_category": "calm"},

        # Energetic/Workout tracks
        {"name": "Till I Collapse", "artist": "Eminem", "album": "The Eminem Show", "danceability": 0.7,
            "energy": 0.95, "valence": 0.6, "tempo": 85, "acousticness": 0.05, "mood_category": "energetic"},
        {"name": "Pump It", "artist": "Black Eyed Peas", "album": "Monkey Business", "danceability": 0.9,
            "energy": 0.9, "valence": 0.8, "tempo": 124, "acousticness": 0.1, "mood_category": "energetic"},
        {"name": "Stronger", "artist": "Kanye West", "album": "Graduation", "danceability": 0.8,
            "energy": 0.85, "valence": 0.7, "tempo": 104, "acousticness": 0.1, "mood_category": "energetic"},

        # Angry/Intense tracks
        {"name": "Break Stuff", "artist": "Limp Bizkit", "album": "Significant Other", "danceability": 0.6,
            "energy": 0.95, "valence": 0.2, "tempo": 95, "acousticness": 0.02, "mood_category": "energetic"},
        {"name": "Bodies", "artist": "Drowning Pool", "album": "Sinner", "danceability": 0.5, "energy": 0.98,
            "valence": 0.15, "tempo": 152, "acousticness": 0.01, "mood_category": "energetic"},
        {"name": "Killing in the Name",
         "artist": "Rage Against the Machine",
         "album": "Rage Against the Machine",
         "danceability": 0.65,
         "energy": 0.97,
         "valence": 0.25,
         "tempo": 85,
         "acousticness": 0.02,
         "mood_category": "energetic"},

        # Romantic tracks
        {"name": "At Last", "artist": "Etta James", "album": "At Last!", "danceability": 0.5,
            "energy": 0.4, "valence": 0.7, "tempo": 75, "acousticness": 0.4, "mood_category": "romantic"},
        {"name": "Perfect", "artist": "Ed Sheeran", "album": "÷", "danceability": 0.6, "energy": 0.4,
            "valence": 0.8, "tempo": 95, "acousticness": 0.6, "mood_category": "romantic"},
        {"name": "Make You Feel My Love", "artist": "Adele", "album": "19", "danceability": 0.3,
            "energy": 0.3, "valence": 0.6, "tempo": 72, "acousticness": 0.8, "mood_category": "romantic"},
    ]

    # Add more audio features
    for track in tracks_data:
        track.update({
            "speechiness": np.random.uniform(0.03, 0.2),
            "instrumentalness": np.random.uniform(0.0, 0.1),
            "liveness": np.random.uniform(0.05, 0.3),
            "loudness": np.random.uniform(-15, -5),
            "popularity": np.random.randint(40, 95),
            "id": f"demo_{len(tracks_data)}_{track['name'].replace(' ', '_')}",
            "playlist_name": "Demo Playlist"
        })

    return pd.DataFrame(tracks_data)


def initialize_demo_state():
    """Initialize demo session state"""
    if 'demo_tracks' not in st.session_state:
        st.session_state.demo_tracks = create_demo_data()

    if 'mood_analyzer' not in st.session_state:
        st.session_state.mood_analyzer = MoodAnalyzer()

    if 'music_analyzer' not in st.session_state:
        st.session_state.music_analyzer = MusicAnalyzer()

    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()
        # Learn from demo data
        st.session_state.recommendation_engine.learn_user_preferences(st.session_state.demo_tracks)


def demo_home():
    """Demo home page"""
    st.markdown('<h1 class="main-header">🎵 AI Music Mood Matcher</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="demo-banner">
        <h3>🎭 Interactive Demo</h3>
        <p>Experience the AI Music Mood Matcher with sample data! This demo shows how the system analyzes mood and recommends music.</p>
        <p><strong>Note:</strong> This demo uses sample tracks. For full functionality with your Spotify playlists, set up the Spotify API credentials.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Demo Tracks", len(st.session_state.demo_tracks))

    with col2:
        unique_artists = st.session_state.demo_tracks['artist'].nunique()
        st.metric("Artists", unique_artists)

    with col3:
        moods = st.session_state.demo_tracks['mood_category'].nunique()
        st.metric("Mood Categories", moods)

    with col4:
        st.metric("AI Models", "Ready!")

    # Show sample tracks
    st.subheader("🎵 Sample Music Library")

    # Audio features visualization
    if st.checkbox("Show Audio Features Analysis"):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Valence vs Energy', 'Danceability Distribution', 'Tempo Distribution', 'Mood Categories'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )

        # Valence vs Energy colored by mood
        for mood in st.session_state.demo_tracks['mood_category'].unique():
            mood_data = st.session_state.demo_tracks[st.session_state.demo_tracks['mood_category'] == mood]
            fig.add_trace(
                go.Scatter(x=mood_data['valence'], y=mood_data['energy'],
                           mode='markers', name=mood.title(), showlegend=True),
                row=1, col=1
            )

        # Danceability distribution
        fig.add_trace(
            go.Histogram(x=st.session_state.demo_tracks['danceability'], name="Danceability", showlegend=False),
            row=1, col=2
        )

        # Tempo distribution
        fig.add_trace(
            go.Histogram(x=st.session_state.demo_tracks['tempo'], name="Tempo", showlegend=False),
            row=2, col=1
        )

        # Mood categories
        mood_counts = st.session_state.demo_tracks['mood_category'].value_counts()
        fig.add_trace(
            go.Bar(x=mood_counts.index, y=mood_counts.values, name="Moods", showlegend=False),
            row=2, col=2
        )

        fig.update_layout(height=600, title="Music Library Analysis")
        st.plotly_chart(fig, use_container_width=True)

    # Sample tracks table
    display_columns = ['name', 'artist', 'mood_category', 'valence', 'energy', 'danceability']
    st.dataframe(st.session_state.demo_tracks[display_columns], use_container_width=True)


def demo_mood_matcher():
    """Demo mood matching interface"""
    st.title("🎭 Mood Matcher Demo")
    st.markdown("Try the AI mood detection and music recommendation system!")

    # Mood input section
    st.subheader("Express Your Mood")

    # Predefined mood examples
    mood_examples = [
        "I'm feeling super energetic and want to work out!",
        "I'm sad and need some comfort music",
        "I want something chill and relaxing for studying",
        "I'm in a romantic mood and want love songs",
        "I'm angry and need some intense music",
    ]

    col1, col2 = st.columns([2, 1])

    # Initialize selected mood if not exists
    if 'selected_mood' not in st.session_state:
        st.session_state.selected_mood = ""

    with col1:
        mood_input = st.text_area(
            "Describe how you're feeling:",
            value=st.session_state.selected_mood,
            placeholder="Type your mood here or select an example...",
            height=100,
            key="mood_input_area"
        )

    with col2:
        st.write("**Quick Examples:**")
        for i, example in enumerate(mood_examples):
            if st.button(f"📝 {example[:30]}...", key=f"example_{i}"):
                st.session_state.selected_mood = example
                st.rerun()

    # Recommendation settings
    col1, col2 = st.columns(2)
    with col1:
        playlist_size = st.slider("Playlist Size", 3, 10, 5)

    if st.button("🎵 Analyze Mood & Generate Playlist", type="primary"):
        # Update session state with current input
        if mood_input.strip():
            st.session_state.selected_mood = mood_input
        if mood_input.strip():
            with st.spinner("Analyzing your mood..."):
                # Analyze mood
                mood_analysis = st.session_state.mood_analyzer.analyze_mood(mood_input)

                # Display mood analysis
                st.subheader(f"🎭 Detected Mood: {mood_analysis['primary_mood'].title()}")

                # Show which method was used
                method = mood_analysis.get('method', 'fallback')
                if method == 'openai':
                    st.success("🤖 Powered by ChatGPT for enhanced accuracy!")
                else:
                    st.info("🔧 Using fallback analysis (Set OPENAI_API_KEY for better results)")

                # Show reasoning if available
                if 'reasoning' in mood_analysis:
                    st.write(f"**AI Reasoning:** {mood_analysis['reasoning']}")

                with st.expander("🔍 Detailed Mood Analysis"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Mood Confidence Scores:**")
                        mood_scores = mood_analysis['mood_scores']
                        # Normalize scores to 0-1 range for progress bars
                        max_score = max(mood_scores.values()) if mood_scores.values() else 1.0
                        for mood, score in sorted(mood_scores.items(), key=lambda x: x[1], reverse=True):
                            if score > 0:
                                normalized_score = min(score / max_score, 1.0) if max_score > 0 else 0
                                st.progress(normalized_score, text=f"{mood.title()}: {score:.2f}")

                    with col2:
                        st.write("**Sentiment Analysis:**")
                        sentiment = mood_analysis['sentiment_scores']
                        st.metric("Positivity", f"{sentiment['vader_positive']:.2f}")
                        st.metric("Negativity", f"{sentiment['vader_negative']:.2f}")
                        st.metric("Overall Sentiment", f"{sentiment['vader_compound']:.2f}")

                # Generate recommendations based on detected mood
                primary_mood = mood_analysis['primary_mood']

                # Simple rule-based recommendation for demo
                audio_preferences = mood_analysis.get('audio_preferences', {})

                if audio_preferences:
                    # Score tracks based on mood preferences
                    scored_tracks = st.session_state.demo_tracks.copy()
                    scores = []

                    for _, track in scored_tracks.iterrows():
                        score = 0
                        count = 0

                        for feature, preference in audio_preferences.items():
                            if feature in track and feature != 'description':
                                feature_value = track[feature]

                                # Handle different preference formats
                                if isinstance(preference, tuple) and len(preference) == 2:
                                    min_val, max_val = preference
                                    if min_val <= feature_value <= max_val:
                                        score += 1.0
                                    else:
                                        # Distance penalty
                                        if feature_value < min_val:
                                            distance = (min_val - feature_value) / max(min_val, 0.1)
                                        else:
                                            distance = (feature_value - max_val) / max(max_val, 0.1)
                                        score += max(0, 1.0 - distance)
                                    count += 1

                        scores.append(score / count if count > 0 else 0.5)

                    scored_tracks['mood_match_score'] = scores

                    # Get top recommendations
                    recommendations = scored_tracks.nlargest(playlist_size, 'mood_match_score')
                else:
                    # Fallback to mood category matching
                    mood_mapping = {
                        'happy': 'happy',
                        'energetic': 'energetic',
                        'sad': 'sad',
                        'calm': 'calm',
                        'romantic': 'romantic',
                        'angry': 'energetic'  # Map angry to energetic for now
                    }

                    target_category = mood_mapping.get(primary_mood, 'calm')
                    category_tracks = st.session_state.demo_tracks[
                        st.session_state.demo_tracks['mood_category'] == target_category
                    ]

                    if len(category_tracks) >= playlist_size:
                        recommendations = category_tracks.sample(playlist_size)
                    else:
                        # Fill with similar moods
                        recommendations = st.session_state.demo_tracks.sample(
                            min(playlist_size, len(st.session_state.demo_tracks)))

                    recommendations['mood_match_score'] = 0.8

                # Display recommendations
                st.subheader("🎵 Your Personalized Playlist")
                st.markdown(f"*Based on your mood: {mood_analysis['primary_mood']}*")

                for i, (_, track) in enumerate(recommendations.iterrows()):
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.markdown(f"""
                        <div class="track-card">
                            <strong>🎵 {track['name']}</strong><br>
                            <em>by {track['artist']}</em><br>
                            <small>from {track['album']}</small>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        mood_score = track.get('mood_match_score', 0)
                        st.metric("Match", f"{mood_score:.1f}")

                    with col3:
                        # Demo feedback
                        feedback_col1, feedback_col2 = st.columns(2)
                        with feedback_col1:
                            if st.button("👍", key=f"like_{i}"):
                                st.success("👍 Liked!")
                        with feedback_col2:
                            if st.button("👎", key=f"dislike_{i}"):
                                st.info("👎 Noted!")

                # Audio features explanation
                with st.expander("🎯 Why These Songs?"):
                    if audio_preferences:
                        st.write(f"**Audio profile for {primary_mood} mood:**")
                        for feature, preference in audio_preferences.items():
                            if feature != 'description' and isinstance(preference, tuple) and len(preference) == 2:
                                min_val, max_val = preference
                                st.write(f"- **{feature.title()}**: {min_val:.2f} - {max_val:.2f}")

                        # Show how recommendations match
                        feature_cols = ['valence', 'energy', 'danceability']
                        avg_features = recommendations[feature_cols].mean()

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=avg_features.values,
                            theta=avg_features.index,
                            fill='toself',
                            name='Recommended Songs',
                            line_color='#1DB954'
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1])
                            ),
                            title="Audio Features Profile of Recommendations"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please describe your mood first!")


def demo_analytics():
    """Demo analytics page"""
    st.title("📊 Music Analytics Demo")

    # Overall library stats
    st.subheader("📈 Library Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tracks", len(st.session_state.demo_tracks))
    with col2:
        st.metric("Unique Artists", st.session_state.demo_tracks['artist'].nunique())
    with col3:
        avg_valence = st.session_state.demo_tracks['valence'].mean()
        st.metric("Avg Happiness", f"{avg_valence:.2f}")
    with col4:
        avg_energy = st.session_state.demo_tracks['energy'].mean()
        st.metric("Avg Energy", f"{avg_energy:.2f}")

    # Mood distribution
    st.subheader("🎭 Mood Distribution")
    mood_counts = st.session_state.demo_tracks['mood_category'].value_counts()
    fig = px.pie(values=mood_counts.values, names=mood_counts.index,
                 title="Distribution of Moods in Library")
    st.plotly_chart(fig, use_container_width=True)

    # Audio features analysis
    st.subheader("🎵 Audio Features Analysis")

    # Radar chart for average features by mood
    audio_features = ['danceability', 'energy', 'valence', 'acousticness']
    mood_profiles = st.session_state.demo_tracks.groupby('mood_category')[audio_features].mean()

    fig = go.Figure()

    for mood in mood_profiles.index:
        fig.add_trace(go.Scatterpolar(
            r=mood_profiles.loc[mood].values,
            theta=audio_features,
            fill='toself',
            name=mood.title()
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Audio Features by Mood Category"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Valence vs Energy scatter
    st.subheader("😊 Mood Mapping: Valence vs Energy")
    fig = px.scatter(st.session_state.demo_tracks,
                     x='valence', y='energy',
                     color='mood_category',
                     hover_data=['name', 'artist'],
                     title="Songs in Valence-Energy Space")
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main demo application"""
    initialize_demo_state()

    # Sidebar navigation
    st.sidebar.title("🎵 Demo Navigation")

    pages = {
        "🏠 Demo Home": "home",
        "🎭 Mood Matcher": "mood",
        "📊 Analytics": "analytics",
        "ℹ️ About": "about"
    }

    selected = st.sidebar.radio("Navigate to:", list(pages.keys()))
    current_page = pages[selected]

    # Page routing
    if current_page == "home":
        demo_home()
    elif current_page == "mood":
        demo_mood_matcher()
    elif current_page == "analytics":
        demo_analytics()
    elif current_page == "about":
        st.title("ℹ️ About AI Music Mood Matcher Demo")
        st.markdown("""
        ## 🎵 What is this Demo?

        This is an **interactive demonstration** of an AI Music Mood Matcher system that:

        - **Analyzes** your mood using natural language processing
        - **Matches** your emotions to audio features of songs
        - **Recommends** curated playlists based on how you're feeling
        - **Visualizes** music patterns and mood relationships

        ## 🎭 Demo Features

        ### **Mood Detection Methods:**
        - **OpenAI GPT Integration** (if API key provided): Advanced natural language understanding
        - **NLTK VADER Sentiment Analysis**: Rule-based sentiment scoring
        - **Keyword Matching**: Pattern recognition for mood classification
        - **Fallback Analysis**: Ensures the demo always works

        ### **Music Analysis:**
        - **18 Curated Demo Tracks** across 5 mood categories (happy, sad, calm, energetic, romantic)
        - **Audio Feature Analysis**: Valence, energy, danceability, tempo, acousticness, and more
        - **Smart Matching Algorithm**: Cosine similarity between mood profiles and song features
        - **Interactive Visualizations**: Radar charts, scatter plots, and distribution analysis

        ## 🧠 How This Demo Works

        1. **Text Input**: You describe your mood in natural language
        2. **Mood Analysis**: AI processes your text to identify emotional state and audio preferences
        3. **Feature Mapping**: Your mood is converted to target audio feature ranges
        4. **Song Scoring**: Each demo track is scored based on how well it matches your mood
        5. **Playlist Generation**: Top-scoring tracks are selected with diversity optimization
        6. **Visualization**: Results are displayed with explanations and charts

        ## 🎯 Demo Limitations

        **This is a proof-of-concept demo with:**
        - **Limited Music Library**: Only 18 sample tracks (not connected to Spotify)
        - **Simulated Recommendations**: Uses pre-defined audio features, not real Spotify data
        - **No User Learning**: Doesn't save preferences between sessions
        - **No Real Playback**: Visual recommendations only, no actual music streaming

        ## 🔬 Try These Demo Features

        - **Mood Matcher**: Test different mood descriptions and see how the AI interprets them
        - **Analytics**: Explore the relationships between audio features and mood categories
        - **Interactive Charts**: Hover over data points to see detailed track information
        - **OpenAI Mode**: Add your OpenAI API key for enhanced mood understanding
        """)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎵 AI Music Mood Matcher Demo**")
    st.sidebar.markdown("Built by @djleamen using Streamlit, Spotify API, OpenAI, and ML techniques")


if __name__ == "__main__":
    main()
