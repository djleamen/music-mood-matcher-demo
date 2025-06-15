import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import uuid

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import after path setup to avoid import issues  # noqa: E402
from core.mood_analyzer import MoodAnalyzer  # noqa: E402
from core.music_analyzer import MusicAnalyzer  # noqa: E402
from core.recommendation import RecommendationEngine  # noqa: E402
from demo_data import TRACKS_DATA  # noqa: E402


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

    # Get demo tracks from external file
    tracks_data = TRACKS_DATA.copy()

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
        # Try to load saved feedback data from data directory
        feedback_file = 'data/user_feedback_data.pkl'
        if st.session_state.recommendation_engine.load_recommendation_data(feedback_file):
            st.toast("🧠 Loaded your previous feedback and preferences!")
    
    # Initialize a session counter for unique keys
    if 'session_counter' not in st.session_state:
        st.session_state.session_counter = 0

def save_feedback_data():
    """Save feedback data to persist learning"""
    if 'recommendation_engine' in st.session_state:
        feedback_file = 'data/user_feedback_data.pkl'
        st.session_state.recommendation_engine.save_recommendation_data(feedback_file)
        # Learn from demo data
        st.session_state.recommendation_engine.learn_user_preferences(st.session_state.demo_tracks)


def demo_home():
    """Demo home page"""
    st.markdown('<h1 class="main-header">🎵 AI Music Mood Matcher</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="demo-banner">
        <h3>🎭 Interactive Demo</h3>
        <p>Experience the AI Music Mood Matcher with sample data! This demo shows how the system analyzes mood and
        recommends music.</p>
        <p><strong>Note:</strong> This demo uses sample tracks. For full functionality with your Spotify playlists,
        set up the Spotify API credentials.</p>
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
        st.plotly_chart(fig, use_container_width=True, key="library_analysis_chart")

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
    with col2:
        # Check if OpenAI is available
        has_openai = (hasattr(st.session_state.recommendation_engine, 'openai_client') and
                     st.session_state.recommendation_engine.openai_client is not None)

        if has_openai:
            use_ai = st.checkbox("🚀 Use AI Enhancement", value=True,
                               help="Use ChatGPT to analyze and select the best songs for your mood")
        else:
            st.info("💡 Set OPENAI_API_KEY for AI enhancements")
            use_ai = False

    if st.button("🎵 Analyze Mood & Generate Playlist", type="primary"):
        # Update session state with current input
        if mood_input.strip():
            st.session_state.selected_mood = mood_input
        if mood_input.strip():
            with st.spinner("Analyzing your mood..."):
                # Analyze mood
                mood_analysis = st.session_state.mood_analyzer.analyze_mood(mood_input)
                
                # Store mood analysis and settings in session state
                st.session_state.current_mood_analysis = mood_analysis
                st.session_state.current_playlist_size = playlist_size
                st.session_state.current_use_ai = use_ai
                st.session_state.current_mood_input = mood_input
                
                # Increment session counter for unique keys
                st.session_state.session_counter += 1

                # Generate recommendations using AI-enhanced system
                with st.spinner("🤖 Generating AI-curated playlist..." if use_ai else "📊 Generating playlist..."):
                    # Use the AI-enhanced recommendation engine
                    recommendations = st.session_state.recommendation_engine.generate_mood_playlist(
                        mood_analysis,
                        st.session_state.demo_tracks,
                        playlist_size=playlist_size,
                        use_ai_enhancement=use_ai
                    )
                    
                # Store the recommendations in session state
                st.session_state.current_recommendations = recommendations
        else:
            st.warning("Please describe your mood first!")

    # Display current playlist if it exists (either just generated or from previous session)
    if 'current_recommendations' in st.session_state and st.session_state.current_recommendations is not None:
        mood_analysis = st.session_state.current_mood_analysis
        recommendations = st.session_state.current_recommendations
        use_ai = st.session_state.current_use_ai
        mood_input = st.session_state.current_mood_input

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

        # Check if AI recommendations were used
        if use_ai and hasattr(st.session_state.recommendation_engine, 'openai_client') and st.session_state.recommendation_engine.openai_client:
            st.success("🚀 Enhanced with ChatGPT song analysis!")
        else:
            st.info("📊 Using standard algorithmic recommendations")

        # Get variables for display
        primary_mood = mood_analysis['primary_mood']
        audio_preferences = mood_analysis.get('audio_preferences', {})

        # Display recommendations
        st.subheader("🎵 Your AI-Curated Playlist")
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

                # Show AI explanation if available
                ai_explanation = track.get('ai_explanation', '')
                if ai_explanation:
                    st.caption(f"🤖 AI: {ai_explanation}")
                elif hasattr(st.session_state.recommendation_engine, 'openai_client') and st.session_state.recommendation_engine.openai_client:
                    # Get AI analysis for this track
                    with st.spinner("Getting AI analysis..."):
                        try:
                            detailed_analysis = st.session_state.recommendation_engine.get_ai_song_analysis(
                                track.to_dict(), primary_mood, mood_input
                            )
                            st.caption(f"🤖 AI Analysis: {detailed_analysis}")
                        except Exception:
                            st.caption("🔧 Algorithmic recommendation")

            with col2:
                # Show different scores if available
                if 'mood_score' in track:
                    mood_score = track['mood_score']
                    st.metric("AI Score", f"{mood_score:.2f}")
                else:
                    mood_score = track.get('mood_match_score', 0.8)
                    st.metric("Match", f"{mood_score:.1f}")

            with col3:
                # Enhanced feedback system with persistent display
                track_id = f"{track['artist']}_{track['name']}"
                track_features = {
                    'valence': track.get('valence', 0.5),
                    'energy': track.get('energy', 0.5),
                    'danceability': track.get('danceability', 0.5)
                }
                
                # Initialize feedback state if not exists
                if 'track_feedback' not in st.session_state:
                    st.session_state.track_feedback = {}
                
                # Check if this track already has feedback
                track_feedback_key = f"{track_id}_{primary_mood}"
                has_feedback = track_feedback_key in st.session_state.track_feedback
                
                if has_feedback:
                    # Show feedback status instead of buttons
                    feedback_status = st.session_state.track_feedback[track_feedback_key]
                    if feedback_status == 'liked':
                        st.success("👍 Liked!")
                    else:
                        st.info("👎 Noted!")
                else:
                    # Show feedback buttons
                    feedback_col1, feedback_col2 = st.columns(2)
                    with feedback_col1:
                        if st.button("👍", key=f"like_{i}"):
                            # Add positive feedback
                            st.session_state.recommendation_engine.add_feedback(
                                track_id=track_id,
                                artist=track['artist'],
                                mood=primary_mood,
                                rating=1.0,
                                feedback_text="Liked",
                                track_features=track_features
                            )
                            save_feedback_data()  # Persist learning
                            st.session_state.track_feedback[track_feedback_key] = 'liked'
                            # No st.rerun() needed - will update on next interaction
                            
                    with feedback_col2:
                        if st.button("👎", key=f"dislike_{i}"):
                            # Add negative feedback
                            st.session_state.recommendation_engine.add_feedback(
                                track_id=track_id,
                                artist=track['artist'],
                                mood=primary_mood,
                                rating=0.0,
                                feedback_text="Disliked",
                                track_features=track_features
                            )
                            save_feedback_data()  # Persist learning
                            st.session_state.track_feedback[track_feedback_key] = 'disliked'
                            # No st.rerun() needed - will update on next interaction

        # Audio features explanation (moved outside the song loop)
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
                # Use uuid to ensure absolutely unique keys
                chart_key = f"mood_audio_features_{str(uuid.uuid4())[:8]}"
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
        
        # Show feedback insights if there's any feedback data (moved outside the song loop)
        insights = st.session_state.recommendation_engine.get_feedback_insights()
        if insights['total_feedback'] > 0:
            with st.expander(f"🧠 Learning Insights ({insights['total_feedback']} feedback items)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**System Learning Status:**")
                    st.info(f"Status: {insights['learning_status']}")
                    
                    if insights['favorite_artists']:
                        st.write("**Favorite Artists:**")
                        for artist, rating in sorted(insights['favorite_artists'].items(), 
                                                    key=lambda x: x[1], reverse=True)[:3]:
                            st.write(f"• {artist} ({rating:.1f}⭐)")
                    
                    if insights['disliked_artists']:
                        st.write("**Learning to Avoid:**")
                        for artist, rating in sorted(insights['disliked_artists'].items(), 
                                                    key=lambda x: x[1])[:3]:
                            st.write(f"• {artist} ({rating:.1f}⭐)")
                
                with col2:
                    st.write("**Feedback Summary:**")
                    st.metric("Total Feedback", insights['total_feedback'])
                    st.metric("Favorites", insights['favorites_count'])
                    st.metric("Blacklisted", insights['blacklist_count'])
                    
                    if insights['mood_feedback_counts']:
                        st.write("**Feedback by Mood:**")
                        for mood, count in insights['mood_feedback_counts'].items():
                            st.write(f"• {mood.title()}: {count} ratings")
    else:
        # No playlist generated yet
        st.info("🎵 Click the button above to analyze your mood and generate a personalized playlist!")


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
    st.plotly_chart(fig, use_container_width=True, key="mood_distribution_pie")

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
    st.plotly_chart(fig, use_container_width=True, key="mood_features_radar")

    # AI Song Comparison Demo (if OpenAI is available)
    if hasattr(st.session_state.recommendation_engine, 'openai_client') and st.session_state.recommendation_engine.openai_client:
        st.subheader("🤖 AI Song Comparison Demo")
        st.write("See how ChatGPT analyzes and ranks songs for different moods")

        # Let user select songs to compare
        available_songs = st.session_state.demo_tracks.sample(6)  # Random sample for demo

        col1, col2 = st.columns(2)
        with col1:
            comparison_mood = st.selectbox("Select Mood for Comparison",
                                         ['happy', 'sad', 'energetic', 'calm', 'romantic', 'focused'])

        with col2:
            if st.button("🔍 Analyze Songs with AI"):
                with st.spinner("ChatGPT is analyzing songs..."):
                    try:
                        # Convert to list of dicts for comparison
                        songs_list = available_songs.to_dict('records')

                        comparison_result = st.session_state.recommendation_engine.compare_songs_for_mood(
                            songs_list, comparison_mood
                        )

                        if comparison_result:
                            st.success("✅ AI Analysis Complete!")

                            rankings = comparison_result.get('ranking', [])
                            explanations = comparison_result.get('explanations', {})

                            st.write(f"**AI Rankings for '{comparison_mood}' mood:**")

                            for rank, song_idx in enumerate(rankings[:3]):  # Show top 3
                                if song_idx < len(songs_list):
                                    song = songs_list[song_idx]
                                    explanation = explanations.get(str(song_idx), "Good match for the mood")

                                    st.markdown(f"""
                                    **#{rank + 1}** 🎵 {song['name']} by {song['artist']}

                                    🤖 *AI Reasoning: {explanation}*
                                    """)
                        else:
                            st.error("AI comparison failed. Please try again.")

                    except Exception as e:
                        st.error(f"AI analysis error: {str(e)}")

        # Show the songs being compared
        st.write("**Songs in comparison:**")
        for i, (_, song) in enumerate(available_songs.iterrows()):
            st.write(f"**{i}:** {song['name']} by {song['artist']} (Valence: {song['valence']:.2f}, Energy: {song['energy']:.2f})")

    # Valence vs Energy scatter
    st.subheader("😊 Mood Mapping: Valence vs Energy")
    fig = px.scatter(st.session_state.demo_tracks,
                     x='valence', y='energy',
                     color='mood_category',
                     hover_data=['name', 'artist'],
                     title="Songs in Valence-Energy Space")
    st.plotly_chart(fig, use_container_width=True, key="valence_energy_scatter")


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

    # Feedback Management Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧠 Learning Management**")
    
    if 'recommendation_engine' in st.session_state:
        insights = st.session_state.recommendation_engine.get_feedback_insights()
        if insights['total_feedback'] > 0:
            st.sidebar.markdown(f"**Status**: {insights['learning_status']}")
            st.sidebar.markdown(f"**Total Feedback**: {insights['total_feedback']}")
            st.sidebar.markdown(f"**Favorites**: {insights['favorites_count']}")
            st.sidebar.markdown(f"**Blacklisted**: {insights['blacklist_count']}")
            
            if st.sidebar.button("🗑️ Clear All Feedback"):
                st.session_state.recommendation_engine.feedback_data = []
                st.session_state.recommendation_engine.learned_preferences = {
                    'artist_preferences': {},
                    'feature_preferences': {},
                    'genre_preferences': {},
                    'track_blacklist': set(),
                    'track_favorites': set()
                }
                if 'track_feedback' in st.session_state:
                    st.session_state.track_feedback = {}
                save_feedback_data()
                st.sidebar.success("✅ Feedback cleared!")
                st.rerun()
        else:
            st.sidebar.markdown("**Status**: No feedback yet")
            st.sidebar.markdown("Rate songs with 👍👎 to start learning!")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎵 AI Music Mood Matcher Demo**")
    st.sidebar.markdown("Built by @djleamen using Streamlit, Spotify API, OpenAI, and ML techniques")


if __name__ == "__main__":
    main()
