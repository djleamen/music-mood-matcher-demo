import os
import json
import pickle
import random
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from .utils import initialize_openai_client


class RecommendationEngine:
    def __init__(self):
        """Initialize recommendation engine"""
        self.user_preferences = {}
        self.feedback_data = []
        self.scaler = MinMaxScaler()
        self.track_vectors = None
        self.track_index = None

        # Initialize OpenAI client if available
        self.openai_client = initialize_openai_client("recommendations")

        # Recommendation weights
        self.weights = {
            'mood_match': 0.4,
            'user_preference': 0.3,
            'popularity': 0.1,
            'diversity': 0.1,
            'feedback_history': 0.1
        }

        # Enhanced feedback learning
        self.learned_preferences = {
            'artist_preferences': {},  # artist -> {mood: avg_rating}
            'feature_preferences': {},  # mood -> {feature: preferred_range}
            'genre_preferences': {},   # mood -> {genre: preference_score}
            'track_blacklist': set(),  # tracks that received negative feedback
            'track_favorites': set()   # tracks that received positive feedback
        }

    def learn_user_preferences(self, user_tracks_df: pd.DataFrame):
        """Learn user preferences from their music library"""
        if user_tracks_df.empty:
            return

        audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo'
        ]

        available_features = [f for f in audio_features if f in user_tracks_df.columns]

        if not available_features:
            return

        # Calculate user preference profile
        self.user_preferences = user_tracks_df[available_features].mean().to_dict()

        # Calculate feature vectors for similarity
        feature_data = user_tracks_df[available_features].fillna(user_tracks_df[available_features].median())
        self.track_vectors = self.scaler.fit_transform(feature_data)
        self.track_index = user_tracks_df.index.tolist()

    def generate_mood_playlist(self,
                               mood_analysis: Dict,
                               all_tracks_df: pd.DataFrame,
                               playlist_size: int = 20,
                               diversity_factor: float = 0.3,
                               use_ai_enhancement: bool = True) -> pd.DataFrame:
        """Generate a playlist based on mood analysis - uses AI enhancement if available"""

        if all_tracks_df.empty:
            return pd.DataFrame()

        # Use AI-enhanced recommendations if OpenAI is available and enabled
        if use_ai_enhancement and self.openai_client:
            return self.generate_ai_enhanced_mood_playlist(
                mood_analysis, all_tracks_df, playlist_size, diversity_factor
            )

        # Fallback to standard method
        audio_preferences = mood_analysis.get('audio_preferences', {})

        # Score all tracks for the target mood
        scored_tracks = self._score_tracks_for_mood(all_tracks_df, audio_preferences)

        # Apply user preference scoring
        if self.user_preferences:
            scored_tracks = self._apply_user_preferences(scored_tracks)

        # Apply feedback-based adjustments
        scored_tracks = self._apply_feedback_adjustments(scored_tracks)

        # Generate diverse playlist
        playlist = self._generate_diverse_playlist(scored_tracks, playlist_size, diversity_factor)

        return playlist

    def _score_tracks_for_mood(self, tracks_df: pd.DataFrame, audio_preferences: Dict) -> pd.DataFrame:
        """Score tracks based on how well they match the target mood"""
        scored_df = tracks_df.copy()

        audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo'
        ]

        available_features = [f for f in audio_features if f in tracks_df.columns]

        if not available_features or not audio_preferences:
            # If no preferences available, use random scoring
            scored_df['mood_score'] = np.random.random(len(tracks_df))
            return scored_df

        mood_scores = []

        for _, track in tracks_df.iterrows():
            score = 0
            count = 0

            for feature in available_features:
                if feature in audio_preferences and not pd.isna(track[feature]):
                    preference = audio_preferences[feature]

                    if isinstance(preference, tuple) and len(preference) == 2:
                        min_val, max_val = preference
                        track_val = track[feature]

                        # Calculate how well the track fits the mood preference
                        if min_val <= track_val <= max_val:
                            # Perfect match
                            feature_score = 1.0
                        else:
                            # Calculate distance penalty
                            if track_val < min_val:
                                distance = (min_val - track_val) / max(min_val, 0.1)
                            else:
                                distance = (track_val - max_val) / max(max_val, 0.1)

                            feature_score = max(0, 1.0 - distance)

                        score += feature_score
                        count += 1

            mood_scores.append(score / count if count > 0 else 0.5)

        scored_df['mood_score'] = mood_scores
        return scored_df

    def _apply_user_preferences(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Apply user preference scoring"""
        if not self.user_preferences:
            scored_df['preference_score'] = 0.5
            return scored_df

        audio_features = list(self.user_preferences.keys())
        available_features = [f for f in audio_features if f in scored_df.columns]

        if not available_features:
            scored_df['preference_score'] = 0.5
            return scored_df

        preference_scores = []

        for _, track in scored_df.iterrows():
            similarity_sum = 0
            count = 0

            for feature in available_features:
                if not pd.isna(track[feature]):
                    user_pref = self.user_preferences[feature]
                    track_val = track[feature]

                    # Calculate similarity (inverse of absolute difference, normalized)
                    max_range = 1.0 if feature != 'tempo' else 200.0
                    diff = abs(user_pref - track_val) / max_range
                    similarity = max(0, 1.0 - diff)

                    similarity_sum += similarity
                    count += 1

            preference_scores.append(similarity_sum / count if count > 0 else 0.5)

        scored_df['preference_score'] = preference_scores
        return scored_df

    def _apply_feedback_adjustments(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Apply adjustments based on learned user preferences"""
        scored_df = scored_df.copy()
        scored_df['feedback_score'] = 0.5  # Default neutral score

        if not self.feedback_data:
            return scored_df

        # Apply learned preferences
        feedback_scores = []

        for _, track in scored_df.iterrows():
            artist = track.get('artist', '')
            track_id = track.get('track_id', str(track.name))

            # Start with neutral score
            feedback_score = 0.5

            # Check blacklist/favorites first (strongest signal)
            if track_id in self.learned_preferences['track_blacklist']:
                feedback_score = 0.1  # Strongly penalize blacklisted tracks
            elif track_id in self.learned_preferences['track_favorites']:
                feedback_score = 0.9  # Strongly favor liked tracks
            else:
                # Use artist preferences if available
                if artist in self.learned_preferences['artist_preferences']:
                    artist_prefs = self.learned_preferences['artist_preferences'][artist]

                    # Get average rating for this artist across all moods
                    all_ratings = []
                    for mood_ratings in artist_prefs.values():
                        all_ratings.extend(mood_ratings)

                    if all_ratings:
                        feedback_score = np.mean(all_ratings)

            feedback_scores.append(feedback_score)

        scored_df['feedback_score'] = feedback_scores

        # Apply feedback adjustment to final score
        scored_df['final_score'] = (
            scored_df['mood_score'] * self.weights['mood_match'] +
            scored_df.get('user_score', 0.5) * self.weights['user_preference'] +
            scored_df['feedback_score'] * self.weights['feedback_history'] +
            scored_df.get('popularity_score', 0.5) * self.weights['popularity']
        )

        return scored_df

    def _generate_diverse_playlist(
            self,
            scored_df: pd.DataFrame,
            playlist_size: int,
            diversity_factor: float) -> pd.DataFrame:
        """Generate a diverse playlist with high scores"""
        if scored_df.empty:
            return pd.DataFrame()

        # Calculate composite score
        scored_df['composite_score'] = (
            scored_df.get('mood_score', 0.5) * self.weights['mood_match']
            + scored_df.get('preference_score', 0.5) * self.weights['user_preference']
            + scored_df.get('popularity', 50) / 100 * self.weights['popularity']
            + scored_df.get('feedback_score', 0.5) * self.weights['feedback_history']
        )

        # Sort by composite score
        sorted_df = scored_df.sort_values('composite_score', ascending=False)

        playlist = []

        # Use a combination of top scores and diversity
        top_candidates = sorted_df.head(min(playlist_size * 3, len(sorted_df)))

        for _, track in top_candidates.iterrows():
            if len(playlist) >= playlist_size:
                break

            artist = track.get('artist', '')

            # Diversity check: limit songs per artist
            artist_count = sum(1 for p in playlist if p.get('artist', '') == artist)
            max_per_artist = max(1, playlist_size // 10)  # Max 10% from same artist

            if artist_count < max_per_artist:
                playlist.append(track.to_dict())
            elif random.random() < diversity_factor:
                # Sometimes include anyway for diversity
                playlist.append(track.to_dict())

        # Fill remaining slots with random high-scoring tracks
        while len(playlist) < playlist_size and len(playlist) < len(sorted_df):
            remaining_tracks = sorted_df[~sorted_df.index.isin([p.get('index', -1) for p in playlist])]
            if not remaining_tracks.empty:
                random_track = remaining_tracks.sample(1).iloc[0]
                playlist.append(random_track.to_dict())
            else:
                break

        return pd.DataFrame(playlist)

    def add_feedback(self, track_id: str, artist: str, mood: str, rating: float, feedback_text: str = "", track_features: Dict = None):
        """Add user feedback for a track recommendation and update model"""
        feedback_entry = {
            'track_id': track_id,
            'artist': artist,
            'mood': mood,
            'rating': rating,  # 0.0 to 1.0 (thumbs down=0.0, thumbs up=1.0)
            'feedback_text': feedback_text,
            'timestamp': pd.Timestamp.now(),
            'track_features': track_features or {}
        }

        self.feedback_data.append(feedback_entry)

        # Learn from this feedback immediately
        self._learn_from_feedback(feedback_entry)

        # Update weights based on feedback patterns
        self._update_recommendation_weights()

        print(f"📝 Feedback recorded: {artist} - {rating:.1f} rating for {mood} mood")

    def _learn_from_feedback(self, feedback: Dict):
        """Learn preferences from individual feedback entry"""
        artist = feedback.get('artist', '')
        mood = feedback.get('mood', '')
        rating = feedback.get('rating', 0.5)
        track_id = feedback.get('track_id', '')
        track_features = feedback.get('track_features', {})

        # Learn artist preferences per mood
        if artist and mood:
            if artist not in self.learned_preferences['artist_preferences']:
                self.learned_preferences['artist_preferences'][artist] = {}

            if mood not in self.learned_preferences['artist_preferences'][artist]:
                self.learned_preferences['artist_preferences'][artist][mood] = []

            self.learned_preferences['artist_preferences'][artist][mood].append(rating)

        # Learn audio feature preferences per mood
        if track_features and mood:
            if mood not in self.learned_preferences['feature_preferences']:
                self.learned_preferences['feature_preferences'][mood] = {}

            for feature, value in track_features.items():
                if isinstance(value, (int, float)):
                    if feature not in self.learned_preferences['feature_preferences'][mood]:
                        self.learned_preferences['feature_preferences'][mood][feature] = []

                    # Weight the feature value by the rating
                    weighted_value = value * rating
                    self.learned_preferences['feature_preferences'][mood][feature].append(weighted_value)

        # Track blacklist and favorites
        if track_id:
            if rating <= 0.3:  # Negative feedback
                self.learned_preferences['track_blacklist'].add(track_id)
                self.learned_preferences['track_favorites'].discard(track_id)
            elif rating >= 0.7:  # Positive feedback
                self.learned_preferences['track_favorites'].add(track_id)
                self.learned_preferences['track_blacklist'].discard(track_id)

    def _update_recommendation_weights(self):
        """Update recommendation weights based on feedback patterns"""
        if len(self.feedback_data) < 10:
            return  # Need enough data

        recent_feedback = self.feedback_data[-20:]  # Last 20 feedback items
        avg_rating = np.mean([f['rating'] for f in recent_feedback])

        # If ratings are low, adjust weights to emphasize user preferences more
        if avg_rating < 0.6:
            self.weights['user_preference'] = min(0.5, self.weights['user_preference'] + 0.05)
            self.weights['mood_match'] = max(0.2, self.weights['mood_match'] - 0.05)
        elif avg_rating > 0.8:
            # If ratings are high, can explore more mood-based recommendations
            self.weights['mood_match'] = min(0.6, self.weights['mood_match'] + 0.05)
            self.weights['diversity'] = min(0.2, self.weights['diversity'] + 0.02)

    def get_feedback_insights(self) -> Dict:
        """Get insights about learned preferences from feedback"""
        insights = {
            'total_feedback': len(self.feedback_data),
            'favorite_artists': {},
            'disliked_artists': {},
            'mood_feedback_counts': {},
            'learning_status': 'Active learning' if len(self.feedback_data) > 0 else 'No feedback yet'
        }

        if not self.feedback_data:
            return insights

        # Analyze artist preferences
        for artist, mood_ratings in self.learned_preferences['artist_preferences'].items():
            avg_ratings = []
            for ratings_list in mood_ratings.values():
                avg_ratings.extend(ratings_list)

            if avg_ratings:
                avg_rating = np.mean(avg_ratings)
                if avg_rating >= 0.7:
                    insights['favorite_artists'][artist] = avg_rating
                elif avg_rating <= 0.3:
                    insights['disliked_artists'][artist] = avg_rating

        # Count feedback per mood
        for feedback in self.feedback_data:
            mood = feedback.get('mood', 'unknown')
            if mood not in insights['mood_feedback_counts']:
                insights['mood_feedback_counts'][mood] = 0
            insights['mood_feedback_counts'][mood] += 1

        insights['favorites_count'] = len(self.learned_preferences['track_favorites'])
        insights['blacklist_count'] = len(self.learned_preferences['track_blacklist'])

        return insights

    def get_similar_tracks(self, track_id: str, all_tracks_df: pd.DataFrame, n_similar: int = 10) -> pd.DataFrame:
        """Find tracks similar to a given track"""
        if all_tracks_df.empty or track_id not in all_tracks_df.index:
            return pd.DataFrame()

        audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo'
        ]

        available_features = [f for f in audio_features if f in all_tracks_df.columns]

        if not available_features:
            return all_tracks_df.sample(min(n_similar, len(all_tracks_df)))

        # Get target track features
        target_track = all_tracks_df.loc[track_id, available_features]

        # Calculate similarity to all other tracks
        similarities = []
        for idx, track in all_tracks_df.iterrows():
            if idx == track_id:
                similarities.append(0)  # Exclude the track itself
                continue

            track_features = track[available_features]

            # Calculate cosine similarity
            try:
                sim = cosine_similarity([target_track.values], [track_features.values])[0][0]
                similarities.append(sim)
            except (ValueError, IndexError):
                similarities.append(0)

        all_tracks_df['similarity'] = similarities
        similar_tracks = all_tracks_df.nlargest(n_similar, 'similarity')

        return similar_tracks.drop('similarity', axis=1)

    def get_recommendation_explanation(self, track: Dict, mood: str) -> str:
        """Generate explanation for why a track was recommended"""
        explanations = []

        # Mood match explanation
        mood_score = track.get('mood_score', 0)
        if mood_score > 0.7:
            explanations.append(f"Excellent match for {mood} mood")
        elif mood_score > 0.5:
            explanations.append(f"Good match for {mood} mood")

        # User preference explanation
        pref_score = track.get('preference_score', 0)
        if pref_score > 0.7:
            explanations.append("Matches your music taste well")

        # Popularity explanation
        popularity = track.get('popularity', 0)
        if popularity > 70:
            explanations.append("Popular track")

        # Feedback explanation
        feedback_score = track.get('feedback_score', 0.5)
        if feedback_score > 0.7:
            explanations.append("Based on your previous positive feedback")

        if not explanations:
            explanations.append("Recommended based on overall analysis")

        return " • ".join(explanations)

    def save_recommendation_data(self, filepath: str):
        """Save recommendation engine data including learned preferences"""
        data = {
            'user_preferences': self.user_preferences,
            'feedback_data': self.feedback_data,
            'learned_preferences': self.learned_preferences,
            'weights': self.weights,
            'scaler': self.scaler
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Saved {len(self.feedback_data)} feedback entries and learned preferences")

    def load_recommendation_data(self, filepath: str):
        """Load recommendation engine data including learned preferences"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.user_preferences = data.get('user_preferences', {})
                self.feedback_data = data.get('feedback_data', [])
                self.learned_preferences = data.get('learned_preferences', {
                    'artist_preferences': {},
                    'feature_preferences': {},
                    'genre_preferences': {},
                    'track_blacklist': set(),
                    'track_favorites': set()
                })
                self.weights = data.get('weights', self.weights)
                self.scaler = data.get('scaler', MinMaxScaler())
                print(f"✅ Loaded {len(self.feedback_data)} feedback entries and learned preferences")
                return True
        return False

    def get_stats(self) -> Dict:
        """Get recommendation engine statistics"""
        return {
            'total_feedback': len(self.feedback_data),
            'avg_rating': np.mean([f['rating'] for f in self.feedback_data]) if self.feedback_data else 0,
            'user_preferences_learned': bool(self.user_preferences),
            'recommendation_weights': self.weights
        }

    def generate_ai_enhanced_mood_playlist(self,
                                          mood_analysis: Dict,
                                          all_tracks_df: pd.DataFrame,
                                          playlist_size: int = 20,
                                          diversity_factor: float = 0.3) -> pd.DataFrame:
        """Generate a playlist using ChatGPT to analyze which songs best fit the mood"""

        if all_tracks_df.empty:
            return pd.DataFrame()

        # Use ChatGPT if available, otherwise fallback to standard method
        if self.openai_client:
            return self._generate_playlist_with_chatgpt(
                mood_analysis, all_tracks_df, playlist_size, diversity_factor
            )
        return self.generate_mood_playlist(
            mood_analysis, all_tracks_df, playlist_size, diversity_factor
        )

    def _generate_playlist_with_chatgpt(self,
                                      mood_analysis: Dict,
                                      all_tracks_df: pd.DataFrame,
                                      playlist_size: int,
                                      diversity_factor: float) -> pd.DataFrame:
        """Use ChatGPT to analyze and rank songs for the detected mood"""

        mood = mood_analysis.get('primary_mood', 'calm')
        reasoning = mood_analysis.get('reasoning', '')
        audio_preferences = mood_analysis.get('audio_preferences', {})

        # First, do a preliminary filtering based on audio features to reduce the dataset
        # This helps avoid hitting token limits with too many songs
        filtered_tracks = self._pre_filter_tracks_for_mood(all_tracks_df, audio_preferences, min(50, len(all_tracks_df)))

        if filtered_tracks.empty:
            return self.generate_mood_playlist(mood_analysis, all_tracks_df, playlist_size, diversity_factor)

        try:
            # Prepare song data for ChatGPT analysis
            songs_for_analysis = []
            for idx, track in filtered_tracks.iterrows():
                # Debug: check track data structure
                if not isinstance(track, pd.Series):
                    print(f"⚠️ Debug: track is not a Series: {type(track)}")
                    continue

                song_info = {
                    'id': str(idx),
                    'name': track.get('name', 'Unknown'),
                    'artist': track.get('artist', 'Unknown Artist'),
                    'audio_features': {
                        'valence': round(track.get('valence', 0.5), 2),
                        'energy': round(track.get('energy', 0.5), 2),
                        'danceability': round(track.get('danceability', 0.5), 2),
                        'acousticness': round(track.get('acousticness', 0.5), 2),
                        'instrumentalness': round(track.get('instrumentalness', 0.5), 2),
                        'tempo': round(track.get('tempo', 120), 0)
                    }
                }
                songs_for_analysis.append(song_info)

            # Create prompt for ChatGPT
            prompt = self._create_song_analysis_prompt(mood, reasoning, songs_for_analysis, playlist_size, audio_preferences)

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert music curator and mood analyst. You understand how audio features relate to emotional states and can identify which songs best match specific moods."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            # Parse ChatGPT response
            result_text = response.choices[0].message.content.strip()
            recommended_ids = self._parse_chatgpt_recommendations(result_text)

            # Build final playlist from recommendations
            recommended_playlist = self._build_playlist_from_chatgpt_recs(
                filtered_tracks, recommended_ids, playlist_size
            )

            if not recommended_playlist.empty:
                # Add explanation for each track
                recommended_playlist['ai_explanation'] = recommended_playlist.apply(
                    lambda row: self._get_ai_song_explanation(row, mood), axis=1
                )
                return recommended_playlist
            # Fallback to standard method
            return self.generate_mood_playlist(mood_analysis, all_tracks_df, playlist_size, diversity_factor)

        except (KeyError, AttributeError, ValueError) as e:
            print(f"⚠️ ChatGPT song analysis failed: {e}")
            # Fallback to standard method (disable AI enhancement to prevent recursion)
            return self.generate_mood_playlist(mood_analysis, all_tracks_df, playlist_size, diversity_factor, use_ai_enhancement=False)

    def _pre_filter_tracks_for_mood(self, tracks_df: pd.DataFrame, audio_preferences: Dict, max_tracks: int) -> pd.DataFrame:
        """Pre-filter tracks based on audio features to reduce dataset size for ChatGPT analysis"""

        if not audio_preferences or tracks_df.empty:
            return tracks_df.sample(min(max_tracks, len(tracks_df)))

        # Score tracks based on audio preferences
        scored_tracks = self._score_tracks_for_mood(tracks_df, audio_preferences)

        # Return top scoring tracks
        top_tracks = scored_tracks.nlargest(max_tracks, 'mood_score')
        return top_tracks.drop('mood_score', axis=1)

    def _create_song_analysis_prompt(self, mood: str, reasoning: str, songs: list, playlist_size: int, audio_preferences: Dict) -> str:
        """Create a detailed prompt for ChatGPT song analysis"""

        # Convert audio preferences to readable description
        pref_description = ""
        if audio_preferences:
            pref_description = f"""
Target Audio Features for {mood} mood:
"""
            for feature, (min_val, max_val) in audio_preferences.items():
                if feature != 'description':
                    pref_description += f"- {feature}: {min_val:.2f} to {max_val:.2f}\n"

        prompt = f"""
I need you to analyze these songs and select the {playlist_size} best ones for someone in a "{mood}" mood.

Context: {reasoning}

{pref_description}

Audio Feature Definitions:
- Valence: Musical positiveness (0.0 = sad/negative, 1.0 = happy/positive)
- Energy: Intensity and power (0.0 = calm, 1.0 = intense)
- Danceability: How suitable for dancing (0.0 = not danceable, 1.0 = very danceable)
- Acousticness: Whether the track is acoustic (0.0 = electronic, 1.0 = acoustic)
- Instrumentalness: Whether track contains vocals (0.0 = vocals, 1.0 = instrumental)
- Tempo: BPM (beats per minute)

Available Songs:
{json.dumps(songs, indent=2)}

Please analyze each song and:
1. Consider how well the song's name, artist, and audio features match the "{mood}" mood
2. Think about the emotional impact and appropriateness for this mood
3. Select the {playlist_size} best matches

Respond with a JSON object containing:
{{
    "recommended_song_ids": ["id1", "id2", "id3", ...],
    "reasoning": "Brief explanation of your selection criteria",
    "song_explanations": {{
        "id1": "Why this song fits the mood",
        "id2": "Why this song fits the mood"
    }}
}}

Focus on songs that truly capture the essence of the "{mood}" mood. Return only valid JSON.
"""
        return prompt

    def _parse_chatgpt_recommendations(self, response_text: str) -> list:
        """Parse ChatGPT response to extract recommended song IDs"""
        try:
            # Clean up the response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            result = json.loads(response_text)
            return result.get('recommended_song_ids', [])

        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse ChatGPT recommendations: {e}")
            return []

    def _build_playlist_from_chatgpt_recs(self, filtered_tracks: pd.DataFrame, recommended_ids: list,
                                        playlist_size: int) -> pd.DataFrame:
        """Build final playlist from ChatGPT recommendations"""

        if not recommended_ids:
            return pd.DataFrame()

        # Convert string IDs back to indices and filter tracks
        playlist_tracks = []
        for track_id in recommended_ids[:playlist_size]:
            try:
                # Handle both string and integer indices
                if track_id in filtered_tracks.index:
                    idx = track_id
                else:
                    # Try to convert to appropriate type
                    try:
                        idx = int(track_id)
                    except ValueError:
                        continue

                if idx in filtered_tracks.index:
                    playlist_tracks.append(filtered_tracks.loc[idx])

            except (KeyError, ValueError):
                continue

        if playlist_tracks:
            return pd.DataFrame(playlist_tracks)
        return pd.DataFrame()

    def _get_ai_song_explanation(self, track_row, mood: str) -> str:
        """Generate explanation for why AI selected this song for the mood"""
        valence = track_row.get('valence', 0.5)
        energy = track_row.get('energy', 0.5)
        danceability = track_row.get('danceability', 0.5)

        explanations = []

        if mood == 'happy' and valence > 0.7:
            explanations.append("high positivity")
        elif mood == 'sad' and valence < 0.3:
            explanations.append("melancholic feel")
        elif mood == 'energetic' and energy > 0.7:
            explanations.append("high energy")
        elif mood == 'calm' and energy < 0.4:
            explanations.append("peaceful energy")
        elif mood == 'party' and danceability > 0.7:
            explanations.append("great danceability")

        if explanations:
            return f"Selected for {mood} mood due to {', '.join(explanations)}"
        return f"AI-curated match for {mood} mood"

    def get_ai_song_analysis(self, track: Dict, mood: str, user_context: str = "") -> str:
        """Get detailed AI analysis of why a specific song fits a mood"""

        if not self.openai_client:
            return self.get_recommendation_explanation(track, mood)

        try:
            # Prepare track information
            track_info = {
                'name': track.get('name', 'Unknown'),
                'artist': track.get('artist', 'Unknown Artist'),
                'audio_features': {
                    'valence': round(track.get('valence', 0.5), 2),
                    'energy': round(track.get('energy', 0.5), 2),
                    'danceability': round(track.get('danceability', 0.5), 2),
                    'acousticness': round(track.get('acousticness', 0.5), 2),
                    'instrumentalness': round(track.get('instrumentalness', 0.5), 2),
                    'tempo': round(track.get('tempo', 120), 0)
                }
            }

            context_part = f"\nUser context: {user_context}" if user_context else ""

            prompt = f"""
Analyze why this song is a good fit for someone in a "{mood}" mood:

Song: "{track_info['name']}" by {track_info['artist']}
Audio Features: {json.dumps(track_info['audio_features'], indent=2)}
{context_part}

Please provide a detailed analysis that includes:
1. How the song's audio features align with the "{mood}" mood
2. What specific elements make it suitable for this emotional state
3. The overall vibe and feeling the song conveys

Keep the explanation concise but insightful (2-3 sentences maximum).
Focus on connecting the technical audio features to the emotional experience.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert music analyst who can explain how songs connect to emotions and moods through their audio characteristics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.4
            )

            explanation = response.choices[0].message.content.strip()
            return explanation

        except (KeyError, AttributeError, ValueError) as e:
            print(f"⚠️ AI song analysis failed: {e}")
            return self.get_recommendation_explanation(track, mood)

    def compare_songs_for_mood(self, songs: list, mood: str) -> Dict:
        """Use AI to compare multiple songs and rank them for a specific mood"""

        if not self.openai_client or len(songs) < 2:
            return {}

        try:
            # Prepare songs for comparison
            songs_data = []
            for i, song in enumerate(songs):
                song_info = {
                    'id': i,
                    'name': song.get('name', 'Unknown'),
                    'artist': song.get('artist', 'Unknown Artist'),
                    'valence': round(song.get('valence', 0.5), 2),
                    'energy': round(song.get('energy', 0.5), 2),
                    'danceability': round(song.get('danceability', 0.5), 2),
                    'tempo': round(song.get('tempo', 120), 0)
                }
                songs_data.append(song_info)

            prompt = f"""
Compare these songs and rank them from best to worst for someone in a "{mood}" mood:

Songs to compare:
{json.dumps(songs_data, indent=2)}

Please analyze each song's suitability for the "{mood}" mood and provide:
1. A ranking from best to worst (use the song IDs)
2. Brief reasoning for each song's ranking

Respond with JSON:
{{
    "ranking": [0, 1, 2, ...],
    "explanations": {{
        "0": "Why this song ranks here",
        "1": "Why this song ranks here"
    }}
}}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing and ranking music for different moods and emotional states."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )

            result_text = response.choices[0].message.content.strip()

            # Parse response
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]

            return json.loads(result_text)

        except (KeyError, AttributeError, ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ AI song comparison failed: {e}")
            return {}
