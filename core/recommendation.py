import pandas as pd
import numpy as np
from typing import Dict
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class RecommendationEngine:
    def __init__(self):
        """Initialize recommendation engine"""
        self.user_preferences = {}
        self.feedback_data = []
        self.scaler = MinMaxScaler()
        self.track_vectors = None
        self.track_index = None

        # Recommendation weights
        self.weights = {
            'mood_match': 0.4,
            'user_preference': 0.3,
            'popularity': 0.1,
            'diversity': 0.1,
            'feedback_history': 0.1
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
                               diversity_factor: float = 0.3) -> pd.DataFrame:
        """Generate a playlist based on mood analysis"""

        if all_tracks_df.empty:
            return pd.DataFrame()

        mood = mood_analysis.get('primary_mood', 'calm')
        audio_preferences = mood_analysis.get('audio_preferences', {})

        # Score all tracks for the target mood
        scored_tracks = self._score_tracks_for_mood(all_tracks_df, audio_preferences, mood)

        # Apply user preference scoring
        if self.user_preferences:
            scored_tracks = self._apply_user_preferences(scored_tracks)

        # Apply feedback-based adjustments
        scored_tracks = self._apply_feedback_adjustments(scored_tracks)

        # Generate diverse playlist
        playlist = self._generate_diverse_playlist(scored_tracks, playlist_size, diversity_factor)

        return playlist

    def _score_tracks_for_mood(self, tracks_df: pd.DataFrame, audio_preferences: Dict, mood: str) -> pd.DataFrame:
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
        """Apply adjustments based on user feedback history"""
        scored_df['feedback_score'] = 0.5  # Default neutral score

        if not self.feedback_data:
            return scored_df

        # Group feedback by artist and mood
        artist_feedback = {}
        mood_feedback = {}

        for feedback in self.feedback_data:
            artist = feedback.get('artist', '')
            mood = feedback.get('mood', '')
            rating = feedback.get('rating', 0.5)

            if artist:
                if artist not in artist_feedback:
                    artist_feedback[artist] = []
                artist_feedback[artist].append(rating)

            if mood:
                if mood not in mood_feedback:
                    mood_feedback[mood] = []
                mood_feedback[mood].append(rating)

        # Calculate average feedback scores
        artist_avg = {artist: np.mean(ratings) for artist, ratings in artist_feedback.items()}

        feedback_scores = []

        for _, track in scored_df.iterrows():
            artist = track.get('artist', '')

            # Start with neutral score
            feedback_score = 0.5

            # Adjust based on artist feedback
            if artist in artist_avg:
                feedback_score = artist_avg[artist]

            feedback_scores.append(feedback_score)

        scored_df['feedback_score'] = feedback_scores
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

    def add_feedback(self, track_id: str, artist: str, mood: str, rating: float, feedback_text: str = ""):
        """Add user feedback for a track recommendation"""
        feedback_entry = {
            'track_id': track_id,
            'artist': artist,
            'mood': mood,
            'rating': rating,  # 0.0 to 1.0
            'feedback_text': feedback_text,
            'timestamp': pd.Timestamp.now()
        }

        self.feedback_data.append(feedback_entry)

        # Update weights based on feedback patterns
        self._update_recommendation_weights()

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
        """Save recommendation engine data"""
        data = {
            'user_preferences': self.user_preferences,
            'feedback_data': self.feedback_data,
            'weights': self.weights,
            'scaler': self.scaler
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_recommendation_data(self, filepath: str):
        """Load recommendation engine data"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.user_preferences = data.get('user_preferences', {})
                self.feedback_data = data.get('feedback_data', [])
                self.weights = data.get('weights', self.weights)
                self.scaler = data.get('scaler', MinMaxScaler())
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
