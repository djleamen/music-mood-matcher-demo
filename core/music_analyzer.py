import os
import pickle
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .mood_analyzer import MoodAnalyzer


class MusicAnalyzer:
    def __init__(self):
        """Initialize music analyzer with audio feature definitions"""
        self.audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo'
        ]

        self.scaler = StandardScaler()
        self.pca = None
        self.clusters = None
        self.cluster_model = None
        self.mood_models = {}

        # Feature descriptions for interpretability
        self.feature_descriptions = {
            'danceability': 'How suitable a track is for dancing (0.0-1.0)',
            'energy': 'Perceptual measure of intensity and power (0.0-1.0)',
            'key': 'Musical key the track is in (0-11)',
            'loudness': 'Overall loudness in decibels (-60 to 0)',
            'mode': 'Modality (major=1, minor=0)',
            'speechiness': 'Presence of spoken words (0.0-1.0)',
            'acousticness': 'Confidence the track is acoustic (0.0-1.0)',
            'instrumentalness': 'Predicts if track contains no vocals (0.0-1.0)',
            'liveness': 'Detects presence of audience (0.0-1.0)',
            'valence': 'Musical positiveness conveyed (0.0-1.0)',
            'tempo': 'Overall estimated tempo in BPM'
        }

    def analyze_music_data(self, df: pd.DataFrame) -> Dict:
        """Analyze music data and extract insights"""
        if df.empty:
            return {}

        # Ensure we have the required audio features
        missing_features = [f for f in self.audio_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing audio features: {missing_features}")
            # Use only available features
            available_features = [f for f in self.audio_features if f in df.columns]
        else:
            available_features = self.audio_features.copy()

        if not available_features:
            return {'error': 'No audio features available for analysis'}

        # Basic statistics
        stats = df[available_features].describe()

        # Correlation analysis
        correlation_matrix = df[available_features].corr()

        # User preference profile (mean values)
        preference_profile = df[available_features].mean().to_dict()

        # Diversity analysis (standard deviation)
        diversity_profile = df[available_features].std().to_dict()

        return {
            'stats': stats,
            'correlation_matrix': correlation_matrix,
            'preference_profile': preference_profile,
            'diversity_profile': diversity_profile,
            'total_tracks': len(df),
            'unique_artists': df['artist'].nunique() if 'artist' in df.columns else 0,
            'available_features': available_features
        }

    def cluster_music(self, df: pd.DataFrame, n_clusters: int = 5) -> Tuple[pd.DataFrame, Dict]:
        """Cluster music based on audio features"""
        if df.empty:
            return df, {}

        # Get available features
        available_features = [f for f in self.audio_features if f in df.columns]
        if not available_features:
            return df, {'error': 'No audio features available for clustering'}

        # Prepare data
        feature_data = df[available_features].fillna(df[available_features].median())

        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)

        # Apply PCA for dimensionality reduction (optional)
        if len(available_features) > 3:
            self.pca = PCA(n_components=min(3, len(available_features)))
            self.pca.fit_transform(scaled_features)

        # Perform clustering
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.cluster_model.fit_predict(scaled_features)

        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters

        # Analyze clusters
        cluster_analysis = self._analyze_clusters(df_clustered, available_features)

        return df_clustered, cluster_analysis

    def _analyze_clusters(self, df: pd.DataFrame, features: List[str]) -> Dict:
        """Analyze characteristics of each cluster"""
        cluster_analysis = {}

        for cluster_id in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster_id]

            # Calculate mean features for this cluster
            cluster_profile = cluster_data[features].mean().to_dict()

            # Get representative tracks
            representative_tracks = cluster_data.sample(min(5, len(cluster_data)))[
                ['name', 'artist', 'playlist_name']
            ].to_dict('records') if 'name' in cluster_data.columns else []

            # Determine cluster mood based on audio features
            cluster_mood = self._infer_cluster_mood(cluster_profile)

            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'profile': cluster_profile,
                'representative_tracks': representative_tracks,
                'inferred_mood': cluster_mood
            }

        return cluster_analysis

    def _infer_cluster_mood(self, profile: Dict) -> str:
        """Infer mood from cluster audio feature profile"""
        valence = profile.get('valence', 0.5)
        energy = profile.get('energy', 0.5)
        danceability = profile.get('danceability', 0.5)
        acousticness = profile.get('acousticness', 0.5)
        tempo = profile.get('tempo', 120)

        # Rule-based mood inference
        if valence > 0.6 and energy > 0.6 and danceability > 0.6:
            return 'happy/energetic'
        if valence < 0.4 and energy < 0.5:
            return 'sad/melancholic'
        if energy > 0.7 and tempo > 120:
            return 'energetic/pump-up'
        if acousticness > 0.5 and energy < 0.5:
            return 'calm/acoustic'
        if 0.4 < valence < 0.7 and energy < 0.6:
            return 'chill/relaxed'
        return 'neutral/mixed'

    def build_mood_models(self, df: pd.DataFrame, mood_analyzer) -> Dict:
        """Build predictive models for each mood"""
        if df.empty:
            return {}

        available_features = [f for f in self.audio_features if f in df.columns]
        if not available_features:
            return {'error': 'No audio features available for modeling'}

        mood_models = {}

        for mood in mood_analyzer.get_available_moods():
            mood_preferences = mood_analyzer.mood_mappings[mood]

            # Create target variable based on how well each track matches the mood
            df['mood_score'] = self._calculate_mood_score(df, mood_preferences, available_features)

            # Prepare training data
            feature_matrix = df[available_features].fillna(df[available_features].median())
            y = df['mood_score']

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)

            # Evaluate model
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)

            mood_models[mood] = {
                'model': model,
                'features': available_features,
                'mse': mse,
                'feature_importance': dict(zip(available_features, model.feature_importances_))
            }

        self.mood_models = mood_models
        return mood_models

    def _calculate_mood_score(self, df: pd.DataFrame, mood_preferences: Dict, features: List[str]) -> pd.Series:
        """Calculate how well each track matches a mood profile"""
        scores = pd.Series(index=df.index, dtype=float)

        for idx, row in df.iterrows():
            score = 0
            count = 0

            for feature in features:
                if feature in mood_preferences and feature in row:
                    feature_value = row[feature]
                    if isinstance(mood_preferences[feature], tuple):
                        min_val, max_val = mood_preferences[feature]
                        # Score based on how well the value fits in the preferred range
                        if min_val <= feature_value <= max_val:
                            score += 1.0
                        else:
                            # Penalize based on distance from range
                            if feature_value < min_val:
                                distance = min_val - feature_value
                            else:
                                distance = feature_value - max_val
                            score += max(0, 1.0 - distance)
                        count += 1

            scores[idx] = score / count if count > 0 else 0

        return scores

    def predict_mood_scores(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """Predict mood scores for tracks using trained models"""
        if not self.mood_models or tracks_df.empty:
            return tracks_df

        results_df = tracks_df.copy()

        for mood, model_info in self.mood_models.items():
            model = model_info['model']
            features = model_info['features']

            # Prepare features
            feature_matrix = tracks_df[features].fillna(tracks_df[features].median())

            # Predict scores
            scores = model.predict(feature_matrix)
            results_df[f'{mood}_score'] = scores

        return results_df

    def get_mood_recommendations(self, mood: str, df: pd.DataFrame, n_recommendations: int = 20) -> pd.DataFrame:
        """Get track recommendations for a specific mood"""
        if mood not in self.mood_models or df.empty:
            return pd.DataFrame()

        # Predict mood scores
        scored_df = self.predict_mood_scores(df)

        # Sort by mood score
        mood_col = f'{mood}_score'
        if mood_col in scored_df.columns:
            recommendations = scored_df.nlargest(n_recommendations, mood_col)
        else:
            # Fallback to rule-based filtering
            recommendations = self._rule_based_recommendations(mood, df, n_recommendations)

        return recommendations

    def _rule_based_recommendations(self, mood: str, df: pd.DataFrame, n_recommendations: int) -> pd.DataFrame:
        """Fallback rule-based recommendations"""
        analyzer = MoodAnalyzer()
        mood_preferences = analyzer.mood_mappings.get(mood, {})

        if not mood_preferences:
            return df.sample(min(n_recommendations, len(df)))

        # Filter based on mood preferences
        filtered_df = df.copy()

        for feature, (min_val, max_val) in mood_preferences.items():
            if feature in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[feature] >= min_val)
                    & (filtered_df[feature] <= max_val)
                ]

        return filtered_df.sample(min(n_recommendations, len(filtered_df)))

    def visualize_music_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create visualizations of music analysis"""
        if df.empty:
            return None

        available_features = [f for f in self.audio_features if f in df.columns]
        if not available_features:
            return None

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Feature distribution
        df[available_features].hist(bins=20, ax=axes[0, 0])
        axes[0, 0].set_title('Audio Feature Distributions')

        # Correlation heatmap
        if len(available_features) > 1:
            corr_matrix = df[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, ax=axes[0, 1], cmap='coolwarm')
            axes[0, 1].set_title('Feature Correlations')

        # Valence vs Energy scatter
        if 'valence' in df.columns and 'energy' in df.columns:
            axes[1, 0].scatter(df['valence'], df['energy'], alpha=0.6)
            axes[1, 0].set_xlabel('Valence')
            axes[1, 0].set_ylabel('Energy')
            axes[1, 0].set_title('Valence vs Energy')

        # Cluster visualization (if clusters exist)
        if 'cluster' in df.columns and 'valence' in df.columns and 'energy' in df.columns:
            for cluster in df['cluster'].unique():
                cluster_data = df[df['cluster'] == cluster]
                axes[1, 1].scatter(cluster_data['valence'], cluster_data['energy'],
                                   label=f'Cluster {cluster}', alpha=0.7)
            axes[1, 1].set_xlabel('Valence')
            axes[1, 1].set_ylabel('Energy')
            axes[1, 1].set_title('Clusters in Valence-Energy Space')
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig

    def save_models(self, filepath: str):
        """Save trained models to disk"""
        model_data = {
            'mood_models': self.mood_models,
            'scaler': self.scaler,
            'pca': self.pca,
            'cluster_model': self.cluster_model
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_models(self, filepath: str):
        """Load trained models from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                self.mood_models = model_data.get('mood_models', {})
                self.scaler = model_data.get('scaler', StandardScaler())
                self.pca = model_data.get('pca')
                self.cluster_model = model_data.get('cluster_model')
                return True
        return False
