import json
import os
import pickle
import re
from typing import Dict, List, Tuple

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

from .utils import initialize_openai_client

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except (LookupError, OSError):
    print("⚠️ NLTK data not available. Using fallback sentiment analysis.")
    NLTK_AVAILABLE = False

    # Fallback sentiment analyzer
    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            # Simple rule-based sentiment analysis
            text = text.lower()
            positive_words = [
                'happy',
                'joy',
                'love',
                'great',
                'amazing',
                'wonderful',
                'fantastic',
                'good',
                'excellent',
                'excited']
            negative_words = [
                'sad',
                'bad',
                'terrible',
                'awful',
                'hate',
                'angry',
                'depressed',
                'miserable',
                'horrible',
                'upset']

            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            total_words = len(text.split())
            pos_score = pos_count / max(total_words, 1)
            neg_score = neg_count / max(total_words, 1)

            compound = pos_score - neg_score
            neutral = max(0, 1 - pos_score - neg_score)

            return {
                'pos': pos_score,
                'neg': neg_score,
                'neu': neutral,
                'compound': compound
            }


class MoodAnalyzer:
    def __init__(self):
        """Initialize mood analyzer with pre-defined mood mappings"""
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Initialize OpenAI client if available
        self.openai_client = initialize_openai_client("mood analysis")

        # Define mood-to-audio-features mapping
        self.mood_mappings = {
            'happy': {
                'valence': (0.6, 1.0),
                'energy': (0.5, 1.0),
                'danceability': (0.5, 1.0),
                'tempo': (100, 180),
                'acousticness': (0.0, 0.5),
                'description': 'Upbeat, positive, energetic music'
            },
            'sad': {
                'valence': (0.0, 0.4),
                'energy': (0.0, 0.5),
                'danceability': (0.0, 0.4),
                'tempo': (60, 120),
                'acousticness': (0.3, 1.0),
                'description': 'Melancholic, slow, emotional music'
            },
            'energetic': {
                'valence': (0.5, 1.0),
                'energy': (0.7, 1.0),
                'danceability': (0.6, 1.0),
                'tempo': (120, 200),
                'acousticness': (0.0, 0.3),
                'description': 'High-energy, fast-paced, motivating music'
            },
            'calm': {
                'valence': (0.3, 0.7),
                'energy': (0.0, 0.4),
                'danceability': (0.0, 0.4),
                'tempo': (60, 100),
                'acousticness': (0.4, 1.0),
                'description': 'Peaceful, relaxing, soothing music'
            },
            'angry': {
                'valence': (0.0, 0.3),
                'energy': (0.7, 1.0),
                'danceability': (0.4, 0.8),
                'tempo': (100, 180),
                'acousticness': (0.0, 0.2),
                'description': 'Aggressive, intense, powerful music'
            },
            'romantic': {
                'valence': (0.4, 0.8),
                'energy': (0.2, 0.6),
                'danceability': (0.3, 0.7),
                'tempo': (70, 130),
                'acousticness': (0.2, 0.8),
                'description': 'Loving, intimate, gentle music'
            },
            'nostalgic': {
                'valence': (0.2, 0.6),
                'energy': (0.2, 0.6),
                'danceability': (0.2, 0.6),
                'tempo': (80, 140),
                'acousticness': (0.3, 0.8),
                'description': 'Wistful, reminiscent, emotional music'
            },
            'focused': {
                'valence': (0.3, 0.7),
                'energy': (0.3, 0.7),
                'danceability': (0.2, 0.5),
                'tempo': (90, 130),
                'instrumentalness': (0.5, 1.0),
                'description': 'Instrumental, steady, concentration music'
            },
            'party': {
                'valence': (0.6, 1.0),
                'energy': (0.7, 1.0),
                'danceability': (0.7, 1.0),
                'tempo': (110, 180),
                'acousticness': (0.0, 0.2),
                'description': 'Fun, danceable, celebratory music'
            },
            'melancholic': {
                'valence': (0.0, 0.3),
                'energy': (0.0, 0.4),
                'danceability': (0.0, 0.3),
                'tempo': (60, 100),
                'acousticness': (0.4, 1.0),
                'description': 'Deep sadness, contemplative, introspective music'
            }
        }

        # Keywords associated with each mood
        self.mood_keywords = {
            'happy': ['happy', 'joyful', 'cheerful', 'upbeat', 'positive', 'bright', 'sunny', 'elated', 'thrilled'],
            'sad': ['sad', 'depressed', 'down', 'blue', 'melancholy', 'sorrowful', 'heartbroken', 'gloomy'],
            'energetic': ['energetic', 'pumped', 'hyped', 'excited', 'motivated', 'powerful', 'dynamic', 'intense',
                         'workout', 'exercise'],
            'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'mellow', 'chill', 'zen', 'quiet',
                    'relaxing', 'soothing'],
            'angry': ['angry', 'mad', 'furious', 'frustrated', 'annoyed', 'irritated', 'rage', 'aggressive'],
            'romantic': ['romantic', 'love', 'intimate', 'passionate', 'sweet', 'tender', 'affectionate'],
            'nostalgic': ['nostalgic', 'memories', 'reminiscent', 'wistful', 'throwback', 'past', 'remember'],
            'focused': ['focused', 'concentrated', 'work', 'study', 'productive', 'alert', 'attentive',
                       'concentration', 'studying', 'focus', 'meditation', 'lock', 'concentrate'],
            'party': ['party', 'celebration', 'fun', 'dancing', 'festive', 'lively', 'social', 'clubbing'],
            'melancholic': ['melancholic', 'contemplative', 'pensive', 'brooding', 'reflective', 'introspective']
        }

        self.custom_training_data = []
        self.trained_model = None

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text input"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using VADER and TextBlob"""
        # VADER analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)

        # TextBlob analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment

        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_sentiment.polarity,
            'textblob_subjectivity': textblob_sentiment.subjectivity
        }

    def detect_mood_keywords(self, text: str) -> Dict[str, float]:
        """Detect mood based on keyword matching"""
        text = self.preprocess_text(text)
        words = text.split()

        mood_scores = {}

        for mood, keywords in self.mood_keywords.items():
            score = 0
            matches = 0

            for word in words:
                for keyword in keywords:
                    # More precise matching to avoid false positives
                    if word == keyword:
                        score += 2  # Exact match gets higher score
                        matches += 1
                        break
                    elif len(keyword) >= 4 and (keyword in word or word in keyword):
                        # Only allow partial matches for longer keywords
                        score += 1
                        matches += 1
                        break

            # Normalize by text length but give weight to number of matches
            if len(words) > 0:
                mood_scores[mood] = (score / len(words)) + (matches * 0.1)
            else:
                mood_scores[mood] = 0

        return mood_scores

    def analyze_mood_with_openai(self, text: str) -> Dict:
        """Analyze mood using OpenAI GPT for more accurate detection"""
        if not self.openai_client:
            # Fallback to regular mood detection
            return self.analyze_mood_fallback(text)

        try:
            # Create a prompt for mood detection
            available_moods = list(self.mood_mappings.keys())
            mood_descriptions = {mood: data['description'] for mood, data in self.mood_mappings.items()}

            prompt = f"""
            Analyze the following text and determine the primary mood. Choose from these options:

            Available moods:
            {json.dumps(mood_descriptions, indent=2)}

            Text to analyze: "{text}"

            Respond with a JSON object containing:
            1. "primary_mood": the most fitting mood from the list above
            2. "confidence": a score from 0.0 to 1.0 indicating confidence
            3. "reasoning": brief explanation of why this mood was chosen
            4. "mood_scores": an object with scores (0.0-1.0) for each mood category

            Important guidelines:
            - "focused" is for work, study, concentration tasks
            - "calm" is for relaxation, chill, peaceful states
            - "energetic" is for high-energy, workout, motivation
            - "party" is for dancing, celebration, social fun
            - Consider context clues like "study", "work", "concentrate", "lock in"

            Return only valid JSON.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing emotional states and moods "
                                                 "from text. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )

            # Parse the response
            result_text = response.choices[0].message.content.strip()

            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting
                if '```json' in result_text:
                    result_text = result_text.split('```json')[1].split('```')[0]
                elif '```' in result_text:
                    result_text = result_text.split('```')[1].split('```')[0]

                result = json.loads(result_text)

                # Validate the result
                primary_mood = result.get('primary_mood', 'calm')
                if primary_mood not in available_moods:
                    primary_mood = 'calm'

                confidence = min(max(result.get('confidence', 0.8), 0.0), 1.0)
                reasoning = result.get('reasoning', 'AI analysis')
                mood_scores = result.get('mood_scores', {mood: 0.1 for mood in available_moods})

                # Ensure all moods have scores
                for mood in available_moods:
                    if mood not in mood_scores:
                        mood_scores[mood] = 0.0

                # Get audio preferences
                audio_preferences = self.mood_mappings.get(primary_mood, {})

                # Get sentiment analysis as backup
                sentiment = self.analyze_sentiment(text)

                return {
                    'primary_mood': primary_mood,
                    'mood_confidence': confidence,
                    'sentiment_scores': sentiment,
                    'mood_scores': mood_scores,
                    'audio_preferences': audio_preferences,
                    'processed_text': self.preprocess_text(text),
                    'reasoning': reasoning,
                    'method': 'openai'
                }

            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse OpenAI response: {e}")
                print(f"Raw response: {result_text}")
                return self.analyze_mood_fallback(text)

        except Exception as e:
            print(f"⚠️ OpenAI API error: {e}")
            return self.analyze_mood_fallback(text)

    def analyze_mood_fallback(self, text: str) -> Dict:
        """Fallback mood analysis using the original method"""
        text = self.preprocess_text(text)

        # Get sentiment analysis
        sentiment = self.analyze_sentiment(text)

        # Get keyword-based mood detection
        keyword_moods = self.detect_mood_keywords(text)

        # Combine sentiment and keywords to determine primary mood
        primary_mood = self._determine_primary_mood(sentiment, keyword_moods, text)

        # Get audio feature preferences for the detected mood
        audio_preferences = self.mood_mappings.get(primary_mood, {})

        return {
            'primary_mood': primary_mood,
            'mood_confidence': max(keyword_moods.values()) if keyword_moods else 0,
            'sentiment_scores': sentiment,
            'mood_scores': keyword_moods,
            'audio_preferences': audio_preferences,
            'processed_text': text,
            'method': 'fallback'
        }

    def analyze_mood(self, text: str) -> Dict:
        """Main mood analysis function - uses OpenAI if available, fallback otherwise"""
        if self.openai_client:
            return self.analyze_mood_with_openai(text)
        else:
            return self.analyze_mood_fallback(text)

    def _determine_primary_mood(self, sentiment: Dict, keyword_moods: Dict, text: str) -> str:
        """Determine primary mood from sentiment and keyword analysis"""
        text_lower = text.lower()

        # Direct keyword priority checks - highest priority
        if 'sad' in text_lower:
            return 'sad'
        if 'angry' in text_lower or 'mad' in text_lower:
            return 'angry'
        if 'romantic' in text_lower or 'love' in text_lower:
            return 'romantic'
        if 'nostalgic' in text_lower:
            return 'nostalgic'

        # Check for energetic terms FIRST (before focus terms)
        if any(term in text_lower for term in ['energetic', 'excited', 'pumped', 'hyped', 'workout', 'exercise']):
            return 'energetic'

        # Check for party terms
        if any(term in text_lower for term in ['party', 'dance', 'dancing', 'celebration', 'fun', 'clubbing']):
            return 'party'

        # Check for focus/concentration terms (after energetic check)
        if any(
            term in text_lower for term in [
                'lock',
                'concentrate',
                'focus',
                'study',
                'studying',
                'work',
                'concentration']):
            # If it mentions chill/relax with study, it's calm studying
            if any(term in text_lower for term in ['chill', 'relax', 'calm', 'peaceful']):
                return 'calm'
            else:
                return 'focused'

        # Check for chill/relaxing terms
        if any(term in text_lower for term in ['chill', 'relax', 'calm', 'peaceful', 'tranquil', 'zen']):
            return 'calm'

        # Check if any keywords were detected
        if keyword_moods:
            max_keyword_mood = max(keyword_moods.items(), key=lambda x: x[1])
            if max_keyword_mood[1] > 0.05:  # Lower threshold for better detection
                return max_keyword_mood[0]

        # Fall back to sentiment-based mood detection
        compound = sentiment['vader_compound']

        if compound >= 0.5:
            return 'happy'
        elif compound <= -0.5:
            return 'sad'
        elif compound <= -0.1:
            return 'melancholic'
        else:
            return 'calm'  # Default to calm for neutral sentiment

    def add_training_data(self, text: str, mood: str, feedback_score: float = 1.0):
        """Add custom training data for mood detection"""
        self.custom_training_data.append({
            'text': self.preprocess_text(text),
            'mood': mood,
            'feedback_score': feedback_score
        })

    def train_custom_model(self):
        """Train a custom model based on user feedback"""
        if len(self.custom_training_data) < 10:
            return False  # Need at least 10 examples

        # Prepare training data
        texts = [item['text'] for item in self.custom_training_data]
        moods = [item['mood'] for item in self.custom_training_data]

        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        vectorizer.fit_transform(texts)

        # Store the trained model
        self.trained_model = {
            'vectorizer': vectorizer,
            'training_texts': texts,
            'training_moods': moods,
            'training_data': self.custom_training_data
        }

        return True

    def predict_with_custom_model(self, text: str) -> Tuple[str, float]:
        """Predict mood using custom trained model"""
        if not self.trained_model:
            return None, 0.0

        processed_text = self.preprocess_text(text)

        # Transform input text
        vectorizer = self.trained_model['vectorizer']
        text_vector = vectorizer.transform([processed_text])

        # Calculate similarity with training examples
        training_vectors = vectorizer.transform(self.trained_model['training_texts'])
        similarities = cosine_similarity(text_vector, training_vectors)[0]

        # Get the most similar example
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]

        if best_similarity > 0.3:  # Threshold for considering a match
            predicted_mood = self.trained_model['training_moods'][best_match_idx]
            return predicted_mood, best_similarity

        return None, 0.0

    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if self.trained_model:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.trained_model,
                    'training_data': self.custom_training_data
                }, f)

    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.trained_model = data['model']
                self.custom_training_data = data['training_data']
                return True
        return False

    def get_mood_description(self, mood: str) -> str:
        """Get description for a mood"""
        return self.mood_mappings.get(mood, {}).get('description', 'No description available')

    def get_available_moods(self) -> List[str]:
        """Get list of all available moods"""
        return list(self.mood_mappings.keys())
