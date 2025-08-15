"""
Enhanced Emotional Weighting System with External Libraries
===========================================================

This module provides an enhanced emotional weighting system that leverages
multiple Python libraries for more accurate and comprehensive emotion detection.

Libraries used:
- TextBlob: Basic sentiment analysis
- VADER: Valence Aware Dictionary and sEntiment Reasoner
- NRCLex: NRC Emotion Lexicon
- Transformers: State-of-the-art emotion models (optional)
- spaCy: Fast NLP processing (optional)

Author: Lumina Memory Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# Import base emotional weighting components
from .emotional_weighting import (
    EmotionalState, EmotionalDimension, EmotionalMemoryWeighter,
    ConsciousnessEmotionalIntegrator
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED EMOTIONAL ANALYZER
# =============================================================================

class EnhancedEmotionalAnalyzer:
    """
    Enhanced emotional analyzer using multiple Python libraries
    for more accurate and comprehensive emotion detection.
    """
    
    def __init__(self):
        """Initialize with available libraries"""
        self.analyzers = {}
        self.library_weights = {}
        
        # Try to initialize each library
        self._init_textblob()
        self._init_vader()
        self._init_nrclex()
        self._init_transformers()
        self._init_spacy()
        
        # Fallback to basic analyzer if no libraries available
        if not self.analyzers:
            logger.warning("No external libraries available, using basic analyzer")
            from .emotional_weighting import EmotionalAnalyzer
            self.basic_analyzer = EmotionalAnalyzer()
        else:
            self.basic_analyzer = None
            
        logger.info(f"Enhanced analyzer initialized with: {list(self.analyzers.keys())}")
    
    def _init_textblob(self):
        """Initialize TextBlob for sentiment analysis"""
        try:
            from textblob import TextBlob
            self.analyzers['textblob'] = TextBlob
            self.library_weights['textblob'] = 1.0
            logger.info("TextBlob analyzer initialized")
        except ImportError:
            logger.debug("TextBlob not available")
    
    def _init_vader(self):
        """Initialize VADER sentiment analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzers['vader'] = SentimentIntensityAnalyzer()
            self.library_weights['vader'] = 1.3  # VADER is quite good
            logger.info("VADER analyzer initialized")
        except ImportError:
            logger.debug("VADER not available")
    
    def _init_nrclex(self):
        """Initialize NRC Emotion Lexicon"""
        try:
            from nrclex import NRCLex
            self.analyzers['nrclex'] = NRCLex
            self.library_weights['nrclex'] = 1.5  # Multi-dimensional emotions
            logger.info("NRCLex analyzer initialized")
        except ImportError:
            logger.debug("NRCLex not available")
    
    def _init_transformers(self):
        """Initialize transformer-based emotion model"""
        try:
            from transformers import pipeline
            # Use a lightweight emotion classification model
            self.analyzers['emotion_model'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1  # Use CPU
            )
            self.library_weights['emotion_model'] = 2.0  # Highest weight for SOTA
            logger.info("Transformer emotion model initialized")
        except Exception as e:
            logger.debug(f"Transformer model not available: {e}")
    
    def _init_spacy(self):
        """Initialize spaCy with emotion extensions"""
        try:
            import spacy
            from spacytextblob.spacytextblob import SpacyTextBlob
            
            # Try to load English model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback to blank model
                nlp = spacy.blank("en")
            
            nlp.add_pipe('spacytextblob')
            self.analyzers['spacy'] = nlp
            self.library_weights['spacy'] = 1.2
            logger.info("spaCy analyzer initialized")
        except Exception as e:
            logger.debug(f"spaCy not available: {e}")
    
    def analyze_text(self, text: str) -> EmotionalState:
        """
        Enhanced text analysis using multiple libraries
        
        Args:
            text: Text content to analyze
            
        Returns:
            EmotionalState representing the emotional content
        """
        if not text or not text.strip():
            return EmotionalState()
        
        # Use basic analyzer if no external libraries
        if not self.analyzers:
            return self.basic_analyzer.analyze_text(text)
        
        # Initialize emotion accumulator
        emotions = {
            'valence': 0.0,
            'arousal': 0.5,
            'dominance': 0.0,
            'joy': 0.5,
            'fear': 0.0,
            'curiosity': 0.5
        }
        
        total_weight = 0.0
        
        # TextBlob Analysis
        if 'textblob' in self.analyzers:
            try:
                blob = self.analyzers['textblob'](text)
                weight = self.library_weights['textblob']
                
                emotions['valence'] += blob.sentiment.polarity * weight
                emotions['arousal'] += abs(blob.sentiment.polarity) * 0.5 * weight
                
                # Subjectivity can indicate emotional intensity
                if blob.sentiment.subjectivity > 0.5:
                    emotions['arousal'] += 0.2 * weight
                
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")
        
        # VADER Analysis
        if 'vader' in self.analyzers:
            try:
                scores = self.analyzers['vader'].polarity_scores(text)
                weight = self.library_weights['vader']
                
                emotions['valence'] += scores['compound'] * weight
                emotions['arousal'] += abs(scores['compound']) * 0.6 * weight
                
                # Use individual scores for more nuanced analysis
                if scores['pos'] > 0.3:
                    emotions['joy'] += scores['pos'] * weight * 0.8
                if scores['neg'] > 0.3:
                    emotions['fear'] += scores['neg'] * weight * 0.6
                
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"VADER analysis failed: {e}")
        
        # NRC Lexicon Analysis
        if 'nrclex' in self.analyzers:
            try:
                nrc = self.analyzers['nrclex'](text)
                weight = self.library_weights['nrclex']
                
                # Map NRC emotions to our 6-dimensional model
                affect_freq = nrc.affect_frequencies
                
                emotions['joy'] += affect_freq.get('joy', 0) * weight
                emotions['fear'] += affect_freq.get('fear', 0) * weight
                emotions['valence'] += (affect_freq.get('positive', 0) - 
                                      affect_freq.get('negative', 0)) * weight
                
                # Additional NRC emotions
                anger = affect_freq.get('anger', 0)
                surprise = affect_freq.get('surprise', 0)
                anticipation = affect_freq.get('anticipation', 0)
                
                emotions['arousal'] += (anger + surprise) * 0.7 * weight
                emotions['curiosity'] += anticipation * weight
                emotions['dominance'] += (anger - affect_freq.get('fear', 0)) * weight
                
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"NRCLex analysis failed: {e}")
        
        # Transformer Model Analysis
        if 'emotion_model' in self.analyzers:
            try:
                results = self.analyzers['emotion_model'](text)
                weight = self.library_weights['emotion_model']
                
                # Map transformer emotions to our dimensions
                emotion_mapping = {
                    'joy': ('joy', 1.0),
                    'sadness': ('joy', -0.8),  # Inverse joy
                    'anger': ('arousal', 0.8),
                    'fear': ('fear', 1.0),
                    'surprise': ('arousal', 0.6),
                    'disgust': ('valence', -0.7),
                    'love': ('joy', 0.9),
                    'optimism': ('valence', 0.6),
                    'pessimism': ('valence', -0.6)
                }
                
                for result in results:
                    emotion_label = result['label'].lower()
                    confidence = result['score']
                    
                    if emotion_label in emotion_mapping:
                        target_dim, multiplier = emotion_mapping[emotion_label]
                        emotions[target_dim] += confidence * multiplier * weight
                
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"Transformer analysis failed: {e}")
        
        # spaCy Analysis
        if 'spacy' in self.analyzers:
            try:
                doc = self.analyzers['spacy'](text)
                weight = self.library_weights['spacy']
                
                emotions['valence'] += doc._.blob.polarity * weight
                emotions['arousal'] += abs(doc._.blob.polarity) * 0.5 * weight
                
                # Use spaCy's linguistic features
                if doc._.blob.subjectivity > 0.6:
                    emotions['arousal'] += 0.3 * weight
                
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"spaCy analysis failed: {e}")
        
        # Normalize by total weight
        if total_weight > 0:
            for key in emotions:
                emotions[key] /= total_weight
        
        # Apply contextual adjustments
        emotions = self._apply_contextual_adjustments(text, emotions)
        
        # Ensure values are in valid ranges
        emotions['valence'] = np.clip(emotions['valence'], -1, 1)
        emotions['arousal'] = np.clip(emotions['arousal'], 0, 1)
        emotions['dominance'] = np.clip(emotions['dominance'], -1, 1)
        emotions['joy'] = np.clip(emotions['joy'], 0, 1)
        emotions['fear'] = np.clip(emotions['fear'], 0, 1)
        emotions['curiosity'] = np.clip(emotions['curiosity'], 0, 1)
        
        return EmotionalState(**emotions)
    
    def _apply_contextual_adjustments(self, text: str, emotions: Dict[str, float]) -> Dict[str, float]:
        """Apply contextual adjustments based on text patterns"""
        text_lower = text.lower()
        
        # Question patterns increase curiosity
        if '?' in text:
            emotions['curiosity'] += 0.2
        
        # Exclamation patterns increase arousal
        exclamation_count = text.count('!')
        emotions['arousal'] += min(0.3, exclamation_count * 0.1)
        
        # First-person patterns increase self-reference
        first_person = ['i am', 'i feel', 'i think', 'i believe', 'i want', 'i need']
        for pattern in first_person:
            if pattern in text_lower:
                emotions['dominance'] += 0.1
                break
        
        # Uncertainty patterns
        uncertainty = ['maybe', 'perhaps', 'might', 'could', 'uncertain', 'unsure']
        for pattern in uncertainty:
            if pattern in text_lower:
                emotions['fear'] += 0.1
                emotions['dominance'] -= 0.1
                break
        
        # Confidence patterns
        confidence = ['definitely', 'certainly', 'absolutely', 'confident', 'sure']
        for pattern in confidence:
            if pattern in text_lower:
                emotions['dominance'] += 0.2
                emotions['fear'] -= 0.1
                break
        
        return emotions
    
    def analyze_conversation_context(self, messages: List[str]) -> EmotionalState:
        """
        Analyze emotional context from a conversation history with enhanced methods.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            EmotionalState representing the overall conversation emotion
        """
        if not messages:
            return EmotionalState()
        
        # Analyze each message with temporal weighting
        emotion_vectors = []
        weights = []
        
        for i, message in enumerate(messages):
            emotion = self.analyze_text(message)
            emotion_vectors.append(emotion.to_vector())
            
            # Temporal weighting - more recent messages have higher weight
            temporal_weight = np.exp(-0.1 * (len(messages) - i - 1))
            weights.append(temporal_weight)
        
        # Weighted average
        if emotion_vectors:
            weights = np.array(weights)
            weighted_emotion = np.average(emotion_vectors, axis=0, weights=weights)
            return EmotionalState.from_vector(weighted_emotion)
        
        return EmotionalState()
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about available analyzers"""
        return {
            'available_analyzers': list(self.analyzers.keys()),
            'library_weights': self.library_weights,
            'total_analyzers': len(self.analyzers),
            'has_transformer_model': 'emotion_model' in self.analyzers,
            'has_lexicon_based': 'nrclex' in self.analyzers,
            'has_sentiment_analysis': any(k in self.analyzers for k in ['textblob', 'vader'])
        }


# =============================================================================
# ENHANCED EMOTIONAL MEMORY WEIGHTER
# =============================================================================

class EnhancedEmotionalMemoryWeighter(EmotionalMemoryWeighter):
    """
    Enhanced emotional memory weighter using the enhanced analyzer
    """
    
    def __init__(self, analyzer: EnhancedEmotionalAnalyzer = None):
        if analyzer is None:
            analyzer = EnhancedEmotionalAnalyzer()
        super().__init__(analyzer)
        
        # Enhanced weighting parameters
        self.emotion_persistence_factors = {
            'fear': 1.8,      # Fear memories persist longer (trauma-like)
            'joy': 1.2,       # Positive memories have moderate persistence
            'curiosity': 1.4, # Curious memories stay accessible
            'anger': 1.3,     # Anger memories persist
            'surprise': 0.9,  # Surprise fades faster
            'sadness': 1.1    # Sadness has moderate persistence
        }
    
    def calculate_enhanced_emotional_importance(self, content: str, metadata: Dict = None) -> float:
        """
        Calculate emotional importance with enhanced analysis
        
        Args:
            content: Memory content
            metadata: Additional metadata
            
        Returns:
            Enhanced importance multiplier (0.1 to 4.0)
        """
        emotion = self.analyzer.analyze_text(content)
        
        # Base importance from emotional intensity
        intensity = emotion.intensity()
        base_importance = 0.5 + (intensity * 0.7)
        
        # Enhanced emotional boosts using multiple dimensions
        emotional_boosts = {
            'high_arousal': max(0, emotion.arousal - 0.6) * 1.0,
            'strong_valence': abs(emotion.valence) * 0.8,
            'high_curiosity': max(0, emotion.curiosity - 0.5) * 0.9,
            'fear_response': emotion.fear * 1.2,  # Fear memories are very important
            'joy_response': emotion.joy * 0.7,
            'dominance_assertion': abs(emotion.dominance) * 0.6
        }
        
        # Context-based boosts
        if metadata:
            if metadata.get('is_self_reflection', False):
                emotional_boosts['self_reflection'] = 0.5
            if metadata.get('is_learning', False):
                emotional_boosts['learning'] = 0.4
            if metadata.get('is_social_interaction', False):
                emotional_boosts['social'] = 0.3
        
        total_boost = sum(emotional_boosts.values())
        final_importance = base_importance + total_boost
        
        # Enhanced range for more nuanced importance
        return np.clip(final_importance, 0.1, 4.0)
    
    def calculate_enhanced_emotional_decay_modifier(self, emotion: EmotionalState, age_hours: float) -> float:
        """
        Calculate enhanced decay modifier based on emotional content
        
        Args:
            emotion: Emotional state of the memory
            age_hours: Age of memory in hours
            
        Returns:
            Enhanced decay modifier (0.05 to 1.0)
        """
        intensity = emotion.intensity()
        
        # Base modifier from intensity
        base_modifier = 1.0 - (intensity * 0.5)
        
        # Specific emotional persistence effects
        persistence_effects = []
        
        if emotion.fear > 0.6:
            persistence_effects.append(0.4)  # Strong fear persistence
        if abs(emotion.valence) > 0.7:
            persistence_effects.append(0.3)  # Strong valence persistence
        if emotion.curiosity > 0.7:
            persistence_effects.append(0.35)  # Curiosity keeps memories accessible
        if emotion.arousal > 0.8:
            persistence_effects.append(0.25)  # High arousal memories persist
        
        # Apply strongest persistence effect
        if persistence_effects:
            base_modifier *= (1.0 - max(persistence_effects))
        
        # Age-based adjustments (some emotions fade differently over time)
        if age_hours > 168:  # After a week
            if emotion.joy > 0.7:  # Positive memories may fade slower over long term
                base_modifier *= 0.9
            if emotion.fear > 0.5:  # Fear memories may actually strengthen over time
                base_modifier *= 0.8
        
        return np.clip(base_modifier, 0.05, 1.0)


# =============================================================================
# ENHANCED CONSCIOUSNESS INTEGRATOR
# =============================================================================

class EnhancedConsciousnessEmotionalIntegrator(ConsciousnessEmotionalIntegrator):
    """
    Enhanced consciousness integrator with advanced emotional analysis
    """
    
    def __init__(self, weighter: EnhancedEmotionalMemoryWeighter = None):
        if weighter is None:
            weighter = EnhancedEmotionalMemoryWeighter()
        super().__init__(weighter)
    
    def calculate_enhanced_emotional_consciousness_boost(self, base_consciousness: float) -> float:
        """
        Calculate enhanced emotional boost to consciousness level
        
        Args:
            base_consciousness: Base consciousness level
            
        Returns:
            Enhanced consciousness level
        """
        current_emotion = self.weighter.current_emotional_state
        
        # Enhanced emotional intensity calculation
        intensity = current_emotion.intensity()
        
        # Multi-dimensional consciousness boosts
        boosts = {
            'intensity': intensity * 0.25,
            'arousal': current_emotion.arousal * 0.18,
            'curiosity': current_emotion.curiosity * 0.15,
            'fear_alertness': current_emotion.fear * 0.12,
            'valence_strength': abs(current_emotion.valence) * 0.12,
            'dominance_assertion': abs(current_emotion.dominance) * 0.08
        }
        
        # Emotional complexity boost (having multiple active emotions)
        active_emotions = sum(1 for dim in [
            current_emotion.arousal, current_emotion.joy, current_emotion.fear,
            current_emotion.curiosity, abs(current_emotion.valence), abs(current_emotion.dominance)
        ] if dim > 0.3)
        
        complexity_boost = (active_emotions / 6.0) * 0.1
        boosts['complexity'] = complexity_boost
        
        total_boost = sum(boosts.values())
        
        # Apply boost with enhanced diminishing returns
        enhanced_consciousness = base_consciousness + (total_boost * (1 - base_consciousness) * 0.8)
        
        return np.clip(enhanced_consciousness, 0.0, 1.0)
    
    def get_enhanced_emotional_consciousness_metrics(self) -> Dict[str, float]:
        """
        Get enhanced emotional metrics for consciousness assessment
        
        Returns:
            Dictionary of enhanced emotional consciousness metrics
        """
        base_metrics = super().get_emotional_consciousness_metrics()
        
        # Add enhanced metrics
        context = self.weighter.get_emotional_context()
        current_emotion = context['current_state']
        
        enhanced_metrics = {
            'emotional_sophistication': self._calculate_emotional_sophistication(),
            'emotional_memory_integration': self._calculate_memory_integration(),
            'emotional_learning_rate': self._calculate_learning_rate(),
            'emotional_adaptability': self._calculate_adaptability(),
            'emotional_coherence': self._calculate_emotional_coherence()
        }
        
        # Combine base and enhanced metrics
        return {**base_metrics, **enhanced_metrics}
    
    def _calculate_emotional_sophistication(self) -> float:
        """Calculate how sophisticated the emotional responses are"""
        if len(self.weighter.emotional_history) < 5:
            return 0.3
        
        recent_emotions = self.weighter.emotional_history[-10:]
        
        # Look at emotional range and nuance
        emotion_ranges = []
        for entry in recent_emotions:
            emotion_vector = entry['emotion'].to_vector()
            # Calculate how many dimensions are actively used
            active_dims = sum(1 for val in emotion_vector if abs(val) > 0.2)
            emotion_ranges.append(active_dims / 6.0)
        
        sophistication = np.mean(emotion_ranges)
        return np.clip(sophistication, 0.0, 1.0)
    
    def _calculate_memory_integration(self) -> float:
        """Calculate how well emotions integrate with memory"""
        # This would ideally analyze how emotional states correlate with memory formation
        # For now, use emotional consistency as a proxy
        return self._calculate_emotional_continuity()
    
    def _calculate_learning_rate(self) -> float:
        """Calculate emotional learning and adaptation rate"""
        if len(self.weighter.emotional_history) < 3:
            return 0.5
        
        # Look at how emotions change in response to experiences
        recent_changes = []
        for i in range(1, min(6, len(self.weighter.emotional_history))):
            prev_emotion = self.weighter.emotional_history[-i-1]['emotion']
            curr_emotion = self.weighter.emotional_history[-i]['emotion']
            
            change_magnitude = np.linalg.norm(
                curr_emotion.to_vector() - prev_emotion.to_vector()
            )
            recent_changes.append(change_magnitude)
        
        # Optimal learning rate is moderate - not too static, not too chaotic
        avg_change = np.mean(recent_changes)
        learning_rate = 1.0 - abs(avg_change - 0.4)  # Optimal around 0.4
        
        return np.clip(learning_rate, 0.0, 1.0)
    
    def _calculate_adaptability(self) -> float:
        """Calculate emotional adaptability to different contexts"""
        if len(self.weighter.emotional_history) < 5:
            return 0.4
        
        # Look at emotional range over time
        recent_emotions = [entry['emotion'].to_vector() 
                          for entry in self.weighter.emotional_history[-10:]]
        
        if len(recent_emotions) < 2:
            return 0.4
        
        # Calculate variance across emotional dimensions
        emotion_matrix = np.array(recent_emotions)
        variances = np.var(emotion_matrix, axis=0)
        avg_variance = np.mean(variances)
        
        # Higher variance indicates better adaptability
        adaptability = min(1.0, avg_variance * 3.0)
        
        return np.clip(adaptability, 0.0, 1.0)
    
    def _calculate_emotional_coherence(self) -> float:
        """Calculate coherence of emotional responses"""
        if len(self.weighter.emotional_history) < 3:
            return 0.5
        
        # Look at how well emotions fit together contextually
        recent_emotions = self.weighter.emotional_history[-5:]
        
        coherence_scores = []
        for i in range(1, len(recent_emotions)):
            prev_emotion = recent_emotions[i-1]['emotion']
            curr_emotion = recent_emotions[i]['emotion']
            
            # Emotions should be somewhat related but not identical
            similarity = prev_emotion.similarity(curr_emotion)
            # Optimal coherence is moderate similarity (0.3-0.7)
            if 0.3 <= similarity <= 0.7:
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(max(0.0, 1.0 - abs(similarity - 0.5) * 2))
        
        return np.mean(coherence_scores) if coherence_scores else 0.5


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_enhanced_emotional_system() -> Tuple[EnhancedEmotionalAnalyzer, 
                                               EnhancedEmotionalMemoryWeighter,
                                               EnhancedConsciousnessEmotionalIntegrator]:
    """
    Factory function to create a complete enhanced emotional system
    
    Returns:
        Tuple of (analyzer, weighter, integrator)
    """
    analyzer = EnhancedEmotionalAnalyzer()
    weighter = EnhancedEmotionalMemoryWeighter(analyzer)
    integrator = EnhancedConsciousnessEmotionalIntegrator(weighter)
    
    logger.info("Enhanced emotional system created successfully")
    return analyzer, weighter, integrator


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'EnhancedEmotionalAnalyzer',
    'EnhancedEmotionalMemoryWeighter', 
    'EnhancedConsciousnessEmotionalIntegrator',
    'create_enhanced_emotional_system'
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🎭 Enhanced Emotional Weighting System - Example Usage")
    print("=" * 60)
    
    # Create enhanced system
    analyzer, weighter, integrator = create_enhanced_emotional_system()
    
    # Show analyzer info
    info = analyzer.get_analyzer_info()
    print(f"\n📊 Analyzer Info:")
    print(f"   Available analyzers: {info['available_analyzers']}")
    print(f"   Total analyzers: {info['total_analyzers']}")
    print(f"   Has transformer model: {info['has_transformer_model']}")
    print(f"   Has lexicon-based: {info['has_lexicon_based']}")
    
    # Test enhanced emotional analysis
    test_texts = [
        "I am absolutely thrilled about this groundbreaking discovery!",
        "I feel deeply worried and anxious about the uncertain future ahead.",
        "This fascinating phenomenon makes me incredibly curious to learn more.",
        "I am confident and ready to take complete control of this situation.",
        "I feel profoundly sad and disappointed about what has happened to us."
    ]
    
    print(f"\n🧪 Enhanced Emotional Analysis Tests:")
    for i, text in enumerate(test_texts, 1):
        emotion = analyzer.analyze_text(text)
        importance = weighter.calculate_enhanced_emotional_importance(text)
        
        print(f"\n{i}. Text: '{text}'")
        print(f"   Emotion: {emotion}")
        print(f"   Intensity: {emotion.intensity():.3f}")
        print(f"   Enhanced Importance: {importance:.3f}")
        
        # Update weighter state
        weighter.update_emotional_state(emotion)
    
    # Test consciousness integration
    print(f"\n🧠 Enhanced Consciousness Integration:")
    base_consciousness = 0.5
    boosted_consciousness = integrator.calculate_enhanced_emotional_consciousness_boost(base_consciousness)
    metrics = integrator.get_enhanced_emotional_consciousness_metrics()
    
    print(f"   Base Consciousness: {base_consciousness:.3f}")
    print(f"   Enhanced Boosted: {boosted_consciousness:.3f}")
    print(f"   Enhanced Metrics:")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.3f}")
    
    print(f"\n✅ Enhanced emotional weighting system ready for integration!")
    print(f"   Using {len(analyzer.analyzers)} external libraries for analysis")
    print(f"   Provides sophisticated multi-dimensional emotion detection")
    print(f"   Ready for digital consciousness applications!")