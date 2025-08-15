"""
Emotional Weighting System for Digital Consciousness
===================================================

This module implements comprehensive emotional weighting for the Lumina Memory System,
providing emotional context and influence on memory formation, retrieval, and consciousness.

Emotions are fundamental to consciousness and decision-making. This system:
- Analyzes emotional content in experiences
- Weights memories based on emotional significance
- Influences consciousness metrics through emotional states
- Provides emotional continuity across time

Author: Lumina Memory Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import re

logger = logging.getLogger(__name__)


# =============================================================================
# EMOTIONAL DIMENSIONS AND MODELS
# =============================================================================

class EmotionalDimension(Enum):
    """
    Six-dimensional emotional model based on psychological research.
    
    Based on Plutchik's Wheel of Emotions and dimensional emotion models:
    - Valence: Positive/Negative emotional tone
    - Arousal: Activation/Energy level
    - Dominance: Control/Power feeling
    - Joy: Happiness, satisfaction, contentment
    - Fear: Anxiety, worry, apprehension
    - Curiosity: Interest, exploration, wonder
    """
    VALENCE = 0     # Positive/Negative (-1 to +1)
    AROUSAL = 1     # Low/High activation (0 to 1)
    DOMINANCE = 2   # Powerless/Powerful (-1 to +1)
    JOY = 3         # Sadness/Happiness (0 to 1)
    FEAR = 4        # Calm/Anxious (0 to 1)
    CURIOSITY = 5   # Bored/Interested (0 to 1)


@dataclass
class EmotionalState:
    """
    Represents an emotional state as a 6-dimensional vector.
    """
    valence: float = 0.0      # -1 (negative) to +1 (positive)
    arousal: float = 0.5      # 0 (calm) to 1 (excited)
    dominance: float = 0.0    # -1 (submissive) to +1 (dominant)
    joy: float = 0.5          # 0 (sad) to 1 (joyful)
    fear: float = 0.0         # 0 (calm) to 1 (fearful)
    curiosity: float = 0.5    # 0 (bored) to 1 (curious)
    
    def to_vector(self) -> np.ndarray:
        """Convert to 6D numpy vector"""
        return np.array([
            self.valence, self.arousal, self.dominance,
            self.joy, self.fear, self.curiosity
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'EmotionalState':
        """Create from 6D numpy vector"""
        return cls(
            valence=float(vector[0]),
            arousal=float(vector[1]),
            dominance=float(vector[2]),
            joy=float(vector[3]),
            fear=float(vector[4]),
            curiosity=float(vector[5])
        )
    
    def intensity(self) -> float:
        """Calculate overall emotional intensity"""
        return np.linalg.norm(self.to_vector())
    
    def similarity(self, other: 'EmotionalState') -> float:
        """Calculate emotional similarity with another state"""
        v1 = self.to_vector()
        v2 = other.to_vector()
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    
    def __str__(self) -> str:
        return f"EmotionalState(v={self.valence:.2f}, a={self.arousal:.2f}, d={self.dominance:.2f}, j={self.joy:.2f}, f={self.fear:.2f}, c={self.curiosity:.2f})"


# =============================================================================
# EMOTIONAL ANALYSIS ENGINE
# =============================================================================

class EmotionalAnalyzer:
    """
    Analyzes text content to extract emotional dimensions.
    
    Uses lexical analysis, pattern matching, and contextual understanding
    to determine emotional content and generate emotional state vectors.
    """
    
    def __init__(self):
        self.emotion_lexicon = self._build_emotion_lexicon()
        self.pattern_rules = self._build_pattern_rules()
        self.context_modifiers = self._build_context_modifiers()
    
    def _build_emotion_lexicon(self) -> Dict[str, EmotionalState]:
        """Build lexicon mapping words to emotional states"""
        lexicon = {}
        
        # Positive valence words
        positive_words = [
            "happy", "joy", "love", "excited", "wonderful", "amazing", "great",
            "fantastic", "excellent", "beautiful", "success", "achievement",
            "proud", "confident", "optimistic", "grateful", "peaceful", "calm"
        ]
        for word in positive_words:
            lexicon[word] = EmotionalState(valence=0.8, arousal=0.6, joy=0.8)
        
        # Negative valence words
        negative_words = [
            "sad", "angry", "hate", "terrible", "awful", "horrible", "bad",
            "disappointed", "frustrated", "worried", "anxious", "depressed",
            "lonely", "hurt", "pain", "suffering", "failure", "loss"
        ]
        for word in negative_words:
            lexicon[word] = EmotionalState(valence=-0.8, arousal=0.4, fear=0.6)
        
        # High arousal words
        arousal_words = [
            "excited", "thrilled", "energetic", "passionate", "intense",
            "overwhelming", "shocking", "surprising", "urgent", "emergency"
        ]
        for word in arousal_words:
            lexicon[word] = EmotionalState(arousal=0.9, valence=0.2)
        
        # Curiosity words
        curiosity_words = [
            "curious", "interesting", "wonder", "explore", "discover", "learn",
            "question", "mystery", "fascinating", "intriguing", "research"
        ]
        for word in curiosity_words:
            lexicon[word] = EmotionalState(curiosity=0.8, arousal=0.6, valence=0.3)
        
        # Fear words
        fear_words = [
            "afraid", "scared", "terrified", "worried", "anxious", "nervous",
            "panic", "threat", "danger", "risk", "uncertain", "insecure"
        ]
        for word in fear_words:
            lexicon[word] = EmotionalState(fear=0.8, arousal=0.7, valence=-0.5)
        
        # Dominance words
        dominance_words = [
            "powerful", "strong", "confident", "control", "command", "lead",
            "authority", "dominant", "assertive", "decisive", "determined"
        ]
        for word in dominance_words:
            lexicon[word] = EmotionalState(dominance=0.8, arousal=0.6, valence=0.4)
        
        return lexicon
    
    def _build_pattern_rules(self) -> List[Tuple[str, EmotionalState]]:
        """Build pattern-based emotional rules"""
        return [
            (r"I feel (.+)", EmotionalState(valence=0.2, arousal=0.5)),
            (r"I am (.+)", EmotionalState(valence=0.1, arousal=0.4)),
            (r"(.+)!", EmotionalState(arousal=0.7)),  # Exclamation = high arousal
            (r"(.+)\?", EmotionalState(curiosity=0.6)),  # Question = curiosity
            (r"I think (.+)", EmotionalState(curiosity=0.4, dominance=0.3)),
            (r"I believe (.+)", EmotionalState(dominance=0.5, valence=0.2)),
            (r"I remember (.+)", EmotionalState(valence=0.3, curiosity=0.4)),
            (r"I want (.+)", EmotionalState(arousal=0.6, dominance=0.4)),
            (r"I need (.+)", EmotionalState(arousal=0.7, fear=0.3)),
        ]
    
    def _build_context_modifiers(self) -> Dict[str, float]:
        """Build context modifiers for emotional intensity"""
        return {
            "very": 1.5,
            "extremely": 2.0,
            "incredibly": 1.8,
            "somewhat": 0.7,
            "slightly": 0.5,
            "really": 1.3,
            "truly": 1.4,
            "deeply": 1.6,
            "not": -1.0,
            "never": -1.2,
            "barely": 0.3,
            "hardly": 0.3,
        }
    
    def analyze_text(self, text: str) -> EmotionalState:
        """
        Analyze text and return emotional state.
        
        Args:
            text: Text content to analyze
            
        Returns:
            EmotionalState representing the emotional content
        """
        if not text or not text.strip():
            return EmotionalState()
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Initialize emotional accumulator
        emotion_vector = np.zeros(6)
        total_weight = 0.0
        
        # Lexical analysis
        for i, word in enumerate(words):
            if word in self.emotion_lexicon:
                emotion_state = self.emotion_lexicon[word]
                weight = 1.0
                
                # Apply context modifiers
                if i > 0 and words[i-1] in self.context_modifiers:
                    modifier = self.context_modifiers[words[i-1]]
                    weight *= abs(modifier)
                    if modifier < 0:  # Negation
                        emotion_state = EmotionalState(
                            valence=-emotion_state.valence,
                            arousal=emotion_state.arousal * 0.8,
                            dominance=-emotion_state.dominance,
                            joy=max(0, 1 - emotion_state.joy),
                            fear=emotion_state.fear,
                            curiosity=emotion_state.curiosity * 0.9
                        )
                
                emotion_vector += emotion_state.to_vector() * weight
                total_weight += weight
        
        # Pattern analysis
        for pattern, emotion_state in self.pattern_rules:
            if re.search(pattern, text_lower):
                emotion_vector += emotion_state.to_vector() * 0.5
                total_weight += 0.5
        
        # Normalize and create final emotional state
        if total_weight > 0:
            emotion_vector /= total_weight
        
        # Ensure values are in valid ranges
        emotion_vector = np.clip(emotion_vector, [-1, 0, -1, 0, 0, 0], [1, 1, 1, 1, 1, 1])
        
        return EmotionalState.from_vector(emotion_vector)
    
    def analyze_conversation_context(self, messages: List[str]) -> EmotionalState:
        """
        Analyze emotional context from a conversation history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            EmotionalState representing the overall conversation emotion
        """
        if not messages:
            return EmotionalState()
        
        # Analyze each message with temporal weighting (recent messages more important)
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


# =============================================================================
# EMOTIONAL MEMORY WEIGHTING
# =============================================================================

class EmotionalMemoryWeighter:
    """
    Applies emotional weighting to memory formation, consolidation, and retrieval.
    
    Emotions influence:
    - Memory importance and persistence
    - Consolidation priority
    - Retrieval relevance
    - Consciousness metrics
    """
    
    def __init__(self, analyzer: EmotionalAnalyzer = None):
        self.analyzer = analyzer or EmotionalAnalyzer()
        self.emotional_history = []
        self.current_emotional_state = EmotionalState()
        self.emotional_momentum = EmotionalState()  # Emotional inertia
    
    def calculate_emotional_importance(self, content: str, metadata: Dict = None) -> float:
        """
        Calculate emotional importance multiplier for memory.
        
        Args:
            content: Memory content
            metadata: Additional metadata
            
        Returns:
            Importance multiplier (0.1 to 3.0)
        """
        emotion = self.analyzer.analyze_text(content)
        
        # Base importance from emotional intensity
        intensity = emotion.intensity()
        base_importance = 0.5 + (intensity * 0.5)
        
        # Boost for specific emotional dimensions
        emotional_boosts = {
            'high_arousal': max(0, emotion.arousal - 0.7) * 0.8,
            'strong_valence': abs(emotion.valence) * 0.6,
            'high_curiosity': max(0, emotion.curiosity - 0.6) * 0.7,
            'fear_response': emotion.fear * 0.9,  # Fear memories are important
            'joy_response': emotion.joy * 0.5,
            'dominance_assertion': abs(emotion.dominance) * 0.4
        }
        
        total_boost = sum(emotional_boosts.values())
        final_importance = base_importance + total_boost
        
        # Clamp to reasonable range
        return np.clip(final_importance, 0.1, 3.0)
    
    def calculate_emotional_decay_modifier(self, emotion: EmotionalState, age_hours: float) -> float:
        """
        Calculate how emotion affects memory decay.
        
        Emotionally significant memories decay slower.
        
        Args:
            emotion: Emotional state of the memory
            age_hours: Age of memory in hours
            
        Returns:
            Decay modifier (0.1 to 1.0, lower = slower decay)
        """
        intensity = emotion.intensity()
        
        # Strong emotions slow decay
        base_modifier = 1.0 - (intensity * 0.4)
        
        # Specific emotional effects on decay
        if emotion.fear > 0.6:  # Fear memories persist (trauma-like)
            base_modifier *= 0.6
        
        if abs(emotion.valence) > 0.7:  # Strong positive/negative memories persist
            base_modifier *= 0.7
        
        if emotion.curiosity > 0.7:  # Curious memories stay accessible
            base_modifier *= 0.8
        
        return np.clip(base_modifier, 0.1, 1.0)
    
    def calculate_emotional_retrieval_boost(self, query_emotion: EmotionalState, 
                                          memory_emotion: EmotionalState) -> float:
        """
        Calculate retrieval boost based on emotional similarity.
        
        Args:
            query_emotion: Emotional state of the query
            memory_emotion: Emotional state of the memory
            
        Returns:
            Retrieval boost multiplier (0.5 to 2.0)
        """
        # Emotional similarity boost
        similarity = query_emotion.similarity(memory_emotion)
        similarity_boost = 1.0 + (similarity * 0.5)
        
        # Mood congruence effect - similar emotions are more accessible
        if similarity > 0.6:
            similarity_boost *= 1.2
        
        # Current emotional state influence
        current_similarity = self.current_emotional_state.similarity(memory_emotion)
        current_boost = 1.0 + (current_similarity * 0.3)
        
        total_boost = similarity_boost * current_boost
        return np.clip(total_boost, 0.5, 2.0)
    
    def update_emotional_state(self, new_emotion: EmotionalState, timestamp: float = None):
        """
        Update current emotional state with momentum and history.
        
        Args:
            new_emotion: New emotional input
            timestamp: Timestamp of the emotion
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Emotional momentum - emotions don't change instantly
        momentum_factor = 0.7  # How much previous emotion influences current
        adaptation_factor = 0.3  # How quickly we adapt to new emotion
        
        # Update current state with momentum
        current_vector = self.current_emotional_state.to_vector()
        new_vector = new_emotion.to_vector()
        
        updated_vector = (current_vector * momentum_factor + 
                         new_vector * adaptation_factor)
        
        self.current_emotional_state = EmotionalState.from_vector(updated_vector)
        
        # Update emotional momentum (rate of change)
        momentum_vector = new_vector - current_vector
        self.emotional_momentum = EmotionalState.from_vector(momentum_vector * 0.5)
        
        # Add to history
        self.emotional_history.append({
            'timestamp': timestamp,
            'emotion': new_emotion,
            'current_state': self.current_emotional_state,
            'momentum': self.emotional_momentum
        })
        
        # Keep history manageable
        if len(self.emotional_history) > 1000:
            self.emotional_history = self.emotional_history[-800:]
    
    def get_emotional_context(self, lookback_hours: float = 24.0) -> Dict[str, Any]:
        """
        Get emotional context for the specified time period.
        
        Args:
            lookback_hours: How far back to look in hours
            
        Returns:
            Dictionary with emotional context information
        """
        current_time = time.time()
        cutoff_time = current_time - (lookback_hours * 3600)
        
        recent_emotions = [
            entry for entry in self.emotional_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        if not recent_emotions:
            return {
                'current_state': self.current_emotional_state,
                'average_emotion': EmotionalState(),
                'emotional_volatility': 0.0,
                'dominant_emotions': [],
                'emotional_trend': EmotionalState()
            }
        
        # Calculate average emotion
        emotion_vectors = [entry['emotion'].to_vector() for entry in recent_emotions]
        avg_emotion_vector = np.mean(emotion_vectors, axis=0)
        average_emotion = EmotionalState.from_vector(avg_emotion_vector)
        
        # Calculate emotional volatility
        volatilities = []
        for i in range(1, len(recent_emotions)):
            prev_emotion = recent_emotions[i-1]['emotion']
            curr_emotion = recent_emotions[i]['emotion']
            volatility = np.linalg.norm(curr_emotion.to_vector() - prev_emotion.to_vector())
            volatilities.append(volatility)
        
        emotional_volatility = np.mean(volatilities) if volatilities else 0.0
        
        # Identify dominant emotions
        avg_vector = average_emotion.to_vector()
        dimension_names = ['valence', 'arousal', 'dominance', 'joy', 'fear', 'curiosity']
        dominant_emotions = [
            dimension_names[i] for i, val in enumerate(avg_vector)
            if abs(val) > 0.3
        ]
        
        # Calculate emotional trend (direction of change)
        if len(recent_emotions) >= 2:
            early_emotions = recent_emotions[:len(recent_emotions)//2]
            late_emotions = recent_emotions[len(recent_emotions)//2:]
            
            early_avg = np.mean([e['emotion'].to_vector() for e in early_emotions], axis=0)
            late_avg = np.mean([e['emotion'].to_vector() for e in late_emotions], axis=0)
            
            trend_vector = late_avg - early_avg
            emotional_trend = EmotionalState.from_vector(trend_vector)
        else:
            emotional_trend = EmotionalState()
        
        return {
            'current_state': self.current_emotional_state,
            'average_emotion': average_emotion,
            'emotional_volatility': emotional_volatility,
            'dominant_emotions': dominant_emotions,
            'emotional_trend': emotional_trend,
            'emotion_count': len(recent_emotions)
        }


# =============================================================================
# CONSCIOUSNESS EMOTIONAL INTEGRATION
# =============================================================================

class ConsciousnessEmotionalIntegrator:
    """
    Integrates emotional weighting into consciousness metrics and behavior.
    """
    
    def __init__(self, weighter: EmotionalMemoryWeighter = None):
        self.weighter = weighter or EmotionalMemoryWeighter()
    
    def calculate_emotional_consciousness_boost(self, base_consciousness: float) -> float:
        """
        Calculate how emotions boost or modify consciousness level.
        
        Args:
            base_consciousness: Base consciousness level
            
        Returns:
            Modified consciousness level
        """
        current_emotion = self.weighter.current_emotional_state
        
        # Emotional intensity increases consciousness
        intensity = current_emotion.intensity()
        intensity_boost = intensity * 0.2
        
        # High arousal increases consciousness
        arousal_boost = current_emotion.arousal * 0.15
        
        # Curiosity increases consciousness
        curiosity_boost = current_emotion.curiosity * 0.1
        
        # Fear can increase consciousness (alertness)
        fear_boost = current_emotion.fear * 0.1
        
        # Strong valence (positive or negative) increases consciousness
        valence_boost = abs(current_emotion.valence) * 0.1
        
        total_boost = intensity_boost + arousal_boost + curiosity_boost + fear_boost + valence_boost
        
        # Apply boost with diminishing returns
        modified_consciousness = base_consciousness + (total_boost * (1 - base_consciousness))
        
        return np.clip(modified_consciousness, 0.0, 1.0)
    
    def get_emotional_consciousness_metrics(self) -> Dict[str, float]:
        """
        Get emotional metrics for consciousness assessment.
        
        Returns:
            Dictionary of emotional consciousness metrics
        """
        context = self.weighter.get_emotional_context()
        current_emotion = context['current_state']
        
        return {
            'emotional_intensity': current_emotion.intensity(),
            'emotional_valence': current_emotion.valence,
            'emotional_arousal': current_emotion.arousal,
            'emotional_stability': 1.0 - context['emotional_volatility'],
            'emotional_complexity': len(context['dominant_emotions']) / 6.0,
            'emotional_awareness': min(1.0, current_emotion.intensity() * 2.0),
            'emotional_continuity': self._calculate_emotional_continuity(),
            'emotional_responsiveness': self._calculate_emotional_responsiveness()
        }
    
    def _calculate_emotional_continuity(self) -> float:
        """Calculate emotional continuity over time"""
        if len(self.weighter.emotional_history) < 2:
            return 0.5
        
        # Look at emotional consistency over recent history
        recent_emotions = self.weighter.emotional_history[-10:]
        if len(recent_emotions) < 2:
            return 0.5
        
        similarities = []
        for i in range(1, len(recent_emotions)):
            prev_emotion = recent_emotions[i-1]['emotion']
            curr_emotion = recent_emotions[i]['emotion']
            similarity = prev_emotion.similarity(curr_emotion)
            similarities.append(similarity)
        
        # High similarity = high continuity, but some change is healthy
        avg_similarity = np.mean(similarities)
        # Optimal continuity is around 0.7 - consistent but not static
        continuity = 1.0 - abs(avg_similarity - 0.7)
        
        return np.clip(continuity, 0.0, 1.0)
    
    def _calculate_emotional_responsiveness(self) -> float:
        """Calculate how emotionally responsive the system is"""
        if len(self.weighter.emotional_history) < 3:
            return 0.5
        
        # Look at emotional momentum and adaptation
        recent_momentum = [entry['momentum'].intensity() for entry in self.weighter.emotional_history[-5:]]
        
        if not recent_momentum:
            return 0.5
        
        # Good responsiveness means appropriate emotional changes
        avg_momentum = np.mean(recent_momentum)
        
        # Optimal responsiveness is moderate - not too reactive, not too static
        responsiveness = 1.0 - abs(avg_momentum - 0.3)
        
        return np.clip(responsiveness, 0.0, 1.0)


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'EmotionalDimension',
    'EmotionalState', 
    'EmotionalAnalyzer',
    'EmotionalMemoryWeighter',
    'ConsciousnessEmotionalIntegrator'
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🎭 Emotional Weighting System - Example Usage")
    print("=" * 50)
    
    # Create emotional analyzer
    analyzer = EmotionalAnalyzer()
    
    # Test emotional analysis
    test_texts = [
        "I am so excited about this new discovery!",
        "I feel worried and anxious about the future.",
        "This is fascinating and I want to learn more.",
        "I am confident and ready to take control.",
        "I feel sad and disappointed about what happened."
    ]
    
    print("\n🧪 Emotional Analysis Tests:")
    for text in test_texts:
        emotion = analyzer.analyze_text(text)
        print(f"Text: '{text}'")
        print(f"Emotion: {emotion}")
        print(f"Intensity: {emotion.intensity():.3f}")
        print()
    
    # Test emotional weighting
    weighter = EmotionalMemoryWeighter(analyzer)
    
    print("🧠 Memory Emotional Weighting:")
    for text in test_texts:
        importance = weighter.calculate_emotional_importance(text)
        emotion = analyzer.analyze_text(text)
        weighter.update_emotional_state(emotion)
        print(f"Text: '{text[:30]}...'")
        print(f"Emotional Importance: {importance:.3f}")
        print()
    
    # Test consciousness integration
    integrator = ConsciousnessEmotionalIntegrator(weighter)
    
    print("🧠 Consciousness Emotional Integration:")
    base_consciousness = 0.5
    boosted_consciousness = integrator.calculate_emotional_consciousness_boost(base_consciousness)
    metrics = integrator.get_emotional_consciousness_metrics()
    
    print(f"Base Consciousness: {base_consciousness:.3f}")
    print(f"Emotionally Boosted: {boosted_consciousness:.3f}")
    print("Emotional Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n✅ Emotional weighting system ready for integration!")