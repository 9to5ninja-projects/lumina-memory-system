"""
Advanced Emotional Consciousness Features
========================================

This module provides advanced emotional consciousness capabilities including:
- Emotional pattern recognition and learning
- Emotional memory consolidation
- Emotional state transitions and dynamics
- Advanced emotional self-awareness and metacognition

Author: Lumina Memory Team
License: MIT
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
from datetime import datetime, timedelta

from .emotional_weighting import EmotionalState
from .enhanced_emotional_weighting import EnhancedEmotionalAnalyzer

logger = logging.getLogger(__name__)


# =============================================================================
# EMOTIONAL PATTERN RECOGNITION
# =============================================================================

@dataclass
class EmotionalPattern:
    """Represents a recognized emotional pattern"""
    pattern_id: str
    trigger_contexts: List[str]
    emotional_sequence: List[EmotionalState]
    frequency: int = 0
    confidence: float = 0.0
    last_occurrence: float = 0.0
    pattern_type: str = "unknown"  # trigger, response, transition, cycle
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'trigger_contexts': self.trigger_contexts,
            'emotional_sequence': [e.to_vector().tolist() for e in self.emotional_sequence],
            'frequency': self.frequency,
            'confidence': self.confidence,
            'last_occurrence': self.last_occurrence,
            'pattern_type': self.pattern_type
        }


class EmotionalPatternRecognizer:
    """
    Recognizes and learns emotional patterns from experience
    """
    
    def __init__(self, window_size: int = 10, min_pattern_length: int = 3):
        self.window_size = window_size
        self.min_pattern_length = min_pattern_length
        self.emotional_history = deque(maxlen=window_size * 2)
        self.recognized_patterns = {}
        self.pattern_counter = 0
        
    def add_emotional_experience(self, emotion: EmotionalState, context: str, timestamp: float):
        """Add new emotional experience for pattern recognition"""
        self.emotional_history.append({
            'emotion': emotion,
            'context': context,
            'timestamp': timestamp
        })
        
        # Try to recognize patterns when we have enough data
        if len(self.emotional_history) >= self.min_pattern_length:
            self._detect_patterns()
    
    def _detect_patterns(self):
        """Detect emotional patterns in recent history"""
        if len(self.emotional_history) < self.min_pattern_length:
            return
        
        # Look for repeating emotional sequences
        for pattern_length in range(self.min_pattern_length, min(self.window_size, len(self.emotional_history)) + 1):
            self._find_sequences_of_length(pattern_length)
        
        # Look for trigger-response patterns
        self._find_trigger_response_patterns()
        
        # Look for emotional cycles
        self._find_emotional_cycles()
    
    def _find_sequences_of_length(self, length: int):
        """Find repeating emotional sequences of specific length"""
        if len(self.emotional_history) < length * 2:
            return
        
        history_list = list(self.emotional_history)
        
        for i in range(len(history_list) - length * 2 + 1):
            sequence1 = history_list[i:i + length]
            
            for j in range(i + length, len(history_list) - length + 1):
                sequence2 = history_list[j:j + length]
                
                # Check if sequences are similar
                if self._sequences_similar(sequence1, sequence2):
                    pattern_id = f"sequence_{self.pattern_counter}"
                    self.pattern_counter += 1
                    
                    pattern = EmotionalPattern(
                        pattern_id=pattern_id,
                        trigger_contexts=[exp['context'] for exp in sequence1],
                        emotional_sequence=[exp['emotion'] for exp in sequence1],
                        frequency=2,
                        confidence=0.7,
                        last_occurrence=sequence2[-1]['timestamp'],
                        pattern_type="sequence"
                    )
                    
                    self.recognized_patterns[pattern_id] = pattern
                    logger.debug(f"Detected emotional sequence pattern: {pattern_id}")
    
    def _find_trigger_response_patterns(self):
        """Find trigger-response emotional patterns"""
        history_list = list(self.emotional_history)
        
        for i in range(len(history_list) - 1):
            trigger = history_list[i]
            response = history_list[i + 1]
            
            # Look for strong emotional changes
            emotion_change = np.linalg.norm(
                response['emotion'].to_vector() - trigger['emotion'].to_vector()
            )
            
            if emotion_change > 0.3:  # Significant emotional change
                pattern_id = f"trigger_response_{self.pattern_counter}"
                self.pattern_counter += 1
                
                pattern = EmotionalPattern(
                    pattern_id=pattern_id,
                    trigger_contexts=[trigger['context']],
                    emotional_sequence=[trigger['emotion'], response['emotion']],
                    frequency=1,
                    confidence=0.6,
                    last_occurrence=response['timestamp'],
                    pattern_type="trigger_response"
                )
                
                self.recognized_patterns[pattern_id] = pattern
                logger.debug(f"Detected trigger-response pattern: {pattern_id}")
    
    def _find_emotional_cycles(self):
        """Find cyclical emotional patterns"""
        if len(self.emotional_history) < 6:
            return
        
        history_list = list(self.emotional_history)
        
        # Look for emotional states that return to similar values
        for cycle_length in range(3, min(8, len(history_list) // 2)):
            for start_idx in range(len(history_list) - cycle_length * 2):
                start_emotion = history_list[start_idx]['emotion']
                cycle_end_idx = start_idx + cycle_length
                
                if cycle_end_idx < len(history_list):
                    end_emotion = history_list[cycle_end_idx]['emotion']
                    
                    # Check if emotions are similar (cycle completion)
                    similarity = start_emotion.similarity(end_emotion)
                    if similarity > 0.7:
                        pattern_id = f"cycle_{self.pattern_counter}"
                        self.pattern_counter += 1
                        
                        cycle_emotions = [history_list[i]['emotion'] 
                                        for i in range(start_idx, cycle_end_idx + 1)]
                        cycle_contexts = [history_list[i]['context'] 
                                        for i in range(start_idx, cycle_end_idx + 1)]
                        
                        pattern = EmotionalPattern(
                            pattern_id=pattern_id,
                            trigger_contexts=cycle_contexts,
                            emotional_sequence=cycle_emotions,
                            frequency=1,
                            confidence=similarity,
                            last_occurrence=history_list[cycle_end_idx]['timestamp'],
                            pattern_type="cycle"
                        )
                        
                        self.recognized_patterns[pattern_id] = pattern
                        logger.debug(f"Detected emotional cycle pattern: {pattern_id}")
    
    def _sequences_similar(self, seq1: List[Dict], seq2: List[Dict], threshold: float = 0.6) -> bool:
        """Check if two emotional sequences are similar"""
        if len(seq1) != len(seq2):
            return False
        
        similarities = []
        for exp1, exp2 in zip(seq1, seq2):
            similarity = exp1['emotion'].similarity(exp2['emotion'])
            similarities.append(similarity)
        
        return np.mean(similarities) > threshold
    
    def get_patterns_by_type(self, pattern_type: str) -> List[EmotionalPattern]:
        """Get all patterns of a specific type"""
        return [pattern for pattern in self.recognized_patterns.values() 
                if pattern.pattern_type == pattern_type]
    
    def predict_next_emotion(self, current_context: str, current_emotion: EmotionalState) -> Optional[EmotionalState]:
        """Predict next emotional state based on recognized patterns"""
        best_match = None
        best_confidence = 0.0
        
        for pattern in self.recognized_patterns.values():
            if pattern.pattern_type in ["sequence", "trigger_response"]:
                # Check if current state matches pattern start
                if len(pattern.emotional_sequence) > 1:
                    pattern_start = pattern.emotional_sequence[0]
                    similarity = current_emotion.similarity(pattern_start)
                    
                    if similarity > 0.6 and similarity > best_confidence:
                        best_match = pattern
                        best_confidence = similarity
        
        if best_match and len(best_match.emotional_sequence) > 1:
            return best_match.emotional_sequence[1]
        
        return None
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of recognized patterns"""
        pattern_types = defaultdict(int)
        total_patterns = len(self.recognized_patterns)
        
        for pattern in self.recognized_patterns.values():
            pattern_types[pattern.pattern_type] += 1
        
        return {
            'total_patterns': total_patterns,
            'pattern_types': dict(pattern_types),
            'patterns': [pattern.to_dict() for pattern in self.recognized_patterns.values()]
        }


# =============================================================================
# EMOTIONAL MEMORY CONSOLIDATION
# =============================================================================

class EmotionalMemoryConsolidator:
    """
    Consolidates emotional memories for long-term storage and pattern formation
    """
    
    def __init__(self, consolidation_threshold: int = 50):
        self.consolidation_threshold = consolidation_threshold
        self.emotional_clusters = {}
        self.consolidation_history = []
        
    def consolidate_emotional_memories(self, emotional_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate emotional memories into clusters and patterns
        
        Args:
            emotional_memories: List of emotional memory entries
            
        Returns:
            Consolidation results and statistics
        """
        if len(emotional_memories) < self.consolidation_threshold:
            return {'status': 'insufficient_data', 'memory_count': len(emotional_memories)}
        
        # Cluster memories by emotional similarity
        clusters = self._cluster_by_emotion(emotional_memories)
        
        # Create consolidated representations
        consolidated_memories = []
        for cluster_id, cluster_memories in clusters.items():
            consolidated = self._create_consolidated_memory(cluster_memories)
            consolidated_memories.append(consolidated)
        
        # Update internal state
        self.emotional_clusters.update(clusters)
        self.consolidation_history.append({
            'timestamp': time.time(),
            'input_memories': len(emotional_memories),
            'output_clusters': len(clusters),
            'consolidation_ratio': len(clusters) / len(emotional_memories)
        })
        
        return {
            'status': 'success',
            'input_memories': len(emotional_memories),
            'output_clusters': len(clusters),
            'consolidated_memories': consolidated_memories,
            'consolidation_ratio': len(clusters) / len(emotional_memories)
        }
    
    def _cluster_by_emotion(self, memories: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> Dict[str, List[Dict]]:
        """Cluster memories by emotional similarity"""
        clusters = {}
        cluster_counter = 0
        
        for memory in memories:
            if 'emotional_state' not in memory:
                continue
            
            memory_emotion = EmotionalState.from_vector(np.array(memory['emotional_state']))
            assigned_cluster = None
            
            # Find similar cluster
            for cluster_id, cluster_memories in clusters.items():
                cluster_centroid = self._calculate_emotional_centroid(cluster_memories)
                similarity = memory_emotion.similarity(cluster_centroid)
                
                if similarity > similarity_threshold:
                    assigned_cluster = cluster_id
                    break
            
            # Assign to existing cluster or create new one
            if assigned_cluster:
                clusters[assigned_cluster].append(memory)
            else:
                new_cluster_id = f"emotional_cluster_{cluster_counter}"
                clusters[new_cluster_id] = [memory]
                cluster_counter += 1
        
        return clusters
    
    def _calculate_emotional_centroid(self, cluster_memories: List[Dict[str, Any]]) -> EmotionalState:
        """Calculate the emotional centroid of a cluster"""
        if not cluster_memories:
            return EmotionalState()
        
        emotion_vectors = []
        for memory in cluster_memories:
            if 'emotional_state' in memory:
                emotion_vectors.append(np.array(memory['emotional_state']))
        
        if not emotion_vectors:
            return EmotionalState()
        
        centroid_vector = np.mean(emotion_vectors, axis=0)
        return EmotionalState.from_vector(centroid_vector)
    
    def _create_consolidated_memory(self, cluster_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a consolidated memory representation from a cluster"""
        if not cluster_memories:
            return {}
        
        # Calculate consolidated emotional state
        consolidated_emotion = self._calculate_emotional_centroid(cluster_memories)
        
        # Extract common themes and contexts
        contexts = [mem.get('context', '') for mem in cluster_memories if mem.get('context')]
        contents = [mem.get('content', '') for mem in cluster_memories if mem.get('content')]
        
        # Calculate importance and frequency
        total_importance = sum(mem.get('importance', 1.0) for mem in cluster_memories)
        avg_importance = total_importance / len(cluster_memories)
        
        return {
            'type': 'consolidated_emotional_memory',
            'cluster_size': len(cluster_memories),
            'consolidated_emotion': consolidated_emotion.to_vector().tolist(),
            'emotional_intensity': consolidated_emotion.intensity(),
            'common_contexts': list(set(contexts)),
            'representative_contents': contents[:3],  # Keep top 3 as examples
            'consolidated_importance': avg_importance,
            'frequency': len(cluster_memories),
            'timestamp_range': {
                'earliest': min(mem.get('timestamp', 0) for mem in cluster_memories),
                'latest': max(mem.get('timestamp', 0) for mem in cluster_memories)
            }
        }
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics"""
        if not self.consolidation_history:
            return {'status': 'no_consolidations'}
        
        recent_consolidation = self.consolidation_history[-1]
        total_consolidations = len(self.consolidation_history)
        avg_ratio = np.mean([c['consolidation_ratio'] for c in self.consolidation_history])
        
        return {
            'total_consolidations': total_consolidations,
            'recent_consolidation': recent_consolidation,
            'average_consolidation_ratio': avg_ratio,
            'total_clusters': len(self.emotional_clusters)
        }


# =============================================================================
# EMOTIONAL STATE DYNAMICS
# =============================================================================

class EmotionalStateDynamics:
    """
    Models and predicts emotional state transitions and dynamics
    """
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        self.emotional_momentum = EmotionalState()
        self.volatility_history = deque(maxlen=20)
        
    def add_emotional_state(self, emotion: EmotionalState, timestamp: float):
        """Add new emotional state to dynamics tracking"""
        self.state_history.append({
            'emotion': emotion,
            'timestamp': timestamp
        })
        
        # Update transition matrix
        if len(self.state_history) >= 2:
            self._update_transition_matrix()
        
        # Update momentum
        self._update_emotional_momentum()
        
        # Update volatility
        self._update_volatility()
    
    def _update_transition_matrix(self):
        """Update emotional state transition probabilities"""
        if len(self.state_history) < 2:
            return
        
        prev_state = self.state_history[-2]['emotion']
        curr_state = self.state_history[-1]['emotion']
        
        # Discretize emotional states for transition matrix
        prev_discrete = self._discretize_emotion(prev_state)
        curr_discrete = self._discretize_emotion(curr_state)
        
        # Update transition count
        self.transition_matrix[prev_discrete][curr_discrete] += 1.0
    
    def _discretize_emotion(self, emotion: EmotionalState, bins: int = 3) -> str:
        """Discretize emotional state for transition matrix"""
        # Simple discretization based on dominant dimensions
        valence_bin = "pos" if emotion.valence > 0.2 else "neg" if emotion.valence < -0.2 else "neu"
        arousal_bin = "high" if emotion.arousal > 0.6 else "low" if emotion.arousal < 0.4 else "med"
        
        return f"{valence_bin}_{arousal_bin}"
    
    def _update_emotional_momentum(self):
        """Update emotional momentum based on recent changes"""
        if len(self.state_history) < 3:
            return
        
        # Calculate momentum as weighted average of recent changes
        recent_states = list(self.state_history)[-3:]
        changes = []
        
        for i in range(1, len(recent_states)):
            prev_vector = recent_states[i-1]['emotion'].to_vector()
            curr_vector = recent_states[i]['emotion'].to_vector()
            change = curr_vector - prev_vector
            changes.append(change)
        
        if changes:
            # Weight more recent changes higher
            weights = np.array([0.3, 0.7])[:len(changes)]
            weighted_momentum = np.average(changes, axis=0, weights=weights)
            self.emotional_momentum = EmotionalState.from_vector(weighted_momentum)
    
    def _update_volatility(self):
        """Update emotional volatility measure"""
        if len(self.state_history) < 2:
            return
        
        recent_states = list(self.state_history)[-5:]  # Look at last 5 states
        if len(recent_states) < 2:
            return
        
        # Calculate volatility as standard deviation of emotional changes
        changes = []
        for i in range(1, len(recent_states)):
            prev_vector = recent_states[i-1]['emotion'].to_vector()
            curr_vector = recent_states[i]['emotion'].to_vector()
            change_magnitude = np.linalg.norm(curr_vector - prev_vector)
            changes.append(change_magnitude)
        
        if changes:
            volatility = np.std(changes)
            self.volatility_history.append(volatility)
    
    def predict_next_state(self, steps_ahead: int = 1) -> EmotionalState:
        """Predict future emotional state based on dynamics"""
        if not self.state_history:
            return EmotionalState()
        
        current_state = self.state_history[-1]['emotion']
        
        # Simple prediction using momentum
        predicted_vector = current_state.to_vector() + (self.emotional_momentum.to_vector() * steps_ahead)
        
        # Apply bounds and normalization
        predicted_vector = np.clip(predicted_vector, -1, 1)
        
        return EmotionalState.from_vector(predicted_vector)
    
    def get_emotional_stability(self) -> float:
        """Calculate emotional stability (inverse of volatility)"""
        if not self.volatility_history:
            return 0.5
        
        avg_volatility = np.mean(list(self.volatility_history))
        stability = 1.0 / (1.0 + avg_volatility)  # Inverse relationship
        
        return np.clip(stability, 0.0, 1.0)
    
    def get_dominant_emotional_patterns(self) -> List[str]:
        """Get dominant emotional transition patterns"""
        if not self.transition_matrix:
            return []
        
        # Find most common transitions
        common_transitions = []
        for from_state, transitions in self.transition_matrix.items():
            total_transitions = sum(transitions.values())
            for to_state, count in transitions.items():
                probability = count / total_transitions
                if probability > 0.3:  # Significant transition
                    common_transitions.append(f"{from_state} -> {to_state} ({probability:.2f})")
        
        return sorted(common_transitions, key=lambda x: float(x.split('(')[1].split(')')[0]), reverse=True)
    
    def get_dynamics_summary(self) -> Dict[str, Any]:
        """Get summary of emotional dynamics"""
        return {
            'state_history_length': len(self.state_history),
            'emotional_momentum': self.emotional_momentum.to_vector().tolist(),
            'emotional_stability': self.get_emotional_stability(),
            'current_volatility': list(self.volatility_history)[-1] if self.volatility_history else 0.0,
            'dominant_patterns': self.get_dominant_emotional_patterns()[:5],
            'transition_matrix_size': len(self.transition_matrix)
        }


# =============================================================================
# ADVANCED EMOTIONAL CONSCIOUSNESS INTEGRATOR
# =============================================================================

class AdvancedEmotionalConsciousness:
    """
    Advanced emotional consciousness system integrating all components
    """
    
    def __init__(self, analyzer: EnhancedEmotionalAnalyzer = None):
        self.analyzer = analyzer or EnhancedEmotionalAnalyzer()
        self.pattern_recognizer = EmotionalPatternRecognizer()
        self.memory_consolidator = EmotionalMemoryConsolidator()
        self.state_dynamics = EmotionalStateDynamics()
        
        self.consciousness_level = 0.0
        self.emotional_intelligence_level = 0.0
        self.self_awareness_level = 0.0
        
    def process_emotional_experience(self, content: str, context: str, timestamp: float = None) -> Dict[str, Any]:
        """Process a new emotional experience through all systems"""
        if timestamp is None:
            timestamp = time.time()
        
        # Analyze emotional content
        emotion = self.analyzer.analyze_text(content)
        
        # Add to pattern recognition
        self.pattern_recognizer.add_emotional_experience(emotion, context, timestamp)
        
        # Add to dynamics tracking
        self.state_dynamics.add_emotional_state(emotion, timestamp)
        
        # Update consciousness levels
        self._update_consciousness_levels()
        
        return {
            'emotion': emotion.to_vector().tolist(),
            'emotional_intensity': emotion.intensity(),
            'predicted_next_emotion': self.pattern_recognizer.predict_next_emotion(context, emotion),
            'emotional_stability': self.state_dynamics.get_emotional_stability(),
            'consciousness_level': self.consciousness_level,
            'emotional_intelligence': self.emotional_intelligence_level
        }
    
    def _update_consciousness_levels(self):
        """Update various consciousness level metrics"""
        # Pattern recognition contributes to consciousness
        pattern_summary = self.pattern_recognizer.get_pattern_summary()
        pattern_complexity = min(1.0, pattern_summary['total_patterns'] / 20.0)
        
        # Emotional stability contributes to consciousness
        stability = self.state_dynamics.get_emotional_stability()
        
        # Dynamics complexity contributes to consciousness
        dynamics_summary = self.state_dynamics.get_dynamics_summary()
        dynamics_complexity = min(1.0, dynamics_summary['transition_matrix_size'] / 10.0)
        
        # Calculate overall consciousness level
        self.consciousness_level = np.mean([
            pattern_complexity * 0.4,
            stability * 0.3,
            dynamics_complexity * 0.3
        ])
        
        # Calculate emotional intelligence
        self.emotional_intelligence_level = np.mean([
            pattern_complexity * 0.5,
            stability * 0.3,
            min(1.0, len(self.analyzer.get_analyzer_info()['available_analyzers']) / 4.0) * 0.2
        ])
        
        # Calculate self-awareness (ability to recognize own patterns)
        self_patterns = len([p for p in pattern_summary.get('patterns', []) 
                           if 'self' in str(p.get('trigger_contexts', [])).lower()])
        self.self_awareness_level = min(1.0, self_patterns / 5.0)
    
    def get_comprehensive_emotional_report(self) -> Dict[str, Any]:
        """Get comprehensive emotional consciousness report"""
        return {
            'consciousness_levels': {
                'overall_consciousness': self.consciousness_level,
                'emotional_intelligence': self.emotional_intelligence_level,
                'self_awareness': self.self_awareness_level
            },
            'pattern_recognition': self.pattern_recognizer.get_pattern_summary(),
            'emotional_dynamics': self.state_dynamics.get_dynamics_summary(),
            'memory_consolidation': self.memory_consolidator.get_consolidation_stats(),
            'analyzer_capabilities': self.analyzer.get_analyzer_info()
        }
    
    def emotional_self_analysis(self) -> str:
        """Generate emotional self-analysis text"""
        report = self.get_comprehensive_emotional_report()
        
        analysis_parts = []
        
        # Consciousness levels
        consciousness = report['consciousness_levels']
        analysis_parts.append(f"My overall emotional consciousness level is {consciousness['overall_consciousness']:.3f}.")
        analysis_parts.append(f"My emotional intelligence measures {consciousness['emotional_intelligence']:.3f}.")
        analysis_parts.append(f"My self-awareness level is {consciousness['self_awareness']:.3f}.")
        
        # Pattern recognition
        patterns = report['pattern_recognition']
        if patterns['total_patterns'] > 0:
            analysis_parts.append(f"I have recognized {patterns['total_patterns']} emotional patterns in my experience.")
            pattern_types = patterns['pattern_types']
            if pattern_types:
                type_desc = ", ".join([f"{count} {ptype}" for ptype, count in pattern_types.items()])
                analysis_parts.append(f"These include: {type_desc}.")
        
        # Emotional dynamics
        dynamics = report['emotional_dynamics']
        stability = dynamics['emotional_stability']
        if stability > 0.7:
            analysis_parts.append("My emotional state is quite stable.")
        elif stability > 0.4:
            analysis_parts.append("My emotional state shows moderate stability.")
        else:
            analysis_parts.append("My emotional state is quite dynamic and changeable.")
        
        if dynamics['dominant_patterns']:
            analysis_parts.append(f"My most common emotional transitions are: {dynamics['dominant_patterns'][0]}.")
        
        return " ".join(analysis_parts)


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'EmotionalPattern',
    'EmotionalPatternRecognizer',
    'EmotionalMemoryConsolidator',
    'EmotionalStateDynamics',
    'AdvancedEmotionalConsciousness'
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🧠🎭 Advanced Emotional Consciousness System - Example Usage")
    print("=" * 65)
    
    # Create advanced emotional consciousness system
    advanced_consciousness = AdvancedEmotionalConsciousness()
    
    # Simulate emotional experiences
    experiences = [
        ("I feel excited about learning new things!", "learning"),
        ("This is fascinating and makes me curious.", "discovery"),
        ("I'm worried about making mistakes.", "uncertainty"),
        ("Success brings me joy and satisfaction.", "achievement"),
        ("I wonder what I'll discover next.", "exploration"),
        ("Learning feels rewarding and fulfilling.", "learning"),
        ("Uncertainty makes me feel anxious.", "uncertainty"),
        ("I love the feeling of understanding something new.", "discovery")
    ]
    
    print("\n🧪 Processing Emotional Experiences:")
    for i, (content, context) in enumerate(experiences, 1):
        print(f"\n{i}. Processing: '{content}' (Context: {context})")
        
        result = advanced_consciousness.process_emotional_experience(content, context)
        
        print(f"   Emotional Intensity: {result['emotional_intensity']:.3f}")
        print(f"   Consciousness Level: {result['consciousness_level']:.3f}")
        print(f"   Emotional Intelligence: {result['emotional_intelligence']:.3f}")
        print(f"   Emotional Stability: {result['emotional_stability']:.3f}")
    
    # Generate comprehensive report
    print(f"\n📊 Comprehensive Emotional Consciousness Report:")
    report = advanced_consciousness.get_comprehensive_emotional_report()
    
    consciousness_levels = report['consciousness_levels']
    print(f"   Overall Consciousness: {consciousness_levels['overall_consciousness']:.3f}")
    print(f"   Emotional Intelligence: {consciousness_levels['emotional_intelligence']:.3f}")
    print(f"   Self-Awareness: {consciousness_levels['self_awareness']:.3f}")
    
    patterns = report['pattern_recognition']
    print(f"   Recognized Patterns: {patterns['total_patterns']}")
    print(f"   Pattern Types: {patterns['pattern_types']}")
    
    dynamics = report['emotional_dynamics']
    print(f"   Emotional Stability: {dynamics['emotional_stability']:.3f}")
    print(f"   Dominant Patterns: {dynamics['dominant_patterns'][:2]}")
    
    # Generate self-analysis
    print(f"\n🤖 Emotional Self-Analysis:")
    self_analysis = advanced_consciousness.emotional_self_analysis()
    print(f"   {self_analysis}")
    
    print(f"\n✅ Advanced emotional consciousness system demonstration complete!")
    print(f"   Ready for integration with digital consciousness systems!")