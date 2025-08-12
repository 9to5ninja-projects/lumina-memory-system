# Enhanced Memory Analytics
# This demonstrates safe development practices

from typing import Dict, Any
import time
from datetime import datetime

class MemoryAnalytics:
    """Enhanced analytics for memory performance monitoring."""
    
    def __init__(self):
        self.query_history = []
        self.performance_metrics = {
            'total_queries': 0,
            'average_response_time': 0.0,
            'memory_efficiency': 0.0,
            'last_updated': datetime.now().isoformat()
        }
    
    def record_query_performance(self, query_time: float, results_count: int):
        """Record query performance metrics."""
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_time': query_time,
            'results_count': results_count
        })
        
        # Update running metrics
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['average_response_time'] = (
            sum(q['query_time'] for q in self.query_history[-100:]) / 
            min(len(self.query_history), 100)  # Rolling average of last 100
        )
        self.performance_metrics['last_updated'] = datetime.now().isoformat()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.query_history:
            return {'status': 'No queries recorded yet'}
            
        recent_queries = self.query_history[-10:]
        
        return {
            'total_queries': len(self.query_history),
            'recent_average_time': sum(q['query_time'] for q in recent_queries) / len(recent_queries),
            'fastest_query': min(q['query_time'] for q in self.query_history),
            'slowest_query': max(q['query_time'] for q in self.query_history),
            'queries_per_minute': len([q for q in recent_queries if 
                (datetime.now() - datetime.fromisoformat(q['timestamp'])).seconds < 60]),
            'last_updated': self.performance_metrics['last_updated']
        }
