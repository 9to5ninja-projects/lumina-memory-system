"""Tests for memory analytics module."""

import pytest
import time
from src.lumina_memory.analytics import MemoryAnalytics


def test_analytics_initialization():
    """Test analytics module initializes correctly."""
    analytics = MemoryAnalytics()
    assert analytics.performance_metrics['total_queries'] == 0
    assert len(analytics.query_history) == 0


def test_record_query_performance():
    """Test recording query performance."""
    analytics = MemoryAnalytics()
    
    # Record a query
    analytics.record_query_performance(0.5, 10)
    
    assert analytics.performance_metrics['total_queries'] == 1
    assert len(analytics.query_history) == 1
    assert analytics.query_history[0]['query_time'] == 0.5
    assert analytics.query_history[0]['results_count'] == 10


def test_performance_summary():
    """Test performance summary generation."""
    analytics = MemoryAnalytics()
    
    # No queries yet
    summary = analytics.get_performance_summary()
    assert 'No queries recorded' in summary['status']
    
    # Add some queries
    for i in range(5):
        analytics.record_query_performance(0.1 + (i * 0.1), i + 1)
    
    summary = analytics.get_performance_summary()
    assert summary['total_queries'] == 5
    assert 'recent_average_time' in summary
    assert 'fastest_query' in summary
    assert 'slowest_query' in summary


def test_rolling_average():
    """Test rolling average calculation."""
    analytics = MemoryAnalytics()
    
    # Add many queries to test rolling average
    for i in range(150):  # More than 100 to test rolling window
        analytics.record_query_performance(0.1, 1)
    
    # Average should be based on last 100 queries only
    assert analytics.performance_metrics['average_response_time'] == 0.1


if __name__ == "__main__":
    print("Running analytics tests...")
    test_analytics_initialization()
    test_record_query_performance()
    test_performance_summary()
    test_rolling_average()
    print(" All tests passed!")
