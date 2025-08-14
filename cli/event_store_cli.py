#!/usr/bin/env python3
"""
M4: Event Store CLI Interface

Command-line interface for event store operations and deterministic rebuild.
Provides tools for event store management, integrity verification, and rebuild operations.

Usage:
    python -m lumina_memory.cli.event_store_cli [command] [options]

Commands:
    init        Initialize new event store
    stats       Show event store statistics
    verify      Verify event store integrity
    rebuild     Perform deterministic rebuild
    snapshot    Create or manage snapshots
    export      Export events to file
    import      Import events from file
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lumina_memory.event_store import create_event_store, EventStore
from lumina_memory.deterministic_rebuild import create_deterministic_rebuilder
from lumina_memory.kernel import LuminaMemory
from lumina_memory.crypto_ids import memory_content_id


class EventStoreCLI:
    """Command-line interface for event store operations."""
    
    def __init__(self):
        self.event_store: Optional[EventStore] = None
        self.storage_path: Optional[Path] = None
    
    def init_event_store(self, storage_path: str, force: bool = False) -> Dict[str, Any]:
        """Initialize new event store."""
        path = Path(storage_path)
        
        if path.exists() and not force:
            return {
                'success': False,
                'error': f'Event store already exists at {path}. Use --force to overwrite.'
            }
        
        try:
            # Create event store
            self.event_store = create_event_store(path)
            self.storage_path = path
            
            # Get initial statistics
            stats = self.event_store.get_statistics()
            
            return {
                'success': True,
                'message': f'Event store initialized at {path}',
                'statistics': stats
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to initialize event store: {e}'
            }
    
    def get_statistics(self, storage_path: str) -> Dict[str, Any]:
        """Get event store statistics."""
        try:
            event_store = create_event_store(Path(storage_path))
            stats = event_store.get_statistics()
            
            return {
                'success': True,
                'statistics': stats
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to get statistics: {e}'
            }
    
    def verify_integrity(self, storage_path: str) -> Dict[str, Any]:
        """Verify event store integrity."""
        try:
            event_store = create_event_store(Path(storage_path))
            
            print("Verifying event store integrity...")
            is_valid = event_store.verify_integrity()
            
            stats = event_store.get_statistics()
            
            return {
                'success': True,
                'integrity_valid': is_valid,
                'statistics': stats,
                'message': 'Integrity verification passed' if is_valid else 'Integrity verification FAILED'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Integrity verification failed: {e}'
            }
    
    def rebuild_from_events(self, storage_path: str, use_snapshot: bool = True, 
                           memory_path: Optional[str] = None) -> Dict[str, Any]:
        """Perform deterministic rebuild from events."""
        try:
            # Create temporary memory instance for rebuild
            memory = LuminaMemory()
            
            print(f"Starting deterministic rebuild from {storage_path}")
            if use_snapshot:
                print("Using latest snapshot for optimization...")
            else:
                print("Rebuilding from all events...")
            
            rebuilder = create_deterministic_rebuilder(Path(storage_path), memory)
            
            if use_snapshot:
                rebuild_stats = rebuilder.rebuild_from_snapshot()
            else:
                rebuild_stats = rebuilder.rebuild_from_scratch()
            
            # Get final verification
            active_set = rebuilder.get_active_set_state()
            
            return {
                'success': True,
                'rebuild_statistics': rebuild_stats,
                'active_set_size': len(active_set),
                'message': f'Rebuild completed successfully in {rebuild_stats.get("rebuild_duration_ms", 0)}ms'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Rebuild failed: {e}'
            }
    
    def create_snapshot(self, storage_path: str) -> Dict[str, Any]:
        """Create new snapshot of current state."""
        try:
            # For CLI, we''ll create empty snapshot as example
            # In practice, this would snapshot current memory state
            event_store = create_event_store(Path(storage_path))
            
            # Create snapshot with empty state (placeholder)
            snapshot = event_store.create_snapshot(
                memory_records=[],
                active_set={}
            )
            
            return {
                'success': True,
                'snapshot_id': snapshot.snapshot_id,
                'timestamp': snapshot.timestamp,
                'message': f'Snapshot created: {snapshot.snapshot_id}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Snapshot creation failed: {e}'
            }
    
    def list_snapshots(self, storage_path: str) -> Dict[str, Any]:
        """List all snapshots."""
        try:
            event_store = create_event_store(Path(storage_path))
            
            # Get latest snapshot info
            latest = event_store.get_latest_snapshot()
            
            snapshots = []
            if latest:
                snapshots.append({
                    'snapshot_id': latest.snapshot_id,
                    'timestamp': latest.timestamp,
                    'last_event_hash': latest.last_event_hash,
                    'record_count': len(latest.memory_records)
                })
            
            return {
                'success': True,
                'snapshots': snapshots,
                'count': len(snapshots)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to list snapshots: {e}'
            }
    
    def export_events(self, storage_path: str, output_file: str, 
                     event_type: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Export events to JSON file."""
        try:
            event_store = create_event_store(Path(storage_path))
            
            # Get events
            if event_type:
                events = event_store.get_events_by_type(event_type, limit)
            else:
                # Get all events via rebuild iterator
                events = list(event_store.rebuild_from_events())
                if limit:
                    events = events[:limit]
            
            # Export to file
            export_data = {
                'export_timestamp': event_store.get_statistics()['latest_timestamp'],
                'event_count': len(events),
                'events': [event.to_dict() for event in events]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'events_exported': len(events),
                'output_file': output_file,
                'message': f'Exported {len(events)} events to {output_file}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Export failed: {e}'
            }
    
    def import_events(self, storage_path: str, input_file: str) -> Dict[str, Any]:
        """Import events from JSON file."""
        try:
            event_store = create_event_store(Path(storage_path))
            
            # Load import data
            with open(input_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_count = 0
            skipped_count = 0
            
            for event_data in import_data.get('events', []):
                try:
                    # Import event (will be idempotent based on content hash)
                    event_store.append_event(
                        event_type=event_data['event_type'],
                        content=event_data['content']
                    )
                    imported_count += 1
                except Exception:
                    skipped_count += 1
                    continue
            
            return {
                'success': True,
                'events_imported': imported_count,
                'events_skipped': skipped_count,
                'input_file': input_file,
                'message': f'Imported {imported_count} events, skipped {skipped_count}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Import failed: {e}'
            }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Event Store CLI')
    parser.add_argument('--storage-path', '-p', required=True,
                       help='Path to event store storage directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize new event store')
    init_parser.add_argument('--force', '-f', action='store_true',
                           help='Force initialization (overwrite existing)')
    
    # Stats command
    subparsers.add_parser('stats', help='Show event store statistics')
    
    # Verify command
    subparsers.add_parser('verify', help='Verify event store integrity')
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Perform deterministic rebuild')
    rebuild_parser.add_argument('--no-snapshot', action='store_true',
                              help='Rebuild from all events (skip snapshot optimization)')
    rebuild_parser.add_argument('--memory-path', '-m',
                              help='Path to memory storage (optional)')
    
    # Snapshot commands
    snapshot_parser = subparsers.add_parser('snapshot', help='Snapshot operations')
    snapshot_subparsers = snapshot_parser.add_subparsers(dest='snapshot_command')
    snapshot_subparsers.add_parser('create', help='Create new snapshot')
    snapshot_subparsers.add_parser('list', help='List all snapshots')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export events to file')
    export_parser.add_argument('--output', '-o', required=True,
                             help='Output JSON file path')
    export_parser.add_argument('--type', '-t',
                             help='Filter by event type')
    export_parser.add_argument('--limit', '-l', type=int,
                             help='Maximum number of events to export')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import events from file')
    import_parser.add_argument('--input', '-i', required=True,
                             help='Input JSON file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = EventStoreCLI()
    result = {}
    
    try:
        if args.command == 'init':
            result = cli.init_event_store(args.storage_path, args.force)
        
        elif args.command == 'stats':
            result = cli.get_statistics(args.storage_path)
        
        elif args.command == 'verify':
            result = cli.verify_integrity(args.storage_path)
        
        elif args.command == 'rebuild':
            result = cli.rebuild_from_events(
                storage_path=args.storage_path,
                use_snapshot=not args.no_snapshot,
                memory_path=args.memory_path
            )
        
        elif args.command == 'snapshot':
            if args.snapshot_command == 'create':
                result = cli.create_snapshot(args.storage_path)
            elif args.snapshot_command == 'list':
                result = cli.list_snapshots(args.storage_path)
            else:
                snapshot_parser.print_help()
                sys.exit(1)
        
        elif args.command == 'export':
            result = cli.export_events(
                storage_path=args.storage_path,
                output_file=args.output,
                event_type=args.type,
                limit=args.limit
            )
        
        elif args.command == 'import':
            result = cli.import_events(args.storage_path, args.input)
        
        else:
            parser.print_help()
            sys.exit(1)
        
        # Output result
        if result.get('success'):
            print(f" {result.get('message', 'Operation completed successfully')}")
            
            # Print additional info
            if 'statistics' in result:
                print(f"\nStatistics:")
                stats = result['statistics']
                print(f"  Total Events: {stats.get('total_events', 0)}")
                print(f"  Total Snapshots: {stats.get('total_snapshots', 0)}")
                print(f"  Hash Chain Length: {stats.get('hash_chain_length', 0)}")
                print(f"  Chain Verified: {stats.get('chain_verified', False)}")
                
                if stats.get('event_counts_by_type'):
                    print(f"  Event Counts by Type:")
                    for event_type, count in stats['event_counts_by_type'].items():
                        print(f"    {event_type}: {count}")
            
            if 'rebuild_statistics' in result:
                print(f"\nRebuild Statistics:")
                rebuild_stats = result['rebuild_statistics']
                for key, value in rebuild_stats.items():
                    if key != 'final_verification':
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            
            if 'snapshots' in result:
                print(f"\nSnapshots ({result['count']}):")
                for snapshot in result['snapshots']:
                    print(f"  ID: {snapshot['snapshot_id'][:16]}...")
                    print(f"  Timestamp: {snapshot['timestamp']}")
                    print(f"  Records: {snapshot['record_count']}")
                    print()
        
        else:
            print(f" {result.get('error', 'Operation failed')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f" Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()