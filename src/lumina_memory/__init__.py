"""
Lumina Memory System - Identity and Integrity Rails

Core modules for content-addressed memory storage with cryptographic integrity.
"""

from .crypto_ids import content_fingerprint, verify_fingerprint, generate_content_id
from .event_hashing import event_hash, verify_chain, create_genesis_event, create_chained_event
from .encryption import (
    new_aesgcm_key, aesgcm_encrypt, aesgcm_decrypt,
    derive_kek, generate_dek, create_envelope, open_envelope
)
from .hrr import reference_vector, bind_vectors, similarity
# from .events import Event, create_ingest_event, create_conflict_event  # TODO: events.py missing

__version__ = "0.4.0"
__all__ = [
    "content_fingerprint", "verify_fingerprint", "generate_content_id",
    "event_hash", "verify_chain", "create_genesis_event", "create_chained_event", 
    "new_aesgcm_key", "aesgcm_encrypt", "aesgcm_decrypt",
    "derive_kek", "generate_dek", "create_envelope", "open_envelope",
    "reference_vector", "bind_vectors", "similarity",
    # "Event", "create_ingest_event", "create_conflict_event"  # TODO: events.py missing
]
