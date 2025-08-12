"""
Content fingerprinting using BLAKE3 for content-addressed IDs.

This module provides content_fingerprint() with canonicalization for 
deterministic content addressing and deduplication.
"""

import hashlib
import hmac
import json
import re
from typing import Dict, Any, Optional, Union

try:
    import blake3
except ImportError:
    # Fallback to hashlib implementation if blake3 not available
    blake3 = None

def _normalize_text(text: str) -> str:
    """Normalize text content for consistent fingerprinting."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace (collapse multiple spaces/newlines to single spaces)
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def _canonicalize_dict(data: Dict[str, Any]) -> str:
    """Canonicalize dictionary to JSON string with sorted keys."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)

def _blake3_hash(data: bytes, key: Optional[bytes] = None) -> str:
    """Hash data using BLAKE3 (with optional keyed mode)."""
    if blake3:
        if key:
            hasher = blake3.blake3(key=key)
        else:
            hasher = blake3.blake3()
        hasher.update(data)
        return hasher.hexdigest()
    else:
        # Fallback to SHA-256 if BLAKE3 not available
        if key:
            return hmac.new(key, data, hashlib.sha256).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()

def content_fingerprint(
    content: str,
    source: str,
    version: str,
    metadata: Union[str, Dict[str, Any]],
    secret_salt: Optional[bytes] = None
) -> Dict[str, str]:
    """
    Generate content-addressed fingerprint for memory content.
    
    Args:
        content: Raw content text
        source: Source identifier  
        version: Embedding model version
        metadata: Metadata dict or string
        secret_salt: Optional salt for private ID generation (32 bytes for BLAKE3)
    
    Returns:
        Dict with "public_cid" and "private_id" keys
    """
    # Normalize content
    normalized_content = _normalize_text(content)
    
    # Parse metadata if string
    if isinstance(metadata, str):
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            parsed_metadata = {"raw": metadata}
    else:
        parsed_metadata = metadata or {}
    
    # Create canonical representation for public CID
    public_data = {
        "content": normalized_content,
        "source": source,
        "version": version,
        "metadata": parsed_metadata
    }
    
    canonical_json = _canonicalize_dict(public_data)
    public_bytes = canonical_json.encode('utf-8')
    
    # Generate public content ID (no key)
    public_cid = _blake3_hash(public_bytes)
    
    # Generate private ID using HMAC/keyed mode if salt provided
    if secret_salt:
        private_id = _blake3_hash(public_bytes, key=secret_salt)
    else:
        private_id = public_cid
    
    return {
        "public_cid": public_cid,
        "private_id": private_id
    }

def verify_fingerprint(
    content: str,
    source: str, 
    version: str,
    metadata: Union[str, Dict[str, Any]],
    expected_public_cid: str,
    secret_salt: Optional[bytes] = None
) -> bool:
    """Verify that content matches expected fingerprint."""
    fp = content_fingerprint(content, source, version, metadata, secret_salt)
    return fp["public_cid"] == expected_public_cid

def generate_content_id(content_dict: Dict[str, Any]) -> str:
    """Generate content ID from arbitrary dictionary data."""
    canonical_json = _canonicalize_dict(content_dict)
    content_bytes = canonical_json.encode('utf-8')
    return _blake3_hash(content_bytes)
