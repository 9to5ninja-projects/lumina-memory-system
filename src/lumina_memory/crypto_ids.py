"""
Content-Addressable Cryptographic IDs for Lumina Memory

This module provides deterministic, cryptographically secure ID generation
for memory objects based on their content. IDs are SHA-256 hashes that enable
content deduplication and tamper detection.

Design Principles:
- Deterministic: Same content  Same ID
- Collision-resistant: SHA-256 cryptographic guarantees
- Content-addressable: ID directly relates to content
- Future-ready: Extensible for post-M12 holographic distribution
"""

import hashlib
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict
import numpy as np


def normalize_content(content: Any) -> str:
    """
    Normalize content to canonical string for consistent hashing.
    
    Handles various content types and produces deterministic string representation.
    """
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, (int, float, bool)):
        return str(content)
    elif isinstance(content, (list, tuple)):
        return json.dumps(sorted(content) if all(isinstance(x, (str, int, float)) for x in content) else list(content), 
                         sort_keys=True, separators=(',', ':'))
    elif isinstance(content, dict):
        return json.dumps(content, sort_keys=True, separators=(',', ':'))
    elif isinstance(content, np.ndarray):
        return content.tobytes().hex()
    else:
        return str(content)


def compute_content_hash(
    content: str,
    embedding: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Compute SHA-256 hash of memory content for content-addressable ID.
    
    Args:
        content: Primary memory content
        embedding: Optional embedding vector
        metadata: Optional metadata dictionary  
        additional_data: Optional additional data for hash computation
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    hasher = hashlib.sha256()
    
    # Add normalized content
    hasher.update(normalize_content(content).encode('utf-8'))
    
    # Add embedding if provided
    if embedding is not None:
        hasher.update(embedding.tobytes())
    
    # Add metadata if provided
    if metadata:
        metadata_str = normalize_content(metadata)
        hasher.update(metadata_str.encode('utf-8'))
    
    # Add additional data if provided
    if additional_data:
        additional_str = normalize_content(additional_data)
        hasher.update(additional_str.encode('utf-8'))
    
    return hasher.hexdigest()


def memory_content_id(
    content: str,
    embedding: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    schema_version: str = "v1.0"
) -> str:
    """
    Generate content-addressable ID for a memory object.
    
    This is the primary function for generating memory IDs based on content.
    The ID is deterministic and collision-resistant.
    
    Args:
        content: Memory content string
        embedding: Embedding vector
        metadata: Optional metadata
        schema_version: Schema version for future compatibility
        
    Returns:
        Content-addressable memory ID
    """
    additional_data = {
        'schema_version': schema_version,
        'content_type': 'memory'
    }
    
    return compute_content_hash(
        content=content,
        embedding=embedding,
        metadata=metadata,
        additional_data=additional_data
    )


def composite_memory_id(memory_ids: List[str], operation: str = "superpose") -> str:
    """
    Generate ID for composite memory created from multiple source memories.
    
    Args:
        memory_ids: List of source memory IDs
        operation: Type of operation (superpose, consolidate, etc.)
        
    Returns:
        Content-addressable ID for composite memory
    """
    # Sort IDs for deterministic ordering
    sorted_ids = sorted(memory_ids)
    
    composite_data = {
        'source_ids': sorted_ids,
        'operation': operation,
        'content_type': 'composite_memory'
    }
    
    return compute_content_hash("", additional_data=composite_data)


def event_content_id(event_data: Dict[str, Any]) -> str:
    """
    Generate content-addressable ID for an event.
    
    Args:
        event_data: Event data dictionary
        
    Returns:
        Content-addressable event ID
    """
    # Remove non-content fields for ID computation
    content_fields = {
        'event_type': event_data.get('event_type'),
        'payload': event_data.get('payload'),
        'timestamp': event_data.get('timestamp'),
        'sequence': event_data.get('sequence')
    }
    
    additional_data = {
        'content_type': 'event',
        'schema_version': event_data.get('schema_version', 'v1.0')
    }
    
    return compute_content_hash("", additional_data={**content_fields, **additional_data})


def verify_content_integrity(
    content_id: str,
    content: str,
    embedding: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    schema_version: str = "v1.0"
) -> bool:
    """
    Verify that content matches its content-addressable ID.
    
    Args:
        content_id: Expected content ID
        content: Content to verify
        embedding: Optional embedding
        metadata: Optional metadata
        schema_version: Schema version
        
    Returns:
        True if content matches ID, False otherwise
    """
    computed_id = memory_content_id(content, embedding, metadata, schema_version)
    return computed_id == content_id


class ContentAddressableIndex:
    """
    Index for managing content-addressable memory objects.
    
    Ensures uniqueness and handles deduplication based on content IDs.
    """
    
    def __init__(self):
        self.index: Dict[str, Dict[str, Any]] = {}
        self.access_count: Dict[str, int] = {}
        
    def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        schema_version: str = "v1.0"
    ) -> tuple[str, bool]:
        """
        Add memory to content-addressable index.
        
        Returns:
            Tuple of (content_id, is_duplicate)
        """
        content_id = memory_content_id(content, embedding, metadata, schema_version)
        
        if content_id in self.index:
            # Content already exists - increment access count
            self.access_count[content_id] = self.access_count.get(content_id, 0) + 1
            return content_id, True
        else:
            # New content - add to index
            self.index[content_id] = {
                'content': content,
                'embedding': embedding,
                'metadata': metadata or {},
                'schema_version': schema_version
            }
            self.access_count[content_id] = 1
            return content_id, False
    
    def get_memory(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by content ID."""
        return self.index.get(content_id)
    
    def verify_index_integrity(self) -> List[str]:
        """
        Verify integrity of all entries in the index.
        
        Returns:
            List of content IDs with integrity violations
        """
        violations = []
        
        for content_id, memory_data in self.index.items():
            if not verify_content_integrity(
                content_id,
                memory_data['content'],
                memory_data.get('embedding'),
                memory_data.get('metadata'),
                memory_data.get('schema_version', 'v1.0')
            ):
                violations.append(content_id)
        
        return violations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_unique_memories': len(self.index),
            'total_access_count': sum(self.access_count.values()),
            'average_access_per_memory': sum(self.access_count.values()) / len(self.index) if self.index else 0,
            'most_accessed_memory': max(self.access_count, key=self.access_count.get) if self.access_count else None
        }


# Global content-addressable index instance
global_content_index = ContentAddressableIndex()

 
 
 d e f   d e t e c t _ c o n t e n t _ c o l l i s i o n ( 
         c o n t e n t _ i d :   s t r , 
         e x i s t i n g _ c o n t e n t :   D i c t [ s t r ,   A n y ] , 
         n e w _ c o n t e n t :   D i c t [ s t r ,   A n y ] 
 )   - >   O p t i o n a l [ D i c t [ s t r ,   A n y ] ] : 
         " " " 
         D e t e c t   p o t e n t i a l   c o n t e n t   c o l l i s i o n   f o r   s a m e   I D . 
         
         A r g s : 
                 c o n t e n t _ i d :   C o n t e n t   I D   b e i n g   c h e c k e d 
                 e x i s t i n g _ c o n t e n t :   E x i s t i n g   c o n t e n t   w i t h   t h i s   I D 
                 n e w _ c o n t e n t :   N e w   c o n t e n t   a t t e m p t i n g   t o   u s e   t h i s   I D 
                 
         R e t u r n s : 
                 C o l l i s i o n   d e t a i l s   i f   d e t e c t e d ,   N o n e   o t h e r w i s e 
         " " " 
         #   V e r i f y   b o t h   c o n t e n t s   a c t u a l l y   p r o d u c e   t h i s   I D 
         e x i s t i n g _ v a l i d   =   v e r i f y _ c o n t e n t _ i n t e g r i t y ( 
                 c o n t e n t _ i d , 
                 e x i s t i n g _ c o n t e n t [ " c o n t e n t " ] , 
                 e x i s t i n g _ c o n t e n t . g e t ( " e m b e d d i n g " ) , 
                 e x i s t i n g _ c o n t e n t . g e t ( " m e t a d a t a " ) 
         ) 
         
         n e w _ v a l i d   =   v e r i f y _ c o n t e n t _ i n t e g r i t y ( 
                 c o n t e n t _ i d , 
                 n e w _ c o n t e n t [ " c o n t e n t " ] ,   
                 n e w _ c o n t e n t . g e t ( " e m b e d d i n g " ) , 
                 n e w _ c o n t e n t . g e t ( " m e t a d a t a " ) 
         ) 
         
         i f   n o t   e x i s t i n g _ v a l i d   o r   n o t   n e w _ v a l i d : 
                 r e t u r n   { 
                         " t y p e " :   " i n v a l i d _ i d " , 
                         " c o n t e n t _ i d " :   c o n t e n t _ i d , 
                         " e x i s t i n g _ v a l i d " :   e x i s t i n g _ v a l i d , 
                         " n e w _ v a l i d " :   n e w _ v a l i d 
                 } 
         
         #   C h e c k   i f   c o n t e n t s   a r e   a c t u a l l y   i d e n t i c a l   ( s h o u l d   b e   f o r   s a m e   h a s h ) 
         i f   e x i s t i n g _ c o n t e n t   ! =   n e w _ c o n t e n t : 
                 r e t u r n   { 
                         " t y p e " :   " h a s h _ c o l l i s i o n " , 
                         " c o n t e n t _ i d " :   c o n t e n t _ i d , 
                         " e x i s t i n g _ c o n t e n t " :   e x i s t i n g _ c o n t e n t , 
                         " n e w _ c o n t e n t " :   n e w _ c o n t e n t 
                 } 
         
         #   C o n t e n t s   a r e   i d e n t i c a l   -   t h i s   i s   e x p e c t e d   d e d u p l i c a t i o n 
         r e t u r n   N o n e 
  
 