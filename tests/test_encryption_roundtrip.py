"""
Tests for AES-GCM encryption round-trip functionality.

These tests ensure that:
1. Encryption/decryption works correctly
2. Keys are generated properly
3. AAD (additional authenticated data) is verified
4. Tampering is detected
"""

import pytest
from lumina_memory.encryption import new_aesgcm_key, aesgcm_encrypt, aesgcm_decrypt

def test_encrypt_decrypt_roundtrip():
    """Basic encryption/decryption should work."""
    key = new_aesgcm_key()
    nonce, ct = aesgcm_encrypt(key, b"secret", aad=b"hdr")
    pt = aesgcm_decrypt(key, nonce, ct, aad=b"hdr")
    assert pt == b"secret"

def test_different_keys_fail():
    """Different keys should not decrypt successfully."""
    key1 = new_aesgcm_key()
    key2 = new_aesgcm_key()
    
    nonce, ct = aesgcm_encrypt(key1, b"secret", aad=b"hdr")
    
    with pytest.raises(Exception):  # Should raise authentication error
        aesgcm_decrypt(key2, nonce, ct, aad=b"hdr")

def test_wrong_aad_fails():
    """Wrong AAD should fail authentication."""
    key = new_aesgcm_key()
    nonce, ct = aesgcm_encrypt(key, b"secret", aad=b"correct_header")
    
    with pytest.raises(Exception):  # Should raise authentication error
        aesgcm_decrypt(key, nonce, ct, aad=b"wrong_header")

def test_tampered_ciphertext_fails():
    """Tampered ciphertext should fail authentication."""
    key = new_aesgcm_key()
    nonce, ct = aesgcm_encrypt(key, b"secret", aad=b"hdr")
    
    # Tamper with ciphertext
    tampered_ct = ct[:-1] + b"\x00"
    
    with pytest.raises(Exception):  # Should raise authentication error
        aesgcm_decrypt(key, nonce, tampered_ct, aad=b"hdr")

def test_nonce_reuse_different_ciphertext():
    """Same plaintext with different nonces should produce different ciphertext."""
    key = new_aesgcm_key()
    
    nonce1, ct1 = aesgcm_encrypt(key, b"secret", aad=b"hdr")
    nonce2, ct2 = aesgcm_encrypt(key, b"secret", aad=b"hdr")
    
    # Nonces should be different (random)
    assert nonce1 != nonce2
    # Ciphertext should be different
    assert ct1 != ct2
    
    # Both should decrypt correctly
    assert aesgcm_decrypt(key, nonce1, ct1, aad=b"hdr") == b"secret"
    assert aesgcm_decrypt(key, nonce2, ct2, aad=b"hdr") == b"secret"

def test_key_generation_unique():
    """Key generation should produce unique keys."""
    keys = [new_aesgcm_key() for _ in range(10)]
    
    # All keys should be different
    assert len(set(keys)) == 10
    
    # All keys should be 32 bytes for AES-256
    for key in keys:
        assert len(key) == 32

def test_empty_plaintext_roundtrip():
    """Empty plaintext should work."""
    key = new_aesgcm_key()
    nonce, ct = aesgcm_encrypt(key, b"", aad=b"empty_data")
    pt = aesgcm_decrypt(key, nonce, ct, aad=b"empty_data")
    assert pt == b""

def test_large_plaintext_roundtrip():
    """Large plaintext should work."""
    key = new_aesgcm_key()
    large_data = b"x" * 10000  # 10KB
    
    nonce, ct = aesgcm_encrypt(key, large_data, aad=b"large_file")
    pt = aesgcm_decrypt(key, nonce, ct, aad=b"large_file")
    assert pt == large_data

def test_no_aad_works():
    """Encryption without AAD should work."""
    key = new_aesgcm_key()
    nonce, ct = aesgcm_encrypt(key, b"secret")  # No AAD
    pt = aesgcm_decrypt(key, nonce, ct)  # No AAD
    assert pt == b"secret"

def test_mixed_aad_usage_fails():
    """Mixing AAD usage should fail."""
    key = new_aesgcm_key()
    nonce, ct = aesgcm_encrypt(key, b"secret", aad=b"header")
    
    with pytest.raises(Exception):  # Should fail - AAD mismatch
        aesgcm_decrypt(key, nonce, ct)  # No AAD provided
