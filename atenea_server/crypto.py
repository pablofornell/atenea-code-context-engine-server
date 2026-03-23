"""
Application-level encryption for Atenea using AES-256-GCM.

Both client and server share a secret key via the ATENEA_SECRET environment variable.
When set, all HTTP request/response bodies are encrypted.

Wire format: base64( nonce[12] || ciphertext || tag[16] )
"""

import os
import base64
import hashlib
import logging

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# Header used to signal that the payload is encrypted
ENCRYPTED_HEADER = "X-Atenea-Encrypted"


def get_secret() -> bytes | None:
    """
    Return the 256-bit AES key derived from ATENEA_SECRET, or None if not set.
    """
    raw = os.environ.get("ATENEA_SECRET")
    if not raw:
        return None
    # Derive a fixed 32-byte key from the secret using SHA-256
    return hashlib.sha256(raw.encode("utf-8")).digest()


def encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt plaintext with AES-256-GCM.
    Returns base64-encoded bytes: nonce(12) || ciphertext || tag(16).
    """
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)  # ciphertext includes tag
    return base64.b64encode(nonce + ciphertext)


def decrypt(token: bytes, key: bytes) -> bytes:
    """
    Decrypt a base64-encoded AES-256-GCM token.
    Token format: base64( nonce[12] || ciphertext || tag[16] )
    """
    raw = base64.b64decode(token)
    if len(raw) < 12 + 16:
        raise ValueError("Encrypted payload too short")
    nonce = raw[:12]
    ciphertext = raw[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)

