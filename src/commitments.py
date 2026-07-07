import hashlib
import secrets

def commit(x: bytes) -> tuple[bytes, bytes]:
    if not isinstance(x, bytes):
        raise TypeError("Input must be bytes")
    
    h = hashlib.new('sha256')
    r = secrets.token_bytes(16)
    h.update(x)
    h.update(r)
    return r, h.digest()

def verify(x: bytes, r: bytes, c: bytes) -> bool:
    if not (isinstance(x, bytes) and isinstance(r, bytes) and isinstance(c, bytes)):
        raise TypeError("All inputs must be bytes")
    
    h = hashlib.new('sha256')
    h.update(x)
    h.update(r)
    return h.digest() == c