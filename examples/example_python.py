"""
Why would you do this through a rust dll vs directly in Python? Because I can :-)

example_python.py - Example of using rust_transformer.dll from Python
Requires: Python 3.x with ctypes (built-in)

Usage: python example_python.py
Make sure rust_transformer.dll is in the same directory or in PATH
"""

import ctypes
import platform
from ctypes import c_char_p, c_void_p, c_float, c_size_t, POINTER, byref
from pathlib import Path

# Determine DLL name based on platform
if platform.system() == "Windows":
    LIB_NAME = "rust_transformer.dll"
elif platform.system() == "Darwin":
    LIB_NAME = "librust_transformer.dylib"
else:
    LIB_NAME = "librust_transformer.so"

# Try to find the DLL
dll_paths = [
    Path(LIB_NAME),  # Current directory
    Path("../target/release") / LIB_NAME,  # Relative to examples folder
    Path(__file__).parent.parent / "target" / "release" / LIB_NAME,
]

lib = None
for path in dll_paths:
    if path.exists():
        lib = ctypes.CDLL(str(path))
        print(f"Loaded library from: {path}")
        break

if lib is None:
    print(f"Could not find {LIB_NAME}. Please build the library first:")
    print("  cargo build --release --lib")
    exit(1)

# Error codes
class EmbedderErrorCode:
    SUCCESS = 0
    NULL_POINTER = 1
    INVALID_UTF8 = 2
    INITIALIZATION_FAILED = 3
    EMBEDDING_FAILED = 4
    INVALID_HANDLE = 5
    BUFFER_TOO_SMALL = 6
    
    ERROR_NAMES = {
        0: "Success",
        1: "Null Pointer",
        2: "Invalid UTF-8",
        3: "Initialization Failed",
        4: "Embedding Failed",
        5: "Invalid Handle",
        6: "Buffer Too Small"
    }
    
    @classmethod
    def get_name(cls, code):
        return cls.ERROR_NAMES.get(code, f"Unknown ({code})")


# Define function signatures
lib.embedder_version.argtypes = []
lib.embedder_version.restype = c_char_p

lib.embedder_init.argtypes = [c_char_p]
lib.embedder_init.restype = c_void_p

lib.embedder_embed.argtypes = [
    c_void_p,  # handle
    c_char_p,  # text
    POINTER(c_float),  # output_buffer
    c_size_t,  # buffer_size
    POINTER(c_size_t)  # actual_size
]
lib.embedder_embed.restype = ctypes.c_int

lib.embedder_embed_batch.argtypes = [
    c_void_p,  # handle
    POINTER(c_char_p),  # texts
    c_size_t,  # num_texts
    POINTER(c_float),  # output_buffer
    c_size_t,  # buffer_size
    POINTER(c_size_t),  # embedding_dim
    POINTER(c_size_t)  # total_written
]
lib.embedder_embed_batch.restype = ctypes.c_int

lib.embedder_get_last_error.argtypes = [c_void_p]
lib.embedder_get_last_error.restype = c_char_p

lib.embedder_free_error.argtypes = [c_void_p]
lib.embedder_free_error.restype = None

lib.embedder_free.argtypes = [c_void_p]
lib.embedder_free.restype = None


class RustEmbedder:
    """Python wrapper for the Rust transformer embedder DLL"""
    
    def __init__(self, model_name="mini_lm_v2"):
        """Initialize the embedder with the specified model"""
        self.handle = lib.embedder_init(model_name.encode('utf-8'))
        if not self.handle:
            raise RuntimeError(f"Failed to initialize embedder with model: {model_name}")
    
    def embed(self, text, buffer_size=512):
        """Generate embedding for a single text string"""
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        
        # Prepare buffers
        output_buffer = (c_float * buffer_size)()
        actual_size = c_size_t()
        
        # Call the function
        result = lib.embedder_embed(
            self.handle,
            text.encode('utf-8'),
            output_buffer,
            buffer_size,
            byref(actual_size)
        )
        
        if result != EmbedderErrorCode.SUCCESS:
            error_msg = self.get_last_error()
            raise RuntimeError(
                f"Embedding failed: {EmbedderErrorCode.get_name(result)}"
                + (f" - {error_msg}" if error_msg else "")
            )
        
        # Convert to Python list
        return list(output_buffer[:actual_size.value])
    
    def embed_batch(self, texts, buffer_size_per_text=512):
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        num_texts = len(texts)
        total_buffer_size = buffer_size_per_text * num_texts
        
        # Prepare C string array
        c_texts = (c_char_p * num_texts)()
        for i, text in enumerate(texts):
            c_texts[i] = text.encode('utf-8')
        
        # Prepare output buffer
        output_buffer = (c_float * total_buffer_size)()
        embedding_dim = c_size_t()
        total_written = c_size_t()
        
        # Call the function
        result = lib.embedder_embed_batch(
            self.handle,
            c_texts,
            num_texts,
            output_buffer,
            total_buffer_size,
            byref(embedding_dim),
            byref(total_written)
        )
        
        if result != EmbedderErrorCode.SUCCESS:
            error_msg = self.get_last_error()
            raise RuntimeError(
                f"Batch embedding failed: {EmbedderErrorCode.get_name(result)}"
                + (f" - {error_msg}" if error_msg else "")
            )
        
        # Convert to list of embeddings
        dim = embedding_dim.value
        embeddings = []
        for i in range(num_texts):
            start = i * dim
            end = start + dim
            embeddings.append(list(output_buffer[start:end]))
        
        return embeddings
    
    def get_last_error(self):
        """Get the last error message"""
        error_ptr = lib.embedder_get_last_error(self.handle)
        if error_ptr:
            error_msg = error_ptr.decode('utf-8')
            lib.embedder_free_error(error_ptr)
            return error_msg
        return None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'handle') and self.handle:
            lib.embedder_free(self.handle)


def main():
    print("Rust Transformer DLL Example (Python)")
    version = lib.embedder_version().decode('utf-8')
    print(f"Version: {version}\n")
    
    # Single embedding example
    print("=== Single Embedding Example ===")
    embedder = RustEmbedder("mini_lm_v2")
    
    text = "Hello, world! This is a test from Python."
    print(f"Input text: {text}\n")
    
    print("Generating embedding...")
    embedding = embedder.embed(text)
    
    print(f"Embedding generated successfully!")
    print(f"Embedding dimension: {len(embedding)}\n")
    
    print("First 10 values:")
    for i, val in enumerate(embedding[:10]):
        print(f"[{i}]: {val:.6f}")
    
    # Calculate L2 norm
    import math
    magnitude = math.sqrt(sum(x*x for x in embedding))
    print(f"\nL2 Norm: {magnitude:.6f}")
    
    # Batch embedding example
    print("\n=== Batch Embedding Example ===")
    texts = [
        "First sentence.",
        "Second sentence is longer.",
        "The third one."
    ]
    
    print(f"Processing {len(texts)} texts:")
    for i, t in enumerate(texts):
        print(f"  [{i}]: {t}")
    
    print("\nGenerating embeddings...")
    embeddings = embedder.embed_batch(texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has dimension: {len(embeddings[0])}")
    
    print("\nFirst 5 values of each embedding:")
    for i, emb in enumerate(embeddings):
        values_str = ", ".join(f"{v:.4f}" for v in emb[:5])
        print(f"  Text {i}: [{values_str}, ...]")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
