# rust_transformer_cli

A high-performance text embedding tool available as both a **CLI application** and a **native DLL/shared library** for integration with other programming languages.

## Dual-Mode Architecture

This project provides two ways to use transformer-based text embeddings:

1. **Command-Line Interface** - Standalone executable for quick embeddings
2. **Native Library (DLL)** - C-compatible API for integration with C, C++, C#, Python, Java, etc.

## Installation & Building

### Build Everything
```bash
cargo build --release
```

Outputs:
- CLI: `target/release/rust_transformer_cli.exe` 
- DLL: `target/release/rust_transformer.dll`

### Build CLI Only
```bash
cargo build --release --bin rust_transformer_cli
```

### Build DLL Only
```bash
cargo build --release --lib
```

---

## CLI Usage
```bash
cli sentence transformer

Author: OpusTecnica <https://github.com/opustecnica/rust_transformer_cli>
Version: 0.3.0

CREDITS: Shamelessly inspired by the work of Martin Contreras Uribe <https://github.com/martin-conur>

Usage: rust_transformer_cli.exe [OPTIONS] --text <TEXT>

Options:
  -t, --text <TEXT>
          The input text to be transformed into an embedding

  -m, --model <MODEL>
          The transformer model to use (e.g., 'mini_lm_v2' or 'jina')

          [default: mini_lm_v2]

  -p, --pretty
          Output pretty-printed JSON instead of compact JSON

  -v, --verbose
          Enable verbose output with additional information

  -j, --json-input
          Treat input text as a JSON array of strings

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version

EXAMPLES:
  # Generate embedding with default model (mini_lm_v2) as compact JSON:
  rust_transformer_cli --text "Hello world"

  # Use a specific model:
  rust_transformer_cli --text "Hello world" --model jina

  # Output pretty-printed JSON:
  rust_transformer_cli --text "Hello world" --pretty

  # Process multiple texts from JSON array:
  rust_transformer_cli --text '["Hello world", "Goodbye"]' --json-input --pretty

  # Process multiple texts (compact output):
  rust_transformer_cli --text '["Hello world", "Goodbye"]' --json-input

  # Combine model and pretty output:
  rust_transformer_cli --text "Hello world" --model jina --pretty
```

---

## Library (DLL) Usage

### Quick Example (Python)
```python
from ctypes import CDLL, c_char_p, c_void_p, c_float, c_size_t, byref

lib = CDLL("rust_transformer.dll")

# Initialize
handle = lib.embedder_init(b"mini_lm_v2")

# Generate embedding  
buffer = (c_float * 512)()
size = c_size_t()
result = lib.embedder_embed(handle, b"Hello world", buffer, 512, byref(size))

if result == 0:  # Success
    embedding = list(buffer[:size.value])
    print(f"Embedding: {embedding[:5]}...")  # First 5 values

# Cleanup
lib.embedder_free(handle)
```

### Quick Example (C#)
```csharp
[DllImport("rust_transformer.dll")]
private static extern IntPtr embedder_init(string model);

[DllImport("rust_transformer.dll")]
private static extern int embedder_embed(
    IntPtr handle, string text, float[] buffer, 
    UIntPtr size, out UIntPtr actual);

[DllImport("rust_transformer.dll")]
private static extern void embedder_free(IntPtr handle);

// Usage
var handle = embedder_init("mini_lm_v2");
var buffer = new float[512];
UIntPtr actual;
embedder_embed(handle, "Hello world", buffer, (UIntPtr)512, out actual);
embedder_free(handle);
```

### API Functions

| Function | Purpose |
|----------|---------|
| `embedder_init(model)` | Initialize embedder |
| `embedder_embed(...)` | Generate single embedding |
| `embedder_embed_batch(...)` | Generate multiple embeddings |
| `embedder_get_last_error(handle)` | Get error message |
| `embedder_free_error(error)` | Free error string |
| `embedder_free(handle)` | Cleanup resources |
| `embedder_version()` | Get library version |

### Documentation

- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Fast start guide with code snippets
- **[docs/DLL_USAGE.md](docs/DLL_USAGE.md)** - Complete API reference
- **[docs/rust_transformer.h](docs/rust_transformer.h)** - C/C++ header file
- **[examples/](examples/)** - Full working examples in C, C#, Python, and PowerShell

---

## Supported Models

| Model | Embedding Size | Speed | Quality | Size |
|-------|---------------|-------|---------|------|
| `mini_lm_v2` | 384 | Fast | Good | ~90MB |
| `jina` | 768 | Slower | Better | ~500MB |

Models are automatically downloaded from HuggingFace on first use.

---

## Features

- **Zero-copy operations** where possible
- **Batch processing** support for multiple texts
- **Thread-safe** (use one handle per thread)
- **Memory-safe** FFI with comprehensive error handling
- **Cross-platform** (Windows DLL, Linux .so, macOS .dylib)
- **No runtime dependencies** - models download automatically

---

## Credits

Shamelessly inspired by the work of [Martin Contreras Uribe](https://github.com/martin-conur)

---

## License

See [LICENSE](LICENSE) file for details.

---

## Links

- Repository: https://github.com/opustecnica/rust_transformer_cli
- Version: 0.4.0
