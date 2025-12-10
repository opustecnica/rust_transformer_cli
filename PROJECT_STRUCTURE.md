# Project Structure

```
rust_transformer_cli/
├── README.md              # Main project documentation
├── LICENSE                # License file
├── Cargo.toml            # Rust package manifest
│
├── src/                  # Source code
│   ├── main.rs          # CLI application entry point
│   ├── lib.rs           # DLL/library exports (FFI)
│   └── embed_utils/     # Core embedding functionality
│
├── docs/                 # Documentation
│   ├── DLL_USAGE.md              # Complete API reference
│   ├── QUICK_REFERENCE.md        # Quick start guide
│   ├── IMPLEMENTATION_SUMMARY.md # Technical details
│   └── rust_transformer.h        # C/C++ header file
│
├── examples/             # Usage examples
│   ├── README.md                 # Examples overview
│   ├── example_c.c              # C language example
│   ├── example_csharp.cs        # C# example
│   ├── example_python.py        # Python example
│   └── example_powershell.ps1   # PowerShell example
│
└── target/               # Build outputs
    └── release/
        ├── rust_transformer_cli.exe  # CLI executable
        ├── rust_transformer.dll      # Native library (Windows)
        └── rust_transformer.dll.lib  # Import library
```

## Quick Links

### Building
```bash
cargo build --release              # Build both CLI and DLL
cargo build --release --bin rust_transformer_cli  # CLI only
cargo build --release --lib        # DLL only
```

### Documentation
- [Main README](README.md) - Project overview and quick start
- [API Reference](docs/DLL_USAGE.md) - Complete FFI API documentation
- [Quick Reference](docs/QUICK_REFERENCE.md) - Code snippets for common tasks
- [C/C++ Header](docs/rust_transformer.h) - Header file for C/C++ integration

### Examples
All examples are in the `examples/` directory:
- **C**: `example_c.c` - Native C integration
- **C#**: `example_csharp.cs` - .NET P/Invoke usage
- **Python**: `example_python.py` - ctypes wrapper with helper class
- **PowerShell**: `example_powershell.ps1` - Add-Type P/Invoke with 3 usage patterns

### Running Examples
```bash
# Python
python examples/example_python.py

# PowerShell
powershell -File examples/example_powershell.ps1

# C (compile first)
gcc examples/example_c.c -o example_c -L./target/release -lrust_transformer

# C# (compile first)
csc examples/example_csharp.cs
```

## Files by Purpose

### User-Facing
- `README.md` - Start here
- `LICENSE` - License terms
- `examples/` - Working code samples
- `docs/` - Detailed documentation

### Development
- `src/` - Source code
- `Cargo.toml` - Build configuration
- `target/` - Build artifacts (generated)

### Distribution
Required files to distribute:
- `rust_transformer.dll` (or .so/.dylib on Linux/macOS)
- Optional: `rust_transformer.h` for C/C++ users
- Optional: Examples for reference

Models (~90MB-500MB) are downloaded automatically on first use.
