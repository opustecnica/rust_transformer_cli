# Quick Reference - Using rust_transformer.dll

## Build Commands

```powershell
# Build everything
cargo build --release

# DLL only
cargo build --release --lib

# CLI only  
cargo build --release --bin rust_transformer_cli
```

## Output Files

- **DLL**: `target\release\rust_transformer.dll` (8.6 MB)
- **CLI**: `target\release\rust_transformer_cli.exe` (9.2 MB)
- **Lib**: `target\release\rust_transformer.dll.lib` (import library for C/C++)

## Basic Usage Pattern

### 1. Initialize
```c
EmbedderHandle handle = embedder_init("mini_lm_v2");
```

### 2. Embed Text
```c
float embedding[512];
size_t actual_size;
int result = embedder_embed(handle, "your text", embedding, 512, &actual_size);
```

### 3. Check Errors
```c
if (result != 0) {
    const char* error = embedder_get_last_error(handle);
    printf("Error: %s\n", error);
    embedder_free_error((char*)error);
}
```

### 4. Cleanup
```c
embedder_free(handle);
```

## Error Codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | Success | Operation succeeded |
| 1 | NullPointer | Invalid pointer passed |
| 2 | InvalidUtf8 | Text is not valid UTF-8 |
| 3 | InitializationFailed | Could not load model |
| 4 | EmbeddingFailed | Embedding generation failed |
| 5 | InvalidHandle | Handle is null or invalid |
| 6 | BufferTooSmall | Output buffer too small |

## Models

| Model Name | Embedding Size | Speed | Quality |
|------------|---------------|-------|---------|
| mini_lm_v2 | 384 | Fast | Good |
| jina | 768 | Slower | Better |

## Python Quick Start

```python
from ctypes import CDLL, c_char_p, c_void_p, c_float, c_size_t, POINTER, byref

# Load library
lib = CDLL("rust_transformer.dll")

# Setup function signatures
lib.embedder_init.restype = c_void_p
lib.embedder_embed.argtypes = [c_void_p, c_char_p, POINTER(c_float), c_size_t, POINTER(c_size_t)]

# Use it
handle = lib.embedder_init(b"mini_lm_v2")
buffer = (c_float * 512)()
size = c_size_t()
result = lib.embedder_embed(handle, b"Hello!", buffer, 512, byref(size))

if result == 0:
    embedding = list(buffer[:size.value])
    print(f"Success! Got {len(embedding)} dimensions")

lib.embedder_free(handle)
```

## C# Quick Start

```csharp
[DllImport("rust_transformer.dll")]
private static extern IntPtr embedder_init(string model);

[DllImport("rust_transformer.dll")]
private static extern int embedder_embed(
    IntPtr handle, string text, 
    float[] buffer, UIntPtr size, out UIntPtr actual);

[DllImport("rust_transformer.dll")]
private static extern void embedder_free(IntPtr handle);

// Use it
var handle = embedder_init("mini_lm_v2");
var buffer = new float[512];
UIntPtr actual;
int result = embedder_embed(handle, "Hello!", buffer, (UIntPtr)512, out actual);
Console.WriteLine($"Got {actual} dimensions");
embedder_free(handle);
```

## PowerShell Quick Start

```powershell
# Run the complete example script
.\examples\example_powershell.ps1

# Or use directly with Add-Type:
$signature = @"
using System;
using System.Runtime.InteropServices;
namespace RT {
    public static class N {
        [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr embedder_init(string model);
        [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int embedder_embed(IntPtr h, string t, float[] b, UIntPtr s, out UIntPtr a);
        [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void embedder_free(IntPtr handle);
    }
}
"@
Add-Type -TypeDefinition $signature

$h = [RT.N]::embedder_init("mini_lm_v2")
$b = New-Object float[] 512
$a = [UIntPtr]::Zero
[RT.N]::embedder_embed($h, "Hello!", $b, [UIntPtr]512, [ref]$a)
Write-Host "Got $a dimensions"
[RT.N]::embedder_free($h)
```

## Tips

- **Reuse handles**: Initialize once, embed many times (faster)
- **Buffer size**: 512 is safe for all models
- **Thread safety**: Create one handle per thread
- **First run**: Will download models (~90MB for mini_lm_v2)
- **Batch mode**: Use `embedder_embed_batch()` for multiple texts

## Files to Distribute

Minimum required:
- `rust_transformer.dll`

The DLL will automatically download models on first use.

## See Also

- `DLL_USAGE.md` - Complete API reference
- `../examples/example_python.py` - Full Python example with wrapper class
- `../examples/example_csharp.cs` - Full C# example
- `../examples/example_powershell.ps1` - Full PowerShell example with Add-Type
- `../examples/example_c.c` - Full C example
