# Using rust_transformer as a DLL/Shared Library

This project now supports building as both a CLI application and a native DLL (`.dll` on Windows, `.so` on Linux, `.dylib` on macOS) that can be used from other programming languages.

## Building

### Build the CLI only (binary)
```powershell
cargo build --release --bin rust_transformer_cli
```
Output: `target\release\rust_transformer_cli.exe`

### Build the DLL only
```powershell
cargo build --release --lib
```
Output: `target\release\rust_transformer.dll`

### Build both CLI and DLL
```powershell
cargo build --release
```
Outputs both the executable and the DLL.

## API Reference

The DLL exposes a C-compatible API with the following functions:

### Initialization

#### `embedder_init`
```c
EmbedderHandle embedder_init(const char* model_name);
```
Initialize an embedder instance with the specified model.

**Parameters:**
- `model_name`: Model to use ("mini_lm_v2" or "jina")

**Returns:** Handle to the embedder, or NULL on failure

**Example:**
```c
EmbedderHandle handle = embedder_init("mini_lm_v2");
```

---

### Single Text Embedding

#### `embedder_embed`
```c
EmbedderErrorCode embedder_embed(
    EmbedderHandle handle,
    const char* text,
    float* output_buffer,
    size_t buffer_size,
    size_t* actual_size
);
```
Generate an embedding for a single text string.

**Parameters:**
- `handle`: Embedder handle from `embedder_init()`
- `text`: Input text (null-terminated UTF-8 string)
- `output_buffer`: Pre-allocated buffer for embedding floats
- `buffer_size`: Size of output_buffer (number of float elements)
- `actual_size`: Output - actual embedding dimension (typically 384 for mini_lm_v2, 768 for jina)

**Returns:** Error code (0 = Success)

**Example:**
```c
float embedding[512];
size_t actual_size;
EmbedderErrorCode err = embedder_embed(
    handle, 
    "Hello world", 
    embedding, 
    512, 
    &actual_size
);
```

---

### Batch Embedding

#### `embedder_embed_batch`
```c
EmbedderErrorCode embedder_embed_batch(
    EmbedderHandle handle,
    const char** texts,
    size_t num_texts,
    float* output_buffer,
    size_t buffer_size,
    size_t* embedding_dim,
    size_t* total_written
);
```
Generate embeddings for multiple texts in a single call.

**Parameters:**
- `handle`: Embedder handle
- `texts`: Array of text pointers (null-terminated UTF-8 strings)
- `num_texts`: Number of texts in the array
- `output_buffer`: Buffer for all embeddings (flattened)
- `buffer_size`: Total buffer size in floats
- `embedding_dim`: Output - dimension of each embedding
- `total_written`: Output - total floats written

**Returns:** Error code (0 = Success)

**Note:** Output buffer contains embeddings sequentially: [emb1, emb2, emb3, ...]

---

### Error Handling

#### `embedder_get_last_error`
```c
const char* embedder_get_last_error(EmbedderHandle handle);
```
Get the last error message from the embedder.

**Returns:** C string with error message, or NULL if no error

**Note:** The returned string must be freed with `embedder_free_error()`

#### `embedder_free_error`
```c
void embedder_free_error(char* error_str);
```
Free an error string returned by `embedder_get_last_error()`.

---

### Cleanup

#### `embedder_free`
```c
void embedder_free(EmbedderHandle handle);
```
Free the embedder handle and release resources.

**Note:** Handle must not be used after this call.

---

### Utility

#### `embedder_version`
```c
const char* embedder_version(void);
```
Get the library version string.

**Returns:** Static version string (do not free)

---

## Error Codes

```c
typedef enum {
    Success = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    InitializationFailed = 3,
    EmbeddingFailed = 4,
    InvalidHandle = 5,
    BufferTooSmall = 6
} EmbedderErrorCode;
```

## Usage Examples

See the `examples/` directory for complete working examples:

- **C**: `examples/example_c.c`
- **C#**: `examples/example_csharp.cs`
- **Python**: `examples/example_python.py`

### Quick Python Example

```python
from ctypes import CDLL, c_char_p, c_void_p, c_float, c_size_t, POINTER, byref

lib = CDLL("rust_transformer.dll")

# Initialize
handle = lib.embedder_init(b"mini_lm_v2")

# Generate embedding
text = b"Hello, world!"
buffer = (c_float * 512)()
actual_size = c_size_t()

result = lib.embedder_embed(handle, text, buffer, 512, byref(actual_size))

if result == 0:  # Success
    embedding = list(buffer[:actual_size.value])
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

# Cleanup
lib.embedder_free(handle)
```

### Quick C# Example

```csharp
[DllImport("rust_transformer.dll")]
private static extern IntPtr embedder_init(string model_name);

[DllImport("rust_transformer.dll")]
private static extern int embedder_embed(
    IntPtr handle, string text, 
    float[] output, UIntPtr size, out UIntPtr actual);

[DllImport("rust_transformer.dll")]
private static extern void embedder_free(IntPtr handle);

// Usage
IntPtr handle = embedder_init("mini_lm_v2");
float[] embedding = new float[512];
UIntPtr actualSize;

int result = embedder_embed(handle, "Hello!", embedding, 
                            (UIntPtr)512, out actualSize);

Console.WriteLine($"Dimension: {actualSize}");
embedder_free(handle);
```

## Memory Management

### Important Notes:

1. **Initialization**: Always check if `embedder_init()` returns NULL
2. **Buffer Size**: Allocate at least 384 floats for mini_lm_v2, 768 for jina
3. **Error Strings**: Must free error strings with `embedder_free_error()`
4. **Handle Cleanup**: Always call `embedder_free()` when done
5. **Thread Safety**: Each thread should use its own embedder handle

## Typical Embedding Dimensions

- **mini_lm_v2** (all-MiniLM-L6-v2): 384 dimensions
- **jina** (jina-embeddings-v2-base-en): 768 dimensions

## Model Loading

Models are loaded from:
1. Local path (if environment variable is set):
   - `BERT_MODEL_FOLDER` for mini_lm_v2
   - `JINA_MODEL_FOLDER` for jina
2. HuggingFace Hub (automatic download and caching)

First-time usage will download models (~90MB for mini_lm_v2, ~500MB for jina).

## Performance Tips

1. **Reuse handles**: Initialize once, embed many times
2. **Batch processing**: Use `embedder_embed_batch()` for multiple texts
3. **Pre-allocate buffers**: Avoid repeated allocations
4. **Model selection**: mini_lm_v2 is faster, jina has better quality

## Troubleshooting

### DLL Not Found
- Ensure the DLL is in the same directory as your executable
- Or add the DLL directory to your system PATH
- Or specify the full path when loading

### Buffer Too Small
- Check the `actual_size` output parameter
- Allocate a larger buffer and retry
- Use 512 or 1024 as safe default buffer size

### Initialization Fails
- Verify model name is correct ("mini_lm_v2" or "jina")
- Check internet connection for first-time model download
- Ensure sufficient disk space for model cache

### Invalid UTF-8 Errors
- Ensure all text strings are valid UTF-8 encoded
- Check for null terminators in C strings

## Integration Checklist

- [ ] Build the DLL with `cargo build --release --lib`
- [ ] Copy DLL to your project directory
- [ ] Define FFI bindings for your language
- [ ] Initialize embedder with model name
- [ ] Allocate output buffer (512+ floats recommended)
- [ ] Call embedding function(s)
- [ ] Check return codes for errors
- [ ] Free handles when done
- [ ] Test with sample inputs
