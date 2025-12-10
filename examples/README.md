# FFI Examples

This folder contains examples of how to use the `rust_transformer.dll` from various languages.

## Building the DLL

First, build the DLL with:

```powershell
cargo build --release --lib
```

The DLL will be located at: `target\release\rust_transformer.dll`

## Available Functions

### `embedder_init(model_name: *const c_char) -> *mut EmbedderHandle`
Initialize an embedder with the specified model ("mini_lm_v2" or "jina").

### `embedder_embed(handle, text, output_buffer, buffer_size, actual_size) -> EmbedderErrorCode`
Generate an embedding for a single text string.

### `embedder_embed_batch(handle, texts, num_texts, output_buffer, buffer_size, embedding_dim, total_written) -> EmbedderErrorCode`
Generate embeddings for multiple texts at once.

### `embedder_get_last_error(handle) -> *const c_char`
Get the last error message.

### `embedder_free_error(error_str)`
Free an error string.

### `embedder_free(handle)`
Free the embedder handle.

### `embedder_version() -> *const c_char`
Get the library version.

## Error Codes

- `0` - Success
- `1` - Null Pointer
- `2` - Invalid UTF-8
- `3` - Initialization Failed
- `4` - Embedding Failed
- `5` - Invalid Handle
- `6` - Buffer Too Small

## See Examples

- `example_c.c` - C usage example
- `example_csharp.cs` - C# usage example
- `example_python.py` - Python (ctypes) usage example
- `example_powershell.ps1` - PowerShell usage example (Add-Type P/Invoke)
