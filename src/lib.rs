// lib.rs - FFI exports for creating a native DLL
mod embed_utils;

use embed_utils::TextEmbedder;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::slice;

/// Opaque handle to the embedder instance
pub struct EmbedderHandle {
    embedder: TextEmbedder,
    last_error: Option<String>,
}

/// Error codes returned by FFI functions
#[repr(C)]
pub enum EmbedderErrorCode {
    Success = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    InitializationFailed = 3,
    EmbeddingFailed = 4,
    InvalidHandle = 5,
    BufferTooSmall = 6,
}

/// Initialize an embedder with the specified model.
///
/// # Parameters
/// - `model_name`: C string containing the model name ("mini_lm_v2" or "jina")
///
/// # Returns
/// - Pointer to EmbedderHandle on success, null pointer on failure
///
/// # Safety
/// - The caller must pass a valid null-terminated C string for model_name
/// - The returned handle must be freed with `embedder_free()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn embedder_init(model_name: *const c_char) -> *mut EmbedderHandle {
    if model_name.is_null() {
        return ptr::null_mut();
    }

    let model_name_str = match unsafe { CStr::from_ptr(model_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match embed_utils::build_text_embedder(model_name_str) {
        Ok(embedder) => {
            let handle = Box::new(EmbedderHandle {
                embedder,
                last_error: None,
            });
            Box::into_raw(handle)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Generate an embedding for a single text string.
///
/// # Parameters
/// - `handle`: Pointer to EmbedderHandle returned by embedder_init()
/// - `text`: C string containing the input text
/// - `output_buffer`: Pre-allocated buffer to receive the embedding floats
/// - `buffer_size`: Size of the output_buffer (number of f32 elements it can hold)
/// - `actual_size`: Output parameter - will be set to the actual embedding dimension
///
/// # Returns
/// - EmbedderErrorCode indicating success or failure
///
/// # Safety
/// - handle must be a valid pointer returned by embedder_init()
/// - text must be a valid null-terminated C string
/// - output_buffer must point to allocated memory of at least buffer_size f32 elements
/// - actual_size must be a valid pointer to write the output size
#[unsafe(no_mangle)]
pub unsafe extern "C" fn embedder_embed(
    handle: *mut EmbedderHandle,
    text: *const c_char,
    output_buffer: *mut f32,
    buffer_size: usize,
    actual_size: *mut usize,
) -> EmbedderErrorCode {
    // Validate pointers
    if handle.is_null() {
        return EmbedderErrorCode::InvalidHandle;
    }
    if text.is_null() || output_buffer.is_null() || actual_size.is_null() {
        return EmbedderErrorCode::NullPointer;
    }

    let handle = unsafe { &mut *handle };

    // Convert C string to Rust string
    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            handle.last_error = Some(format!("Invalid UTF-8: {}", e));
            return EmbedderErrorCode::InvalidUtf8;
        }
    };

    // Generate embedding
    match handle.embedder.embed(text_str) {
        Ok(embedding) => {
            let embed_len = embedding.len();
            unsafe { *actual_size = embed_len };

            if embed_len > buffer_size {
                handle.last_error = Some(format!(
                    "Buffer too small: need {} but got {}",
                    embed_len, buffer_size
                ));
                return EmbedderErrorCode::BufferTooSmall;
            }

            // Copy embedding to output buffer
            let output_slice = unsafe { slice::from_raw_parts_mut(output_buffer, embed_len) };
            output_slice.copy_from_slice(&embedding);

            handle.last_error = None;
            EmbedderErrorCode::Success
        }
        Err(e) => {
            handle.last_error = Some(format!("Embedding failed: {}", e));
            EmbedderErrorCode::EmbeddingFailed
        }
    }
}

/// Generate embeddings for multiple text strings (batch processing).
///
/// # Parameters
/// - `handle`: Pointer to EmbedderHandle
/// - `texts`: Array of C string pointers
/// - `num_texts`: Number of texts in the array
/// - `output_buffer`: Pre-allocated buffer to receive all embeddings (flattened)
/// - `buffer_size`: Total size of output_buffer (number of f32 elements)
/// - `embedding_dim`: Output parameter - embedding dimension per text
/// - `total_written`: Output parameter - total number of floats written
///
/// # Returns
/// - EmbedderErrorCode indicating success or failure
///
/// # Safety
/// - handle must be valid
/// - texts must point to an array of num_texts valid C string pointers
/// - output_buffer must have space for num_texts * embedding_dim floats
#[unsafe(no_mangle)]
pub unsafe extern "C" fn embedder_embed_batch(
    handle: *mut EmbedderHandle,
    texts: *const *const c_char,
    num_texts: usize,
    output_buffer: *mut f32,
    buffer_size: usize,
    embedding_dim: *mut usize,
    total_written: *mut usize,
) -> EmbedderErrorCode {
    if handle.is_null() || texts.is_null() || output_buffer.is_null() {
        return EmbedderErrorCode::NullPointer;
    }
    if embedding_dim.is_null() || total_written.is_null() {
        return EmbedderErrorCode::NullPointer;
    }

    let handle = unsafe { &mut *handle };
    let text_ptrs = unsafe { slice::from_raw_parts(texts, num_texts) };

    let mut offset = 0;

    for (i, &text_ptr) in text_ptrs.iter().enumerate() {
        if text_ptr.is_null() {
            handle.last_error = Some(format!("Null text pointer at index {}", i));
            return EmbedderErrorCode::NullPointer;
        }

        let text_str = match unsafe { CStr::from_ptr(text_ptr) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.last_error = Some(format!("Invalid UTF-8 at index {}: {}", i, e));
                return EmbedderErrorCode::InvalidUtf8;
            }
        };

        match handle.embedder.embed(text_str) {
            Ok(embedding) => {
                if i == 0 {
                    let dim = embedding.len();
                    unsafe { *embedding_dim = dim };
                }

                if offset + embedding.len() > buffer_size {
                    handle.last_error = Some(format!(
                        "Buffer overflow at text {}: need {} total but buffer is {}",
                        i,
                        offset + embedding.len(),
                        buffer_size
                    ));
                    return EmbedderErrorCode::BufferTooSmall;
                }

                let output_slice = unsafe {
                    slice::from_raw_parts_mut(output_buffer.add(offset), embedding.len())
                };
                output_slice.copy_from_slice(&embedding);
                offset += embedding.len();
            }
            Err(e) => {
                handle.last_error = Some(format!("Embedding failed at text {}: {}", i, e));
                return EmbedderErrorCode::EmbeddingFailed;
            }
        }
    }

    unsafe { *total_written = offset };
    handle.last_error = None;
    EmbedderErrorCode::Success
}

/// Get the last error message from the embedder.
///
/// # Parameters
/// - `handle`: Pointer to EmbedderHandle
///
/// # Returns
/// - C string containing the error message, or null if no error
/// - The returned string is valid until the next operation on this handle
///
/// # Safety
/// - handle must be valid
/// - The returned string pointer is only valid until the next call to any embedder function
/// - The caller must NOT free the returned string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn embedder_get_last_error(handle: *mut EmbedderHandle) -> *const c_char {
    if handle.is_null() {
        return ptr::null();
    }

    let handle = unsafe { &*handle };

    match &handle.last_error {
        Some(err) => {
            // This creates a potential memory leak, but it's safer for FFI
            // Alternative: use a fixed buffer in EmbedderHandle
            match CString::new(err.as_str()) {
                Ok(c_string) => c_string.into_raw(),
                Err(_) => ptr::null(),
            }
        }
        None => ptr::null(),
    }
}

/// Free the error string returned by embedder_get_last_error.
///
/// # Parameters
/// - `error_str`: String pointer returned by embedder_get_last_error()
///
/// # Safety
/// - error_str must be a pointer returned by embedder_get_last_error()
/// - Must only be called once per error string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn embedder_free_error(error_str: *mut c_char) {
    if !error_str.is_null() {
        unsafe { drop(CString::from_raw(error_str)) };
    }
}

/// Free the embedder handle and release all associated resources.
///
/// # Parameters
/// - `handle`: Pointer to EmbedderHandle to free
///
/// # Safety
/// - handle must be a valid pointer returned by embedder_init()
/// - handle must not be used after this call
/// - Must only be called once per handle
#[unsafe(no_mangle)]
pub unsafe extern "C" fn embedder_free(handle: *mut EmbedderHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Get the library version string.
///
/// # Returns
/// - Static C string with version information
///
/// # Safety
/// - Always safe to call
/// - Returned string is statically allocated and must NOT be freed
#[unsafe(no_mangle)]
pub extern "C" fn embedder_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

// Re-export for use in the CLI binary
pub use embed_utils::build_text_embedder;
