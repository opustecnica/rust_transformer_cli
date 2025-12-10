/*
 * rust_transformer.h
 * 
 * C/C++ header file for rust_transformer.dll
 * Text embedding library with support for transformer models
 * 
 * Version: 0.4.0
 */

#ifndef RUST_TRANSFORMER_H
#define RUST_TRANSFORMER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to embedder instance */
typedef void* EmbedderHandle;

/* Error codes returned by API functions */
typedef enum {
    EMBEDDER_SUCCESS = 0,
    EMBEDDER_NULL_POINTER = 1,
    EMBEDDER_INVALID_UTF8 = 2,
    EMBEDDER_INITIALIZATION_FAILED = 3,
    EMBEDDER_EMBEDDING_FAILED = 4,
    EMBEDDER_INVALID_HANDLE = 5,
    EMBEDDER_BUFFER_TOO_SMALL = 6
} EmbedderErrorCode;

/*
 * Initialize an embedder with the specified model.
 * 
 * Parameters:
 *   model_name: Name of the model ("mini_lm_v2" or "jina")
 * 
 * Returns:
 *   Handle to embedder instance, or NULL on failure
 * 
 * Example:
 *   EmbedderHandle handle = embedder_init("mini_lm_v2");
 *   if (handle == NULL) {
 *       fprintf(stderr, "Failed to initialize\n");
 *       return 1;
 *   }
 */
EmbedderHandle embedder_init(const char* model_name);

/*
 * Generate an embedding for a single text string.
 * 
 * Parameters:
 *   handle: Embedder handle from embedder_init()
 *   text: Input text (null-terminated UTF-8 string)
 *   output_buffer: Pre-allocated buffer for embedding floats
 *   buffer_size: Size of output_buffer (number of float elements)
 *   actual_size: Output - actual embedding dimension
 * 
 * Returns:
 *   EMBEDDER_SUCCESS on success, error code otherwise
 * 
 * Example:
 *   float embedding[512];
 *   size_t actual_size;
 *   EmbedderErrorCode err = embedder_embed(
 *       handle, "Hello world", embedding, 512, &actual_size
 *   );
 *   if (err == EMBEDDER_SUCCESS) {
 *       printf("Embedding dimension: %zu\n", actual_size);
 *   }
 */
EmbedderErrorCode embedder_embed(
    EmbedderHandle handle,
    const char* text,
    float* output_buffer,
    size_t buffer_size,
    size_t* actual_size
);

/*
 * Generate embeddings for multiple texts in a batch.
 * 
 * Parameters:
 *   handle: Embedder handle
 *   texts: Array of text pointers (null-terminated UTF-8 strings)
 *   num_texts: Number of texts in array
 *   output_buffer: Buffer for all embeddings (flattened)
 *   buffer_size: Total buffer size in floats
 *   embedding_dim: Output - dimension of each embedding
 *   total_written: Output - total number of floats written
 * 
 * Returns:
 *   EMBEDDER_SUCCESS on success, error code otherwise
 * 
 * Note:
 *   Output buffer contains embeddings sequentially.
 *   For 3 texts with dimension 384, buffer contains 1152 floats.
 * 
 * Example:
 *   const char* texts[] = {"First", "Second", "Third"};
 *   float buffer[512 * 3];
 *   size_t dim, written;
 *   EmbedderErrorCode err = embedder_embed_batch(
 *       handle, texts, 3, buffer, 512 * 3, &dim, &written
 *   );
 */
EmbedderErrorCode embedder_embed_batch(
    EmbedderHandle handle,
    const char** texts,
    size_t num_texts,
    float* output_buffer,
    size_t buffer_size,
    size_t* embedding_dim,
    size_t* total_written
);

/*
 * Get the last error message from the embedder.
 * 
 * Parameters:
 *   handle: Embedder handle
 * 
 * Returns:
 *   C string with error message, or NULL if no error
 * 
 * Note:
 *   The returned string must be freed with embedder_free_error()
 * 
 * Example:
 *   if (err != EMBEDDER_SUCCESS) {
 *       const char* error_msg = embedder_get_last_error(handle);
 *       if (error_msg != NULL) {
 *           fprintf(stderr, "Error: %s\n", error_msg);
 *           embedder_free_error((char*)error_msg);
 *       }
 *   }
 */
const char* embedder_get_last_error(EmbedderHandle handle);

/*
 * Free an error string returned by embedder_get_last_error().
 * 
 * Parameters:
 *   error_str: String pointer from embedder_get_last_error()
 * 
 * Note:
 *   Must be called exactly once per error string
 */
void embedder_free_error(char* error_str);

/*
 * Free the embedder handle and release all resources.
 * 
 * Parameters:
 *   handle: Embedder handle to free
 * 
 * Note:
 *   Handle must not be used after this call
 *   Must be called exactly once per handle
 * 
 * Example:
 *   embedder_free(handle);
 *   handle = NULL; // Good practice
 */
void embedder_free(EmbedderHandle handle);

/*
 * Get the library version string.
 * 
 * Returns:
 *   Static version string (do not free)
 * 
 * Example:
 *   printf("Version: %s\n", embedder_version());
 */
const char* embedder_version(void);

#ifdef __cplusplus
}
#endif

#endif /* RUST_TRANSFORMER_H */
