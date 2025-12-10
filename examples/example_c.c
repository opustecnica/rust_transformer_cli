// example_c.c - Example of using rust_transformer.dll from C
// Compile with: gcc example_c.c -o example_c.exe -L../target/release -lrust_transformer

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Match the Rust FFI definitions
typedef void* EmbedderHandle;

typedef enum {
    Success = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    InitializationFailed = 3,
    EmbeddingFailed = 4,
    InvalidHandle = 5,
    BufferTooSmall = 6
} EmbedderErrorCode;

// Function declarations
extern EmbedderHandle embedder_init(const char* model_name);
extern EmbedderErrorCode embedder_embed(
    EmbedderHandle handle,
    const char* text,
    float* output_buffer,
    size_t buffer_size,
    size_t* actual_size
);
extern const char* embedder_get_last_error(EmbedderHandle handle);
extern void embedder_free_error(char* error_str);
extern void embedder_free(EmbedderHandle handle);
extern const char* embedder_version(void);

int main() {
    printf("Rust Transformer DLL Example (C)\n");
    printf("Version: %s\n\n", embedder_version());

    // Initialize embedder with mini_lm_v2 model
    printf("Initializing embedder...\n");
    EmbedderHandle handle = embedder_init("mini_lm_v2");
    
    if (handle == NULL) {
        fprintf(stderr, "Failed to initialize embedder\n");
        return 1;
    }
    printf("Embedder initialized successfully\n\n");

    // Prepare input text
    const char* text = "Hello, world! This is a test.";
    printf("Input text: %s\n\n", text);

    // Allocate output buffer (typical embedding size is 384 for mini_lm_v2)
    size_t buffer_size = 512;
    float* embedding = (float*)malloc(buffer_size * sizeof(float));
    if (embedding == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        embedder_free(handle);
        return 1;
    }

    size_t actual_size = 0;
    
    // Generate embedding
    printf("Generating embedding...\n");
    EmbedderErrorCode result = embedder_embed(
        handle,
        text,
        embedding,
        buffer_size,
        &actual_size
    );

    if (result != Success) {
        fprintf(stderr, "Embedding failed with error code: %d\n", result);
        const char* error_msg = embedder_get_last_error(handle);
        if (error_msg != NULL) {
            fprintf(stderr, "Error: %s\n", error_msg);
            embedder_free_error((char*)error_msg);
        }
        free(embedding);
        embedder_free(handle);
        return 1;
    }

    printf("Embedding generated successfully!\n");
    printf("Embedding dimension: %zu\n\n", actual_size);

    // Print first 10 values of the embedding
    printf("First 10 values:\n");
    for (size_t i = 0; i < (actual_size < 10 ? actual_size : 10); i++) {
        printf("[%zu]: %.6f\n", i, embedding[i]);
    }

    // Cleanup
    free(embedding);
    embedder_free(handle);
    
    printf("\nDone!\n");
    return 0;
}
