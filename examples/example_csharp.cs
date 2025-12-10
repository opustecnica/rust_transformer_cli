// example_csharp.cs - Example of using rust_transformer.dll from C#
// Compile with: csc example_csharp.cs
// Run with the DLL in the same directory or in PATH

using System;
using System.Runtime.InteropServices;
using System.Text;

class RustTransformerExample
{
    // Error codes
    enum EmbedderErrorCode
    {
        Success = 0,
        NullPointer = 1,
        InvalidUtf8 = 2,
        InitializationFailed = 3,
        EmbeddingFailed = 4,
        InvalidHandle = 5,
        BufferTooSmall = 6
    }

    // P/Invoke declarations
    [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr embedder_init(
        [MarshalAs(UnmanagedType.LPStr)] string model_name);

    [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern EmbedderErrorCode embedder_embed(
        IntPtr handle,
        [MarshalAs(UnmanagedType.LPStr)] string text,
        [Out] float[] output_buffer,
        UIntPtr buffer_size,
        out UIntPtr actual_size);

    [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr embedder_get_last_error(IntPtr handle);

    [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void embedder_free_error(IntPtr error_str);

    [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void embedder_free(IntPtr handle);

    [DllImport("rust_transformer.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr embedder_version();

    static string GetLastError(IntPtr handle)
    {
        IntPtr errorPtr = embedder_get_last_error(handle);
        if (errorPtr == IntPtr.Zero)
            return null;

        string error = Marshal.PtrToStringAnsi(errorPtr);
        embedder_free_error(errorPtr);
        return error;
    }

    static void Main()
    {
        Console.WriteLine("Rust Transformer DLL Example (C#)");
        
        // Get version
        IntPtr versionPtr = embedder_version();
        string version = Marshal.PtrToStringAnsi(versionPtr);
        Console.WriteLine($"Version: {version}\n");

        // Initialize embedder
        Console.WriteLine("Initializing embedder...");
        IntPtr handle = embedder_init("mini_lm_v2");
        
        if (handle == IntPtr.Zero)
        {
            Console.WriteLine("Failed to initialize embedder");
            return;
        }
        Console.WriteLine("Embedder initialized successfully\n");

        try
        {
            // Prepare input
            string text = "Hello, world! This is a test from C#.";
            Console.WriteLine($"Input text: {text}\n");

            // Allocate output buffer
            const int bufferSize = 512;
            float[] embedding = new float[bufferSize];
            UIntPtr actualSize;

            // Generate embedding
            Console.WriteLine("Generating embedding...");
            EmbedderErrorCode result = embedder_embed(
                handle,
                text,
                embedding,
                (UIntPtr)bufferSize,
                out actualSize);

            if (result != EmbedderErrorCode.Success)
            {
                Console.WriteLine($"Embedding failed with error code: {result}");
                string error = GetLastError(handle);
                if (error != null)
                    Console.WriteLine($"Error: {error}");
                return;
            }

            Console.WriteLine("Embedding generated successfully!");
            Console.WriteLine($"Embedding dimension: {actualSize}\n");

            // Print first 10 values
            Console.WriteLine("First 10 values:");
            int count = Math.Min((int)actualSize, 10);
            for (int i = 0; i < count; i++)
            {
                Console.WriteLine($"[{i}]: {embedding[i]:F6}");
            }

            // Calculate magnitude (L2 norm)
            double magnitude = 0;
            for (int i = 0; i < (int)actualSize; i++)
            {
                magnitude += embedding[i] * embedding[i];
            }
            magnitude = Math.Sqrt(magnitude);
            Console.WriteLine($"\nL2 Norm: {magnitude:F6}");
        }
        finally
        {
            // Cleanup
            embedder_free(handle);
            Console.WriteLine("\nDone!");
        }
    }
}
