<#
.SYNOPSIS
    Example of using rust_transformer.dll from PowerShell
    
.DESCRIPTION
    This script demonstrates how to call the Rust transformer DLL from PowerShell
    using the native Add-Type cmdlet to define P/Invoke signatures.
    
.NOTES
    Requirements: PowerShell 5.1 or later
    Make sure rust_transformer.dll is in the same directory or in your PATH
#>

# Set strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Determine DLL path (try multiple locations)
$dllPaths = @(
    '.\rust_transformer.dll',
    '..\target\release\rust_transformer.dll',
    "$PSScriptRoot\..\target\release\rust_transformer.dll"
)

$dllPath = $null
foreach ($path in $dllPaths) {
    $resolvedPath = Resolve-Path $path -ErrorAction SilentlyContinue
    if ($resolvedPath -and (Test-Path $resolvedPath)) {
        $dllPath = $resolvedPath.Path
        Write-Host "Found DLL at: $dllPath" -ForegroundColor Green
        break
    }
}

if (-not $dllPath) {
    Write-Error @"
Could not find rust_transformer.dll
Please build the DLL first with: cargo build --release --lib
Searched locations:
$($dllPaths -join "`n")
"@
    exit 1
}

# Escape backslashes for C# string literal
$dllPathEscaped = $dllPath -replace '\\', '\\\\'

# Define the C# code that will provide P/Invoke access to the DLL
$signature = @"
using System;
using System.Runtime.InteropServices;

namespace RustTransformer {
    
    public enum ErrorCode {
        Success = 0,
        NullPointer = 1,
        InvalidUtf8 = 2,
        InitializationFailed = 3,
        EmbeddingFailed = 4,
        InvalidHandle = 5,
        BufferTooSmall = 6
    }
    
    public static class Native {
        private const string DllName = "$dllPathEscaped";
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr embedder_init(
            [MarshalAs(UnmanagedType.LPStr)] string model_name);
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode embedder_embed(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string text,
            [Out] float[] output_buffer,
            UIntPtr buffer_size,
            out UIntPtr actual_size);
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode embedder_embed_batch(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] texts,
            UIntPtr num_texts,
            [Out] float[] output_buffer,
            UIntPtr buffer_size,
            out UIntPtr embedding_dim,
            out UIntPtr total_written);
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr embedder_get_last_error(IntPtr handle);
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void embedder_free_error(IntPtr error_str);
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void embedder_free(IntPtr handle);
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr embedder_version();
    }
    
    public class Embedder : IDisposable {
        private IntPtr handle;
        private bool disposed = false;
        
        public Embedder(string modelName) {
            handle = Native.embedder_init(modelName);
            if (handle == IntPtr.Zero) {
                throw new Exception("Failed to initialize embedder with model: " + modelName);
            }
        }
        
        public float[] Embed(string text, int bufferSize = 512) {
            if (disposed) throw new ObjectDisposedException(nameof(Embedder));
            
            float[] buffer = new float[bufferSize];
            UIntPtr actualSize;
            
            ErrorCode result = Native.embedder_embed(
                handle, text, buffer, (UIntPtr)bufferSize, out actualSize);
            
            if (result != ErrorCode.Success) {
                string error = GetLastError();
                throw new Exception($"Embedding failed: {result}" + 
                    (error != null ? $" - {error}" : ""));
            }
            
            // Resize array to actual size
            float[] embedding = new float[(int)actualSize];
            Array.Copy(buffer, embedding, (int)actualSize);
            return embedding;
        }
        
        public float[][] EmbedBatch(string[] texts, int bufferSizePerText = 512) {
            if (disposed) throw new ObjectDisposedException(nameof(Embedder));
            
            int totalBufferSize = bufferSizePerText * texts.Length;
            float[] buffer = new float[totalBufferSize];
            UIntPtr embeddingDim, totalWritten;
            
            ErrorCode result = Native.embedder_embed_batch(
                handle, texts, (UIntPtr)texts.Length, buffer, 
                (UIntPtr)totalBufferSize, out embeddingDim, out totalWritten);
            
            if (result != ErrorCode.Success) {
                string error = GetLastError();
                throw new Exception($"Batch embedding failed: {result}" + 
                    (error != null ? $" - {error}" : ""));
            }
            
            // Split flattened buffer into individual embeddings
            int dim = (int)embeddingDim;
            float[][] embeddings = new float[texts.Length][];
            for (int i = 0; i < texts.Length; i++) {
                embeddings[i] = new float[dim];
                Array.Copy(buffer, i * dim, embeddings[i], 0, dim);
            }
            
            return embeddings;
        }
        
        private string GetLastError() {
            IntPtr errorPtr = Native.embedder_get_last_error(handle);
            if (errorPtr == IntPtr.Zero) return null;
            
            string error = Marshal.PtrToStringAnsi(errorPtr);
            Native.embedder_free_error(errorPtr);
            return error;
        }
        
        public void Dispose() {
            if (!disposed) {
                if (handle != IntPtr.Zero) {
                    Native.embedder_free(handle);
                    handle = IntPtr.Zero;
                }
                disposed = true;
            }
        }
        
        ~Embedder() {
            Dispose();
        }
    }
}
"@

# Add the type to PowerShell
try {
    Add-Type -TypeDefinition $signature -ErrorAction Stop
    Write-Host '✓ Successfully loaded Rust Transformer DLL' -ForegroundColor Green
} catch {
    # Type might already be loaded
    if ($_.Exception.Message -notlike '*already exists*') {
        throw
    }
    Write-Host '✓ Rust Transformer DLL already loaded' -ForegroundColor Yellow
}

# Get version
$versionPtr = [RustTransformer.Native]::embedder_version()
$version = [System.Runtime.InteropServices.Marshal]::PtrToStringAnsi($versionPtr)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host 'Rust Transformer DLL Example (PowerShell)' -ForegroundColor Cyan
Write-Host "Version: $version" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Example 1: Single Embedding
Write-Host '=== Example 1: Single Text Embedding ===' -ForegroundColor Yellow
Write-Host ''

try {
    $embedder = New-Object RustTransformer.Embedder('mini_lm_v2')
    Write-Host '✓ Embedder initialized successfully' -ForegroundColor Green
    
    $text = 'Hello, world! This is a test from PowerShell.'
    Write-Host "Input text: $text`n" -ForegroundColor White
    
    Write-Host 'Generating embedding...' -ForegroundColor Gray
    $embedding = $embedder.Embed($text)
    
    Write-Host '✓ Embedding generated successfully!' -ForegroundColor Green
    Write-Host "Embedding dimension: $($embedding.Length)" -ForegroundColor Cyan
    
    # Show first 10 values
    Write-Host "`nFirst 10 values:" -ForegroundColor White
    for ($i = 0; $i -lt [Math]::Min(10, $embedding.Length); $i++) {
        Write-Host ('  [{0:D2}]: {1:F6}' -f $i, $embedding[$i]) -ForegroundColor Gray
    }
    
    # Calculate L2 norm
    $sumSquares = 0.0
    foreach ($val in $embedding) {
        $sumSquares += $val * $val
    }
    $l2Norm = [Math]::Sqrt($sumSquares)
    Write-Host ("`nL2 Norm: {0:F6}" -f $l2Norm) -ForegroundColor Cyan
    
} catch {
    Write-Error "Error: $_"
    exit 1
} finally {
    if ($embedder) {
        $embedder.Dispose()
        Write-Host "`n✓ Embedder resources freed" -ForegroundColor Green
    }
}

# Example 2: Batch Embedding
Write-Host "`n`n=== Example 2: Batch Text Embedding ===" -ForegroundColor Yellow
Write-Host ''

try {
    $embedder = New-Object RustTransformer.Embedder('mini_lm_v2')
    
    $texts = @(
        'First sentence.',
        'Second sentence is longer.',
        'The third one.',
        'PowerShell is awesome!'
    )
    
    Write-Host "Processing $($texts.Length) texts:" -ForegroundColor White
    for ($i = 0; $i -lt $texts.Length; $i++) {
        Write-Host ('  [{0}]: {1}' -f $i, $texts[$i]) -ForegroundColor Gray
    }
    
    Write-Host "`nGenerating embeddings..." -ForegroundColor Gray
    $embeddings = $embedder.EmbedBatch($texts)
    
    Write-Host "✓ Generated $($embeddings.Length) embeddings" -ForegroundColor Green
    Write-Host "Each embedding has dimension: $($embeddings[0].Length)" -ForegroundColor Cyan
    
    # Show first 5 values of each embedding
    Write-Host "`nFirst 5 values of each embedding:" -ForegroundColor White
    for ($i = 0; $i -lt $embeddings.Length; $i++) {
        $first5 = $embeddings[$i][0..4] | ForEach-Object { '{0:F4}' -f $_ }
        $valuesStr = $first5 -join ', '
        Write-Host ('  Text {0}: [{1}, ...]' -f $i, $valuesStr) -ForegroundColor Gray
    }
    
    # Calculate cosine similarity between first two texts
    if ($embeddings.Length -ge 2) {
        Write-Host "`nCalculating cosine similarity between texts 0 and 1..." -ForegroundColor White
        
        $dotProduct = 0.0
        for ($i = 0; $i -lt $embeddings[0].Length; $i++) {
            $dotProduct += $embeddings[0][$i] * $embeddings[1][$i]
        }
        
        # Since embeddings are normalized, cosine similarity = dot product
        Write-Host ('Cosine Similarity: {0:F6}' -f $dotProduct) -ForegroundColor Cyan
    }
    
} catch {
    Write-Error "Error: $_"
    exit 1
} finally {
    if ($embedder) {
        $embedder.Dispose()
        Write-Host "`n✓ Embedder resources freed" -ForegroundColor Green
    }
}

# Example 3: Using the low-level API directly
Write-Host "`n`n=== Example 3: Low-Level API Usage ===" -ForegroundColor Yellow
Write-Host ''

$handle = [IntPtr]::Zero
try {
    Write-Host 'Initializing with low-level API...' -ForegroundColor Gray
    $handle = [RustTransformer.Native]::embedder_init('mini_lm_v2')
    
    if ($handle -eq [IntPtr]::Zero) {
        throw 'Failed to initialize embedder'
    }
    Write-Host "✓ Handle obtained: 0x$($handle.ToString('X'))" -ForegroundColor Green
    
    $text = 'Low-level API test'
    $buffer = New-Object float[] 512
    $actualSize = [UIntPtr]::Zero
    
    Write-Host 'Calling embedder_embed directly...' -ForegroundColor Gray
    $result = [RustTransformer.Native]::embedder_embed(
        $handle, $text, $buffer, [UIntPtr]512, [ref]$actualSize)
    
    if ($result -eq [RustTransformer.ErrorCode]::Success) {
        Write-Host "✓ Success! Embedding dimension: $actualSize" -ForegroundColor Green
        
        # Convert UIntPtr to int for array indexing
        $actualSizeInt = $actualSize.ToUInt32()
        
        # Show stats
        $mean = ($buffer[0..($actualSizeInt - 1)] | Measure-Object -Average).Average
        $max = ($buffer[0..($actualSizeInt - 1)] | Measure-Object -Maximum).Maximum
        $min = ($buffer[0..($actualSizeInt - 1)] | Measure-Object -Minimum).Minimum
        
        Write-Host "`nEmbedding Statistics:" -ForegroundColor White
        Write-Host ('  Mean:    {0:F6}' -f $mean) -ForegroundColor Gray
        Write-Host ('  Max:     {0:F6}' -f $max) -ForegroundColor Gray
        Write-Host ('  Min:     {0:F6}' -f $min) -ForegroundColor Gray
    } else {
        Write-Host "✗ Error: $result" -ForegroundColor Red
    }
    
} catch {
    Write-Error "Error: $_"
    exit 1
} finally {
    if ($handle -ne [IntPtr]::Zero) {
        [RustTransformer.Native]::embedder_free($handle)
        Write-Host "`n✓ Handle freed" -ForegroundColor Green
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host 'All examples completed successfully!' -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan
