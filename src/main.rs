use clap::Parser;
use rust_transformer::build_text_embedder;
use serde::Serialize;

/// A standalone CLI for text-to-embedding transformations.
#[derive(Parser, Debug)]
#[command(
    author = env!("CARGO_PKG_AUTHORS"),
    version = env!("CARGO_PKG_VERSION"),
    about = env!("CARGO_PKG_DESCRIPTION"),
    long_about = concat!(
        "\n", env!("CARGO_PKG_DESCRIPTION"),
        "\n\nAuthor: ", env!("CARGO_PKG_AUTHORS"),
        "\nVersion: ", env!("CARGO_PKG_VERSION"),
        "\n\nCREDITS: Shamelessly inspired by the work of Martin Contreras Uribe <https://github.com/martin-conur>"
    ),
    after_help = "EXAMPLES:\n  \
    # Generate embedding with default model (mini_lm_v2) as compact JSON:\n  \
    rust_transformer_cli --text \"Hello world\"\n\n  \
    # Use a specific model:\n  \
    rust_transformer_cli --text \"Hello world\" --model jina\n\n  \
    # Output pretty-printed JSON:\n  \
    rust_transformer_cli --text \"Hello world\" --pretty\n\n  \
    # Process multiple texts from JSON array:\n  \
    rust_transformer_cli --text '[\"Hello world\", \"Goodbye\"]' --json-input --pretty\n\n  \
    # Process multiple texts (compact output):\n  \
    rust_transformer_cli --text '[\"Hello world\", \"Goodbye\"]' --json-input\n\n  \
    # Combine model and pretty output:\n  \
    rust_transformer_cli --text \"Hello world\" --model jina --pretty"
)]
struct Args {
    /// The input text to be transformed into an embedding.
    #[arg(short, long)]
    text: String,

    /// The transformer model to use (e.g., 'mini_lm_v2' or 'jina').
    #[arg(short, long, default_value = "mini_lm_v2")]
    model: String,

    /// Output pretty-printed JSON instead of compact JSON.
    #[arg(short, long, default_value_t = false)]
    pretty: bool,

    /// Enable verbose output with additional information.
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// Treat input text as a JSON array of strings.
    #[arg(short, long, default_value_t = false)]
    json_input: bool,
}

#[derive(Serialize)]
struct EmbedResult {
    text: String,
    embed: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.verbose {
        println!("Initializing model: {}...", args.model);
    }

    // --- STEP 1: INITIALIZE THE EMBEDDER ---
    // This loads the tokenizer and model weights, which is the slow part.
    let mut embedder = match build_text_embedder(&args.model) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("\nError initializing embedder for {}: {}", args.model, e);
            std::process::exit(1);
        }
    };

    // --- STEP 2: GENERATE THE EMBEDDING ---
    // This is the fast part—the actual inference.

    // Check if input should be treated as JSON array
    let output = if args.json_input {
        // Handle JSON array input
        let text_array: Vec<String> = serde_json::from_str(&args.text)
            .map_err(|e| format!("Failed to parse JSON array: {}", e))?;

        let mut results = Vec::new();
        for text in text_array {
            match embedder.embed(&text) {
                Ok(embedding_array) => {
                    results.push(EmbedResult {
                        text: text.clone(),
                        embed: embedding_array,
                    });
                }
                Err(e) => {
                    eprintln!("\nError during transformation for text '{}': {}", text, e);
                    std::process::exit(1);
                }
            }
        }

        if args.verbose {
            println!(
                "\nTransformation Output ({} texts processed):",
                results.len()
            );
        }

        if args.pretty {
            serde_json::to_string_pretty(&results)?
        } else {
            serde_json::to_string(&results)?
        }
    } else {
        // Handle single text input
        match embedder.embed(&args.text) {
            Ok(embedding_array) => {
                if args.verbose {
                    println!(
                        "\nTransformation Output ({} elements):",
                        embedding_array.len()
                    );
                }

                if args.pretty {
                    serde_json::to_string_pretty(&embedding_array)?
                } else {
                    serde_json::to_string(&embedding_array)?
                }
            }
            Err(e) => {
                eprintln!("\nError during transformation: {}", e);
                std::process::exit(1);
            }
        }
    };

    println!("{}", output);

    Ok(())
}
