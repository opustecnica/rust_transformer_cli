# rust_transformer_cli
```bash
cli sentence transformer

Author: OpusTecnica <https://github.com/opustecnica/rust_transformer_cli>
Version: 0.3.0

CREDITS: Shamelessly inspired by the work of Martin Contreras Uribe <https://github.com/martin-conur>

Usage: rust_transformer_cli.exe [OPTIONS] --text <TEXT>

Options:
  -t, --text <TEXT>
          The input text to be transformed into an embedding

  -m, --model <MODEL>
          The transformer model to use (e.g., 'mini_lm_v2' or 'jina')

          [default: mini_lm_v2]

  -p, --pretty
          Output pretty-printed JSON instead of compact JSON

  -v, --verbose
          Enable verbose output with additional information

  -j, --json-input
          Treat input text as a JSON array of strings

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version

EXAMPLES:
  # Generate embedding with default model (mini_lm_v2) as compact JSON:
  rust_transformer_cli --text "Hello world"

  # Use a specific model:
  rust_transformer_cli --text "Hello world" --model jina

  # Output pretty-printed JSON:
  rust_transformer_cli --text "Hello world" --pretty

  # Process multiple texts from JSON array:
  rust_transformer_cli --text '["Hello world", "Goodbye"]' --json-input --pretty

  # Process multiple texts (compact output):
  rust_transformer_cli --text '["Hello world", "Goodbye"]' --json-input

  # Combine model and pretty output:
  rust_transformer_cli --text "Hello world" --model jina --pretty
```
