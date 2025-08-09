# Language Model Inference Script

This script runs inference for a language model using Hugging Face Transformers, supporting chat-style message input and optional system message prepending.

## Requirements
- Python 3.8+
- torch
- transformers

Install dependencies:
```sh
pip install torch transformers
```

## Usage

### Basic Example
Run inference with a model and a JSON file containing messages:
```sh
python inference.py --model_name <model_path_or_name> --message_file messages.json
```

### Interactive Chat Mode
You can chat interactively with the model. After each response, you can enter a new message (including an empty message) and continue the conversation. Type `exit` or `quit` to end the session.

Start interactive mode with no initial message:
```sh
python inference.py --model_name <model_path_or_name>
```

Or, after running with a message or message file, you will be prompted to continue interactively.
### With a System Message
To prepend a system message from a text file:
```sh
python inference.py --model_name <model_path_or_name> --message_file messages.json --system_message_file system.txt
```

### With a Single Message
You can also provide a single message directly:
```sh
python inference.py --model_name <model_path_or_name> --message "Hello, how are you?"
```

### Additional Options
- `--torch_dtype`: Set torch dtype (e.g., auto, float16, bfloat16, float32)
- `--max_new_tokens`: Max new tokens to generate (default: 1000)
- `--device`: Device to run the model on (e.g., cpu, cuda:0, mps)

## Message File Format
The `--message_file` should be a JSON file containing a list of messages, e.g.:
```json
[
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi! How can I help you?"}
]
```

## Notes
 - You can send empty messages (just press Enter) in interactive mode; these will be appended and sent to the model.

## System Message File
The `--system_message_file` should be a plain text file. Its content will be prepended as a system message, e.g.:
```
You are a helpful assistant.
```

## Example
```sh
python inference.py --model_name Qwen/Qwen3-1.7B --message_file messages.json --system_message_file system.txt --device cpu
```

### Interactive Example
```sh
python inference.py --model_name Qwen/Qwen3-1.7B
```
## License
MIT
