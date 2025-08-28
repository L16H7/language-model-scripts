import argparse
import json
import sys
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextStreamer


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for language model.")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Torch dtype (e.g., auto, float16, bfloat16)",
    )
    parser.add_argument(
        "--message_file",
        type=str,
        default=None,
        help="Path to JSON file with messages",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1000, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Single message string (used if no JSON file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g., cpu, cuda:0)",
    )
    parser.add_argument(
        "--system_message_file",
        type=str,
        default=None,
        help="Path to a txt file containing a system message to prepend to messages",
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="Path to the adapter directory",
    )
    return parser.parse_args()


def get_dtype(dtype_str):
    if dtype_str == "auto":
        return "auto"
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported torch_dtype: {dtype_str}")


def load_messages(args):
    # Load messages from file or argument
    if args.message_file:
        with open(args.message_file, "r") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise ValueError("JSON file must contain a list of messages.")
    elif args.message:
        messages = [{"role": "user", "content": args.message}]
    else:
        messages = []

    # Prepend system message if provided
    if args.system_message_file:
        with open(args.system_message_file, "r") as f:
            system_content = f.read().strip()
        if system_content:
            messages = [{"role": "system", "content": system_content}] + messages
    return messages


def get_user_input():
    user_input = input("\nYou: ").strip()
    if user_input.lower() == "q":
        print("Exiting interactive chat.")
        sys.exit(0)
    return user_input


def interactive_chat(model, tokenizer, messages, max_new_tokens):
    print("\nEntering interactive chat mode. Type 'q' to quit.\n")

    if len(messages) == 0 or (len(messages) == 1 and messages[0]["role"] == "system"):
        user_input = get_user_input()
        messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    while True:
        print("\nAssistant:", flush=True)
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.5,
            top_k=1,
            repetition_penalty=1.2,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
        updated_context = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        user_input = get_user_input()
        user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(
            f"{updated_context}\n{user_text}", return_tensors="pt"
        ).to(model.device)


def main():
    args = parse_args()
    torch_dtype = get_dtype(args.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(args.device)

    if args.lora_adapter:
        model = PeftModel.from_pretrained(model, args.lora_adapter)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    messages = load_messages(args)

    interactive_chat(model, tokenizer, messages, args.max_new_tokens)


if __name__ == "__main__":
    main()
