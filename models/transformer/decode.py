import argparse

import torch
import torch.nn.functional as F

from models.tokenizer.tokenizer import Tokenizer
from models.transformer.transformer import TransformerLM
from models.util import load_checkpoint


def softmax_with_temperature(logits, temperature):
    """Apply softmax with temperature on logits."""
    return F.softmax(logits / temperature, dim=-1)


def top_p_sampling(probs, top_p=0.9):
    """Apply top-p sampling to the probability distribution."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask to the right to keep at least one element
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_probs[sorted_indices_to_remove] = 0
    # Re-normalize the probabilities
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    return new_probs


def decode(model, tokenizer, prompt, max_length, temperature=1.0, top_p=0.9):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        generated = input_ids.tolist()[0]

        for _ in range(max_length):
            output = model(input_ids)
            logits = output[0, -1, :]
            probs = softmax_with_temperature(logits, temperature)
            probs = top_p_sampling(probs, top_p=top_p)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token_id)
            if next_token_id == 0:
                break
            input_ids = torch.cat((input_ids, torch.tensor([[next_token_id]], dtype=torch.long)), dim=1)

        generated_text = tokenizer.decode(generated)

    return generated_text


def load_model(
    dataset: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    resid_pdrop: float,
):
    lm = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
    )
    load_checkpoint(f"checkpoints/{dataset}_best.pth", lm, None)

    return lm


def load_tokenizer(dataset: str):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=f"data/tokenizer/{dataset}-vocab.pkl",
        merges_filepath=f"data/tokenizer/{dataset}-merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Decode text from a Transformer model.")
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Vocabulary size of the model.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        required=True,
        help="Context length of the model.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=True,
        help="Dimension of the model.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        required=True,
        help="Number of layers in the model.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="Number of heads in the model.",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        required=True,
        help="Dimension of the feedforward network.",
    )
    parser.add_argument(
        "--attn_pdrop",
        type=float,
        required=True,
        help="Dropout probability for attention layers.",
    )
    parser.add_argument(
        "--resid_pdrop",
        type=float,
        required=True,
        help="Dropout probability for residual connections.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Initial text prompt to start generating text.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum length of the generated text.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax scaling. Lower is more deterministic.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p value for nucleus sampling. Lower is more focused.",
    )
    parser.add_argument(
        "--model_dataset",
        type=str,
        required=True,
        help="Dataset used to train the model (tiny | owt | corpus).",
    )
    parser.add_argument(
        "--tokenizer_dataset",
        type=str,
        required=True,
        help="Dataset used to train the tokenizer (tiny | owt | corpus).",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        dataset=args.model_dataset,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        resid_pdrop=args.resid_pdrop,
    ).to(device)
    tokenizer = load_tokenizer(args.tokenizer_dataset)

    # Generate text
    generated_text = decode(model, tokenizer, args.prompt, args.max_length, args.temperature, args.top_p)
    print("Generated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()
