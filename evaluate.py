import torch
from rich.pretty import pprint

from core.config import GPTConfig
from core.layers import GPT
from omnivault.modules.activation import Softmax
from core.tokenizer import Tokenizer
from core.utils import load_checkpoint
from core.config import parse_args


def softmax_with_temperature(dist, temperature: float) -> torch.Tensor:
    softmax = Softmax(dim=-1)
    return softmax(dist / temperature)


def generate(
    model: GPT,
    device: torch.device,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int,
    temperature: float,
    p: float = 0.9,
):
    tokens = tokenizer.encode(prompt)
    print("Generating from tokens:")
    decoded = ""
    model.eval()
    with torch.no_grad():
        while len(tokens) < max_length and not decoded.endswith("<|endoftext|>"):
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(input_tensor)
            logits = logits[0, -1]
            probs = softmax_with_temperature(logits, temperature)
            if p < 1.0:
                sorted_probs, sorted_indices = torch.sort(
                    probs, dim=-1, descending=True
                )
                cumulative_sorted_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumulative_sorted_probs < 0.9
                nucleus[0] = nucleus[0] | (~nucleus.any())
                if not nucleus.any():
                    nucleus[0] = True
                non_nucleus_indices = sorted_indices[~nucleus]
                probs[non_nucleus_indices] = 0.0
                # Renormalize the probabilities
                # print(probs.sum())
                probs /= probs.sum()

            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            decoded = tokenizer.decode(tokens)
        return decoded


def main(args):
    gpt_config = GPTConfig(
        approximate=None,
        activation_name="gelu",
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        context_length=args.context_length,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        bias=False,
        vocab_size=args.vocab_size,
        num_blocks=args.num_layers,
        token_position_pdrop=0.1,
        weight_tie=True,
    )
    model = GPT(config=gpt_config)

    load_checkpoint(src=args.checkpoint_path, model=model)

    tokenizer = Tokenizer.from_files(
        special_tokens=["<|endoftext|>"],
        vocab_filepath=args.vocab_filepath,
        merges_filepath=args.merges_filepath,
    )
    pprint(tokenizer.encode(args.prompt))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = generate(
        model=model,
        device=device,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        p=args.top_p,
    )
    return result


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    generated = main(args)
    pprint(generated)
