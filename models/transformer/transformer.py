import torch
from torch import nn

from models.transformer.layers import RMSNorm, TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
        post_norm: bool = False,
        layer_norm: bool = True,
        parallel: bool = False,
    ):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.residual_pdrop = residual_pdrop

        # Token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    attn_pdrop,
                    residual_pdrop,
                    post_norm,
                    layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(residual_pdrop)

        # Output layers
        self.ln_final = RMSNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        # Generate token embeddings + position embeddings
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        )

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)

        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
