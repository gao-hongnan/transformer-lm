import torch
import torch.nn as nn

from models.transformer.util import gelu, scaled_dot_product_attention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        attn_pdrop=None,
        residual_pdrop=None,
        post_norm=False,
        layer_norm=True,
        parallel=False,
    ):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.ln2 = RMSNorm(d_model, eps=1e-5)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.dropout = (
            nn.Dropout(residual_pdrop, inplace=False)
            if residual_pdrop is not None
            else nn.Identity()
        )
        if not layer_norm:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.post_norm = post_norm
        self.parallel = parallel

    def prenorm_forward(self, x):
        y = x + self.dropout(self.attn(self.ln1(x)))
        return y + self.dropout(self.ffn(self.ln2(y)))

    def postnorm_forward(self, x):
        y = self.ln1(x + self.dropout(self.attn(x)))
        return self.ln2(y + self.dropout(self.ffn(y)))

    def parallel_forward(self, x):
        y1, y2 = self.norm1(x), self.norm2(x)
        y1 = self.dropout(self.attn(y1))
        y2 = self.dropout(self.ffn(y2))

        return x + y1 + y2

    def forward(self, x):
        if self.parallel:
            return self.parallel_forward(x)

        if self.post_norm:
            return self.postnorm_forward(x)

        return self.prenorm_forward(x)


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dk = self.dv = d_model // num_heads

        self.q_proj = nn.Linear(self.dk * num_heads, d_model, bias=False)
        self.k_proj = nn.Linear(self.dk * num_heads, d_model, bias=False)
        self.v_proj = nn.Linear(self.dk * num_heads, d_model, bias=False)

        self.output_proj = nn.Linear(self.dk * num_heads, d_model, bias=False)

        self.attn_pdrop = attn_pdrop

    def forward(self, x) -> torch.FloatTensor:
        if x.dim() == 2:
            x.unsqueeze(0)

        batch, seq_len, _ = x.shape

        queries = x @ self.q_proj.weight.T
        keys = x @ self.k_proj.weight.T
        values = x @ self.v_proj.weight.T

        queries = queries.view(batch, seq_len, self.num_heads, self.dk).transpose(1, 2)
        keys = keys.view(batch, seq_len, self.num_heads, self.dk).transpose(1, 2)
        values = values.view(batch, seq_len, self.num_heads, self.dk).transpose(1, 2)

        mask = torch.triu(torch.ones((seq_len, seq_len)).bool(), diagonal=1).to(
            x.device
        )
        attn = scaled_dot_product_attention(
            queries, keys, values, mask=mask, pdrop=self.attn_pdrop
        )

        attn = attn.transpose(1, 2)
        attn = attn.reshape(batch, seq_len, -1)
        return self.output_proj(attn)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, gain=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.zeros((d_model,)))
        self.eps = eps
        self.gain = torch.ones(d_model) if gain is None else gain

    def forward(self, x: torch.FloatTensor):
        x_len = len(x.shape)
        n = x * self.weight.view(*[1] * (x_len - 1), self.d_model)
        d = torch.sqrt(
            (1 / self.d_model) * torch.square(x).sum(-1, keepdim=True) + self.eps
        )
        return n / d


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = gelu

    def forward(self, x: torch.FloatTensor):
        x = self.activation(self.w1(x))
        x = self.w2(x)
        return x
