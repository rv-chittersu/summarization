import torch.nn as nn
import torch
from torch import Tensor


class AdditiveAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.encoder_projection = nn.Linear(2*d_model, d_model, bias=False)
        self.decoder_projection = nn.Linear(d_model, d_model, bias=False)
        self.weight_projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, decoder_states, decoder_mask, encoder_states, encoder_mask):

        # decoder_states: batch, dec_len, emb_dim
        # decoder_mask: batch, dec_len
        # encoder_states: batch, enc_len, 2*emb_dim
        # encoder_mask: batch, enc_len

        batch, decoder_len, d_model = decoder_states.shape
        _, encoder_len, _ = encoder_states.shape

        proj_encoder_states: Tensor = self.encoder_projection(encoder_states)\
            .repeat(1, decoder_len, 1)\
            .reshape(batch, decoder_len, encoder_len, d_model)
        proj_decoder_states: Tensor = self.decoder_projection(decoder_states)\
            .repeat(1, 1, encoder_len)\
            .reshape(batch, decoder_len, encoder_len, d_model)

        weights = self.weight_projection(proj_encoder_states + proj_decoder_states)\
            .reshape(batch, decoder_len, encoder_len)  # (batch, decoder_len, encoder_len)
        encoder_mask = torch.log(encoder_mask.unsqueeze(1)
                                 .repeat(1, decoder_len, 1)
                                 .float())  # (batch, decoder_len, encoder_len)

        masked_weights = weights + encoder_mask
        attn_dist = masked_weights.softmax(2)  # (batch_size, decoder_len, encoder_len)

        encoder_states = encoder_states.repeat(1, decoder_len, 1)\
            .reshape(batch, decoder_len, encoder_len, 2*d_model)
        context = torch.mul(encoder_states, attn_dist.repeat(1, 1, 2*d_model).reshape(batch, decoder_len, encoder_len, 2*d_model))
        context = context.sum(dim=2)

        return context, attn_dist


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.encoder_projection = nn.Linear(2*d_model, d_model)

    def forward(self, decoder_states, decoder_mask, encoder_states, encoder_mask):
        # decoder_states: batch, dec_len, emb_dim
        # decoder_mask: batch, dec_len
        # encoder_states: batch, enc_len, 2*emb_dim
        # encoder_mask: batch, enc_len

        encoder_states = self.encoder_projection(encoder_states)  # batch, encoder_len, dim
        weights = torch.mul(decoder_states.unsqueeze(2), encoder_states.unsqueeze(1)).sum(dim=3).squeeze(3)  # batch, decoder_len, encoder_len
        encoder_mask = torch.log(encoder_mask.float()).unsqueeze(1)  # batch, 1, encoder_len
        masked_weights = weights + encoder_mask
        attn_dist = masked_weights.softmax(2).unsqueeze(3)  # batch, decoder_len, encoder_len, 1
        context = torch.mul(attn_dist, encoder_states.unsqueeze(1)).sum(2).squeeze(2)  # batch_size, decoder_len, dim
        return context, attn_dist
