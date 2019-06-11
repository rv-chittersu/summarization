import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_
from numpy import random


random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class SummarizationModel(nn.Module):

    def __init__(self, encoder, decoder, attention, generator, pad_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.generator = generator
        self.optimizer = None
        self.pad_id = pad_id
        self.max_grad_norm = None

    def set_optimizer(self, opt):
        self.optimizer = opt

    def set_max_grad_norm(self, max_grad_norm):
        self.max_grad_norm = max_grad_norm

    def forward(self, input_ids, input_mask, input_lens, output_ids, output_mask, output_lens, extended_input_ids, target_ids, extra_vocab):
        # pass through encoder and decoder
        encoder_states, init = self.encoder(input_ids, input_mask, input_lens)
        decoder_states, final_state = self.decoder(init, output_ids, output_mask, output_lens)

        # get context and attention_dist
        context, attention_dist = self.attention(decoder_states, output_mask, encoder_states, input_mask)

        # get dist
        final_dist = self.generator(attention_dist, context, decoder_states, extended_input_ids, extra_vocab)

        # return loss
        return self.compute_loss(final_dist, target_ids)

    def compute_loss(self, final_dist, output_ids):
        output_ids = output_ids.reshape(-1)
        final_dist = final_dist.reshape(output_ids.size(0), -1)
        print(final_dist.shape)
        print(output_ids.shape)
        loss = f.cross_entropy(final_dist, output_ids, ignore_index=self.pad_id, reduction='mean')
        if self.training:
            loss.backward()
            clip_grad_norm_(self.parameters(recurse=True), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()
