import torch.nn as nn
import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=layers, bidirectional=True, batch_first=True)

        hidden_state = torch.randn(layers * 2, 1, hidden_dim, dtype=torch.float)
        self.initial_hidden_state = torch.nn.Parameter(hidden_state, requires_grad=True)

        cell_state = torch.randn(layers * 2, 1,  hidden_dim, dtype=torch.float)
        self.initial_cell_state = torch.nn.Parameter(cell_state, requires_grad=True)

        self.hidden_state_projector = torch.nn.Linear(2*hidden_dim, hidden_dim)
        self.cell_state_projector = torch.nn.Linear(2*hidden_dim, hidden_dim)

    def forward(self, input_tensor, mask, lengths):
        # input_tensor : batch_size, seq_len
        # mask : batch_size, seq_len
        # lengths : batch_size, 1
        batch_size, seq_len = input_tensor.shape

        hidden_state = self.initial_hidden_state.repeat(1, batch_size, 1)
        cell_state = self.initial_cell_state.repeat(1, batch_size, 1)

        # DEAL WITH PAD ID
        embeddings = self.embedding(input_tensor)  # batch_size, seq_len, emb_dim

        # pack embeddings
        packed_embeddings = pack_padded_sequence(embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)
        # encode
        packed_hidden_states, (hidden_state, cell_state) = self.lstm(packed_embeddings, (hidden_state, cell_state))
        # unpack hidden_states
        hidden_states = pad_packed_sequence(packed_hidden_states, padding_value=0.0, batch_first=True, total_length=seq_len)
        hidden_states = hidden_states[0]

        # hidden_states: batch, seq_len, 2*hidden_dim
        # hidden_state: layers*2, batch_size, hidden_dim
        # cell_state: layers*2,batch_size, hidden_dim

        # reorder hidden and cell state
        hidden_state = hidden_state.view(self.layers, 2, batch_size, -1)
        cell_state = cell_state.view(self.layers, 2, batch_size, -1)

        hidden_state = torch.cat(torch.split(hidden_state, (1, 1), 1), 3).squeeze(1)  # layers, batch, hidden_dim*2
        cell_state = torch.cat(torch.split(cell_state, (1, 1), 1), 3).squeeze(1)  # layers, batch, hidden_dim*2

        decoder_hidden_state = self.hidden_state_projector(hidden_state)  # layers, batch, hidden_dim
        decoder_cell_state = self.cell_state_projector(cell_state)  # layers, batch, hidden_dim

        return hidden_states, (decoder_hidden_state, decoder_cell_state)


class LSTMDecoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, layers):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=layers, bidirectional=False, batch_first=True)

    def forward(self, context, input_tensor, mask, lengths):
        batch_size, seq_len = mask.shape
        embeddings = self.embedding(input_tensor)  # batch_size, seq_len, emb_dim

        # pack embeddings
        packed_embeddings = pack_padded_sequence(embeddings, lengths=lengths, batch_first=True, enforce_sorted=False)

        # encode
        packed_hidden_states, final_state = self.lstm(packed_embeddings, context)

        # unpack hidden_states
        hidden_states = pad_packed_sequence(packed_hidden_states, padding_value=0.0, batch_first=True, total_length= seq_len)
        hidden_states = hidden_states[0]
        return hidden_states, final_state


class PointerGenerator(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_projector = nn.Linear(d_model, d_model, bias=False)
        self.context_projector = nn.Linear(2*d_model, d_model, bias=False)
        self.generator = nn.Linear(d_model, vocab_size, bias=True)

        self.decoder_gen_projector = nn.Linear(d_model, 1, bias=False)
        self.context_gen_projector = nn.Linear(2*d_model, 1, bias=False)
        self.gen = nn.Linear(1, 1, bias=True)

    def forward(self, attn_dist, context,  decoder_states, extended_input, extra_vocab):
        # attn_dist: batch_size, decoder_len, encoder_len
        # decoder_states: batch_size, dec_len, dim
        # context: batch_size, decoder_len, 2*dim
        # ext_inp: batch_size, enc_len
        # extra_vocab(int): extra_vocab

        batch_size, decoder_len, encoder_len = attn_dist.shape

        proj_decoder_states = self.decoder_projector(decoder_states)  # batch_size, decoder_len, dim
        proj_context = self.context_projector(context)  # batch_size, decoder_len, dim
        result = proj_decoder_states + proj_context  # batch_size, decoder_len, dim

        p_vocab: Tensor = self.generator(result).softmax(dim=2)  # (batch, decoder_len, vocab)

        # get pgen
        p_gen: Tensor = self.gen(self.decoder_gen_projector(decoder_states)
                                 + self.context_gen_projector(context)).sigmoid().squeeze(2)  # (batch, decoder_len)

        # complement of pgen
        p_gen_comp = torch.ones(p_gen.shape, dtype=torch.float) - p_gen  # (batch, decoder_len)

        # pgen * p_vocab
        p_vocab = torch.mul(p_vocab, p_gen.unsqueeze(2).repeat(1, 1, self.vocab_size))  # (batch_size, decoder_len, vocab)

        # pad pgen
        padding = torch.zeros(size=(batch_size, decoder_len, extra_vocab), dtype=torch.float)
        p_vocab = torch.cat([p_vocab, padding], dim=2)  # (batch, decoder_len, vocab + extra_vocab)

        # (1 - pgen)*attn_dist
        p_attn = torch.mul(attn_dist, p_gen_comp.unsqueeze(2).repeat(1, 1, encoder_len))

        # p_vocab: batch, decoder_len, vocab + extra_vocab
        # p_attn: batch, decoder_len, encoder_len
        # ext_inp: batch_size, enc_len

        # total normalized attention
        indices = extended_input.expand(attn_dist.size(1),-1,-1).transpose(0,1)
        p_vocab.scatter_add_(index=indices, src=p_attn, dim=2)  # batch, seq_len, vocab

        return p_vocab
