import torch

# TODO
# 1. Length normalization
# 2. Coverage Penalty
# 3. Trigrams Removal


class Beam(object):
    def __init__(self, tokens, log_probs, state):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state

    def extend(self, tokens, log_probs, state):
        result = []
        for index in range(tokens.size(0)):
            result.append(Beam(tokens=self.tokens + [tokens[index]],
                               log_probs=self.log_probs + [log_probs[index]],
                               state=state))
        return result

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class LSTMBeamSearch:

    def __init__(self, beam_size, vocab, max_decode_len, model):
        super().__init__()
        self.beam_size = beam_size
        self.vocab = vocab
        self.max_decode_len = max_decode_len
        self.model = model

    @staticmethod
    def sort_beams(beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    @staticmethod
    def get_prev_states(beams):
        hidden_states = []
        cell_states = []
        for beam in beams:
            hidden, cell = beam.state
            hidden_states.append(hidden)
            cell_states.append(cell)
        # general shape of cell/hidden state: 2, 2, batch, hidden_size
        return torch.cat(hidden_states, dim=2), torch.cat(cell_states, dim=2)

    def generate(self, input_ids, extended_input_ids, extra_vocab):
        encoder_states, decoder_init = self.model.encoder(input_ids, torch.ones(size=input_ids.size), [input_ids.size(1)])

        # initialize beams
        beams = [Beam(tokens=self.vocab.start_id,
                      log_probs=[0.0],
                      state=decoder_init)]
        results = []
        steps = 0
        while steps < self.max_decode_len & len(results) < self.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [i if i < self.vocab.size else self.vocab.oov_id for i in latest_tokens]
            latest_tokens = torch.tensor(latest_tokens)

            prev_states = self.get_prev_states(beams)

            decoder_states, (latest_hidden, latest_cell) = self.model.decoder(prev_states, latest_tokens, torch.ones(size=[len(beams), 1]), 1)
            context, attention_dist = self.model.attention(decoder_states, torch.ones(size=[len(beams), 1]), encoder_states, torch.ones(size=input_ids.size))

            final_dist = self.model.generator(attention_dist, context, decoder_states, extended_input_ids, extra_vocab)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.beam_size * 2)

            topk_ids = topk_ids.squeeze(dim=1)
            topk_log_probs = topk_log_probs.squeeze(dim=1)

            new_beams = []
            latest_hidden = torch.split(latest_hidden, len(beams), dim=2).squeeze(2)
            latest_cell = torch.split(latest_cell, len(beams), dim=2).squeeze(2)
            topk_ids = torch.split(topk_ids, len(beams), dim=0).squeeze(0)
            topk_log_probs = torch.split(topk_log_probs, len(beams), dim=0).squeeze(0)
            for index, beam in enumerate(beams):
                new_beams.append(beam.extend(topk_ids[index], topk_log_probs[index], (latest_hidden[index], latest_cell[index])))

            beams = []
            for beam in new_beams:
                if beam.latest_token == self.vocab.stop_decoding_id:
                    results.append(beam.tokens)
                else:
                    beams.append(beam)

            steps += 1

        return self.vocab.to_words(beams[0].tokens, clean=True)
