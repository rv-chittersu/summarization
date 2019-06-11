import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_src_batch(batch):
    enc_batch = torch.from_numpy(batch.enc_batch).long().to(device)
    enc_padding_mask = torch.from_numpy(batch.enc_padding_mask).float().to(device=device)
    enc_lens = batch.enc_lens

    enc_batch_extend_vocab = torch.from_numpy(batch.enc_batch_extend_vocab).long().to(device=device)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, batch.max_art_oovs


def prepare_tgt_batch(batch):
    dec_batch = torch.from_numpy(batch.dec_batch).long().to(device=device)
    dec_padding_mask = torch.from_numpy(batch.dec_padding_mask).int().to(device=device)
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = torch.from_numpy(dec_lens).float().to(device=device)

    target_batch = torch.from_numpy(batch.target_batch).long().to(device=device)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
