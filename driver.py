from modules.nn import *
from modules.model import SummarizationModel
from modules.attn import AdditiveAttention
from torch.optim.adagrad import Adagrad
from utils.data import *
from utils.utils import *
from utils.batcher import Batcher
from utils.helper import *
import utils.config as conf
import argparse


use_cuda = conf.use_gpu and torch.cuda.is_available()


def set_optimizer(model):
    optimizer = Adagrad(model.parameters(recurse=True), lr=conf.lr, initial_accumulator_value=conf.adagrad_init_acc)
    model.set_optimizer(optimizer)
    model.set_max_grad_norm(conf.max_grad_norm)


def get_model(vocab):
    encoder = LSTMEncoder(conf.vocab_size, conf.embedding_dim, conf.hidden_dim, conf.layers)
    decoder = LSTMDecoder(conf.vocab_size, conf.embedding_dim, conf.hidden_dim, conf.layers)
    attention = AdditiveAttention(conf.hidden_dim)
    generator = PointerGenerator(conf.hidden_dim, conf.vocab_size)
    model = SummarizationModel(encoder, decoder, attention, generator, pad_id=vocab.get_pad_id)
    if use_cuda:
        model.cuda()
    return model


def step(batch, model):
    input_ids, input_mask, input_lens, extended_input_ids, extra_zeros = prepare_src_batch(batch)
    output_ids, output_mask, max_dec_len, output_lens, target_batch = prepare_tgt_batch(batch)
    loss = model(input_ids, input_mask, input_lens, output_ids, output_mask, output_lens, extended_input_ids, target_batch, extra_zeros)
    return loss


def iterate(init, max_iters, batcher, model, mode):
    batch_i = 0
    total_loss = 0
    while batch_i < max_iters:
        batch = batcher.next_batch()
        loss = step(batch, model)
        total_loss += loss
        batch_i += 1
        if batch_i % 100 == 0:
            total_loss /= 100
            if mode == 'train':
                print("Batch : {:>3} Avg Train Loss {:.5f}".format(batch_i + init, total_loss))
            else:
                print("Batch : {:>3} Avg Validation Loss {:.5f}".format(batch_i + init, total_loss))
            total_loss = 0.0


def train(model):
    train_batcher = Batcher(conf.train_data_path, vocab, mode='train', batch_size=conf.batch_size,single_pass=False)
    eval_batcher = Batcher(conf.eval_data_path, vocab, mode='train', batch_size=conf.batch_size, single_pass=False)
    set_optimizer(model)
    # iterate over batch and print results
    index = 0
    while index < config.iters:
        model.train()
        iterate(index, 10000, train_batcher, model, 'train')
        model.eval()
        iterate(0, 100, eval_batcher, model, 'eval')
        index += 10000
        torch.save(model.state_dict(), conf.save_path + "/model-" + str(index) + ".pt")


def test(model, path, vocab):
    model.load_state_dict(torch.load(path))
    model.eval()

    beam_search = LSTMBeamSearch(conf.beam_size, conf.vocab_size, conf.max_decode_len, model)
    batcher = Batcher(config.decode_data_path, vocab, mode='decode', batch_size=1, single_pass=True)

    counter = 0
    batch = batcher.next_batch()

    while batch is not None:
        input_ids, input_mask, input_lens, extended_input_ids, extra_zeros = prepare_src_batch(batch)
        best_summary = beam_search.generate(input_ids, extended_input_ids, extra_zeros)
        output_ids = [int(t) for t in best_summary.tokens[1:]]
        decoded_words = outputids2words(output_ids, vocab, batch.art_oovs[0])

        try:
            fst_stop_idx = decoded_words.index(STOP_DECODING)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words

        write_for_rouge(batch.original_abstracts_sents[0], decoded_words, counter,
                        conf.rouge_ref_dir, conf.rouge_dec_dir)
        batch = batcher.next_batch()
        counter += 1

    results_dict = rouge_eval(conf.rouge_ref_dir, conf.rouge_dec_dir)
    rouge_log(results_dict, conf.decode_dir)


def summarize(model, path):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--mode', dest='mode', default='train', help='train or eval', required=True)
    parser.add_argument('--save_path', dest='save_path', default='ckpt/', help='Path where model will be saved')
    parser.add_argument('--model', dest='model', help='Path where model will be loaded from for test')
    args = parser.parse_args()

    vocab = Vocab(conf.vocab_path, conf.vocab_size)
    model = get_model(vocab)

    if args.mode == 'train':
        train(model)
    elif args.mode == 'eval':
        pass
