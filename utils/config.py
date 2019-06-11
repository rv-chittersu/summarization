# Model Configs
embedding_dim = 512
hidden_dim = 512
layers = 2
beam_size = 4

# Data Configs
vocab_size = 50000
max_encode_len = 400
max_decode_len = 100

# Training Configs
lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0
batch_size = 8
iters = 100000
use_gpu = False


# Paths
train_data_path = "data/chunked/train_*"
eval_data_path = "data/val.bin"
decode_data_path = "data/test.bin"
vocab_path = "data/vocab"
save_path = "ckpt"
log_root = "log"
rouge_ref_dir = 'rouge_ref'
rouge_dec_dir = 'rouge/dec'
decode_dir = 'rouge'
