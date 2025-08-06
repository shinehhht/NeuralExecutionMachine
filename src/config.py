from types import SimpleNamespace

config = SimpleNamespace(
    n_regs = 4,
    n_mem = 4,
    n_val = 8,
    batch_size = 1,
    hidden_dim = 64,
    intermidate_dim = 1024,
    max_tokens = 100,
    dropout = 0.3,
    prog_max_length = 4,
    n_op = 3,
    n_dst = 2,
    n_src1 = 2,
    n_src2 = 2,
    n_imm = 2,
    pc_bit = 2
)