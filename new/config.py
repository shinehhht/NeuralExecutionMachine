from types import SimpleNamespace

config = SimpleNamespace(
   hidden_dim = 1024,
   num_heads = 8,
   dropout = 0.3,
   num_instructions = 4,
   num_op = 2,
   instruction_types = 4,
   total_slots_registers=27,
   n_regs = 4,
   vocab_size = 10,
   input_bits = 10,
   output_bits = 11
)