from types import SimpleNamespace

config = SimpleNamespace(
   hidden_dim = 1024,
   num_heads = 8,
   dropout = 0.3,
   total_slots = 18,
   num_instructions = 4,
   num_op = 2,
   num_cycles = 2,
   instruction_types = 4
)