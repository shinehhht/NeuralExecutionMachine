from types import SimpleNamespace

config = SimpleNamespace(
    OP_SET = ['Add','Sub','Mul','cmp_eq','cmp_lt','cmp_gt','jump','halt'],
    value_size = 10,
    n_regs = 6,
    batch_size = 1,
    total_instrution_lines = 3
)

program = {
    'Add': 2,
    'Sub':2,
    
}