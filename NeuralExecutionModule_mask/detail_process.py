
opcode_type = ["nop","add","sub","and","or","not","sin","cmp"]
def format_program_details(details):
    lines = []
    
    for i, line_data in enumerate(details.get('per_line', [])):
        lines.append(f"Line {i+1}:")
        
        if 'opcode_probs_line' in line_data:
            lines.append("  Opcode Probabilities:")
            max_prob = max(line_data['opcode_probs_line'])
            max_index = line_data['opcode_probs_line'].index(max_prob)
            max_type = opcode_type[max_index]
            for j, prob in enumerate(line_data['opcode_probs_line']):
                lines.append(f"    Opcode {opcode_type[j]}: {prob:.4f}")
            lines.append(f"highest probability opcode is {max_type}")
        """
        if 'arith_x' in line_data:
            lines.append("  Arithmetic X:")
            for j, (prob1, prob2) in enumerate(line_data['arith_x']):
                lines.append(f"    [{j+1:2d}]: {prob1:.4f}, {prob2:.4f}")
        
        if 'arith_y' in line_data:
            lines.append("  Arithmetic Y:")
            for j, (prob1, prob2) in enumerate(line_data['arith_y']):
                lines.append(f"    [{j+1:2d}]: {prob1:.4f}, {prob2:.4f}")
        
        if 'actual_return_value' in line_data:
            lines.append("  Actual Return Values:")
            for j, (val1, val2) in enumerate(line_data['actual_return_value']):
                lines.append(f"    [{j+1:2d}]: {val1:.6f}, {val2:.6f}")
        """
        lines.append("")
    
    return '\n'.join(lines)