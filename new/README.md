
For each transformer block

Input: LLM hidden states H from current layer (L, d)
Output: Fused hidden states H_fused (L, d)

1. Define fixed learnable queries Q_slots
   - Q_slots contains 34 vectors
   - Each vector represents a slot:
     - First 8 slots correspond to 8 instructions
     - Second 2 slots correspond to operands op1 and op2
     - Next 16 slots correspond to mask for op 
     - Last 8 slots correspond to condition flags



2. Project LLM hidden states H into K and V
   - K (L,d),
   - V (L,d)

3. Cross-Attention
   - Input: Q, K, V
   - Output: Z (34, d)
   - Z[:8,:] → instruction features
   - Z[8:10,:] → operand features
   - Z[10:26,:] → mask for operand
   - Z[26:,:] → condition features

4. Projection
    - Project the instruction features to a distribution of instruction
    - Project the condition features to ？

5. Interpreter (Exec)
   - Input: instruction and operand features from previous step
   - Execute the corresponding operations
   - Output: Result vector O_i (1, d)

6. Process the 'program vector' for the current round
    - Concat the result vector and instruction features (8+1,d)
    - project to a set of k and v
    - concat kv sets as new kv for the next round 

7. Repeat steps 3, 4, 5, and 6 for the desired number of cycles.


8. Fuse exec output back into LLM hidden states
    - Record the result vector for each round: O(T,d)
    - H_fused = H + CrossAttn(H, K=W_K@O, V=W_V@O)

9. Output H_fused
   - Feed into next transformer layer or use for next-token prediction