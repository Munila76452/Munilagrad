import math
import random
from .engine import value
def dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
def matmul(matrix, vector):
    return [dot_product(row, vector) for row in matrix]

class RNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        # W_xh: Input -> Hidden
        self.W_xh = [[value(random.uniform(-0.1, 0.1)) for _ in range(input_size)] for _ in range(hidden_size)]
        # W_hh: Hidden -> Hidden
        self.W_hh = [[value(random.uniform(-0.1, 0.1)) for _ in range(hidden_size)] for _ in range(hidden_size)]
        # W_hy: Hidden -> Output
        self.W_hy = [[value(random.uniform(-0.1, 0.1)) for _ in range(hidden_size)] for _ in range(output_size)]
        # Initialize biases
        self.b_h = [value(0.0) for _ in range(hidden_size)]
        self.b_y = [value(0.0) for _ in range(output_size)]

    def forward(self, x_t, h_prev):
        input_contribution = matmul(self.W_xh, x_t)
        memory_contribution = matmul(self.W_hh, h_prev)
        h_next = []
        for inp, mem, b in zip(input_contribution, memory_contribution, self.b_h):
            raw = inp + mem + b
            h_next.append(raw.tanh())
        # Compute the output for this specific time step (y_t)
        output_raw = matmul(self.W_hy, h_next)
        y_t = [out + b for out, b in zip(output_raw, self.b_y)]
        
        return y_t, h_next
    
    def parameters(self):
        params = []
        for row in self.W_xh : params.extend(row)
        for row in self.W_hh : params.extend(row)
        for row in self.W_hy : params.extend(row)
        params.extend(self.b_h)
        params.extend(self.b_y)
        return params
# Setup dimensions: e.g., word embeddings of size 4, hidden state of size 8, vocabulary of 50 words
# input_size = 4
# hidden_size = 8
# vocab_size = 50 

# rnn = RNNCell(input_size, hidden_size, vocab_size)

# Mocking our sequence of 4 words (each word is a vector of size 4)
# sequence = [
#     [0.1, 0.2, 0.3, 0.4], # "The"
#     [0.5, 0.6, 0.7, 0.8], # "code"
#     [0.9, 0.1, 0.2, 0.3], # "compiles"
#     [0.4, 0.5, 0.6, 0.7]  # "perfectly"
# ]
# h_current = [0.0 for _ in range(hidden_size)]
# outputs = []
# for t, x_t in enumerate(sequence):
#     y_t, h_current = rnn.forward(x_t, h_current)
#     outputs.append(y_t)
#     print(f"Time Step {t+1}: Processed input. Hidden state updated.")
# The final h_current now holds the "context" of the entire sequence.
# outputs[-1] contains the raw scores for our next-word prediction.
# print('h_current : ',h_current)
# print('outputs : ',outputs)
