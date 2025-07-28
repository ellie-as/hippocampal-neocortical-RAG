import random
import string
import numpy as np
from hpc_utils import get_patterns, normalize, convert_string_pattern
from sequence_hopfield_utils import ContinuousHopfield

all_chars = string.ascii_letters + string.digits + string.punctuation + ' []'

class HPC:

    def __init__(self, beta=10000, decay_rate=0.9):
        self.seqs = []
        self.beta = beta
        self.decay_rate = decay_rate
        self.hpc = ContinuousHopfield(pat_size=len(all_chars), beta=self.beta, do_normalization=True)

    def encode(self, sequences_list):
        patterns_list = []
        next_patterns_list = []
        for seq in sequences_list:
            pattern_vecs, next_pattern_vecs = get_patterns(seq, decay_rate=self.decay_rate)
            patterns_list.append(pattern_vecs)
            next_patterns_list.append(next_pattern_vecs)

        pattern_vecs = np.concatenate(patterns_list)
        next_pattern_vecs = np.concatenate(next_patterns_list)

        self.hpc.learn(pattern_vecs, next_pattern_vecs)
        return self.hpc

    def recall(self, input_str, output_len=1000):
        test_pat = convert_string_pattern(input_str)
        pat = test_pat[-1]
        all_chars_str = ""
        for iter in range(output_len):
            pat = self.hpc.retrieve(pat, max_iter=1)
            flattened = normalize([p[0] for p in pat])
            pat = np.array(flattened).reshape(-1, 1)
            char = all_chars[np.argmax(flattened)]
            # ']' is the end of sequence token
            if char == ']':
                break
            all_chars_str += char
        return all_chars_str

    def replay(self):
        # '[' is the start of sequence token
        return self.recall('[')