"""
The code in this file is based on https://github.com/ml-jku/hopfield-layers.
This repo accompanies Ramsauer et al. (2020) (https://arxiv.org/abs/2008.02217).
We have adapted the code to work for sequences, with decaying activation from previous states.
"""
import numpy as np

# to avoid memory errors for very high beta, we use the limit of softmax
LARGE_THRESHOLD = 5000

class ContinuousHopfield:
    def __init__(self, pat_size, beta=1, do_normalization=False):
        self.size = pat_size  # size of individual pattern
        self.beta = beta
        self.max_norm = np.sqrt(self.size)
        if do_normalization:
            self.softmax = self.softmax_normalized
        else:
            self.softmax = self.softmax_unnormalized

        return

    def learn(self, patterns, next_patterns):
        """expects patterns as numpy arrays and stores them col-wise in pattern matrix
        """
        self.num_pat = len(patterns)
        assert (all(type(x) is np.ndarray for x in patterns)), 'not all input patterns are numpy arrays'
        assert (all(len(x.shape) == 2 for x in patterns)), 'not all input patterns have dimension 2'
        assert (all(1 == x.shape[1] for x in patterns)), 'not all input patterns have shape (-1,1) '
        self.patterns = np.array(patterns).squeeze(axis=-1).T  # save patterns col-wise
        self.next_patterns = np.array(next_patterns).squeeze(axis=-1).T  # save next patterns col-wise
        self.M = max(np.linalg.norm(vec) for vec in patterns)  # maximal norm of actually stored patterns
        return

    def retrieve(self, partial_pattern, max_iter=np.inf, thresh=0.5):
        # partial patterns have to be provided with None/0 at empty spots
        if partial_pattern.size != self.size:
            raise ValueError("Input pattern %r does not match state size: %d vs %d"
                             % (partial_pattern, len(partial_pattern), self.size))

        if None in partial_pattern:
            raise NotImplementedError("None elements not supported")

        assert type(partial_pattern) == np.ndarray, 'test pattern was not numpy array'
        assert len(partial_pattern.shape) == 2 and 1 == partial_pattern.shape[
            1], 'test pattern with shape %r is not a col-vector' % (partial_pattern.shape,)

        pat_old = partial_pattern.copy()
        iters = 0

        while iters < max_iter:

            # the limit of softmax for large beta is just setting the largest element to 1, others to 0
            if self.beta > LARGE_THRESHOLD:
                # in the limit of high beta we want:
                # pat_new = self.next_patterns @ self.limit_softmax(self.patterns.T @ pat_old)
                # softmax in the limit of high beta produces a one-hot vector, so:
                one_hot_vector = self.limit_softmax(self.patterns.T @ pat_old)
                index_of_one = np.argmax(one_hot_vector)
                pat_new = self.next_patterns[:, index_of_one].reshape(-1, 1)

            else:
                pat_new = self.next_patterns @ self.softmax(self.beta * self.patterns.T @ pat_old)

            if np.count_nonzero(pat_old != pat_new) <= thresh:  # converged
                break
            else:
                pat_old = pat_new
            iters += 1

        return pat_new

    @staticmethod
    def softmax_unnormalized(z):
        numerators = np.exp(z)  # top
        denominator = np.sum(numerators)  # bottom
        return numerators / denominator

    def softmax_normalized(self, z):
        numerators = np.exp(z / self.max_norm)  # top
        denominator = np.sum(numerators)  # bottom
        return numerators / denominator

    @staticmethod
    def limit_softmax(x):
        output = np.zeros_like(x)
        # we want random tiebreak, whereas np.argmax(x) just takes first index of max value
        max_ind = np.random.choice(np.where(x == x.max())[0])
        output[max_ind] = 1.0
        return output

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm