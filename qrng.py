import numpy as np


class HierarchicalQRNG:
    """
    Sample from a 2^n-bin distribution using a binary tree of biased
    qubit measurements. Exactly n measurements per sample, one
    LabOne Q compilation per measurement.
    """

    def __init__(self, experiments, qubit, session, drive_freq, amp_pi, n_bits=8):
        self.experiments = experiments
        self.qubit = qubit
        self.session = session
        self.drive_freq = drive_freq
        self.amp_pi = amp_pi
        self.n_bits = n_bits
        self.N = 2**n_bits
        self._biases = None

    # ------------------------------------------------------------------
    # bias  ∈ [0,1]  →  pulse amplitude  (calibrated π-pulse convention)
    # ------------------------------------------------------------------
    def _amp_for_bias(self, p):
        theta = 2 * np.arcsin(np.sqrt(np.clip(p, 0, 1)))
        return (theta / np.pi) * self.amp_pi

    # ------------------------------------------------------------------
    # 1.  Build the binary tree of conditional biases
    # ------------------------------------------------------------------
    def load_distribution(self, bin_probs):
        p = np.asarray(bin_probs, float)
        assert len(p) == self.N, f"need exactly {self.N} bins, got {len(p)}"
        p = p / p.sum()

        biases = np.zeros(self.N)  # heap-indexed; index 0 unused
        for k in range(1, self.N):
            depth = int(np.floor(np.log2(k)))
            width = self.N // (2**depth)
            lo = (k - 2**depth) * width
            hi = lo + width
            mid = (lo + hi) // 2
            total = p[lo:hi].sum()
            biases[k] = (p[mid:hi].sum() / total) if total > 1e-15 else 0.5
        self._biases = biases

    # ------------------------------------------------------------------
    # 2.  Measure a single biased qubit (compile + simulate one pulse)
    # ------------------------------------------------------------------
    def _measure_bit(self, bias):
        amp = self._amp_for_bias(bias)
        exp = self.experiments.make_rabi_experiment([amp])
        compiled = self.session.compile(exp)
        t, wf = self.experiments.get_waveform(compiled)
        self.qubit.reset()
        self.qubit.evolve(t, wf, self.drive_freq)
        return int(self.qubit.measure(shots=1)[0])

    # ------------------------------------------------------------------
    # 3.  Walk the tree
    # ------------------------------------------------------------------
    def sample_one(self):
        if self._biases is None:
            raise RuntimeError("Call load_distribution() first.")
        node = 1
        for _ in range(self.n_bits):
            bit = self._measure_bit(self._biases[node])
            node = 2 * node + bit  # 0 → left, 1 → right
        return node - self.N  # bin index in [0, N)

    def sample(self, n_samples):
        return np.array([self.sample_one() for _ in range(n_samples)])
