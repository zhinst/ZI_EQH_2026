from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qubit import HiddenQubit


class ClassicalQubitPair:
    """A local hidden-variable (LHV) substitute for HiddenQubitPair.

    Implements the same ``reset`` / ``evolve`` / ``wait`` / ``measure`` API as
    ``HiddenQubitPair``, so it can be dropped in as a black-box replacement.
    Internally it uses a shared hidden variable λ drawn at preparation time
    rather than quantum mechanics, and is therefore guaranteed to satisfy
    |S| ≤ 2 (the classical CHSH bound).

    How a shot works:
        1. At ``reset()``, a new hidden variable λ ∈ [0, 2π) is drawn and
           shared by both sides — analogous to a Bell pair being prepared.
        2. ``evolve()`` computes the effective measurement angle θᵢ for each
           qubit from the area of its LabOne Q waveform, mirroring the
           rotation the quantum version would apply.
        3. ``measure()`` produces a deterministic classical bit per side as a
           function of (θᵢ, λ) according to the chosen LHV strategy.

    The single-qubit marginals are identical to those of HiddenQubitPair: each
    qubit individually returns P(0) = P(1) = 0.5 for any measurement angle,
    matching the maximally mixed marginals of the Bell state.  The difference
    only appears in the joint statistics measured by the CHSH test.

    Args:
        q0:       First qubit (used only to read ``_omega`` for angle calibration).
        q1:       Second qubit (same).
        strategy: ``"optimal"`` — the pseudo-CHSH-optimal LHV that saturates
                  |S| = 2; ``"random"`` — fair coin on each side, giving S ≈ 0.
    """

    def __init__(self, q0: HiddenQubit, q1: HiddenQubit, strategy: str = "optimal") -> None:
        self.q0 = q0
        self.q1 = q1
        self.strategy = strategy

        self._lambda: float = 0.0
        self._theta0: float = 0.0
        self._theta1: float = 0.0

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Draw a fresh shared hidden variable λ, simulating Bell-pair preparation.

        Resets the accumulated measurement angles to zero.  Call this at the
        start of each new experimental setting.
        """
        self._lambda = np.random.uniform(0, 2 * np.pi)
        self._theta0 = 0.0
        self._theta1 = 0.0

    def wait(self, duration: float, n_steps: int = 200) -> None:
        """No-op: the LHV model has no time evolution between pulses.

        Accepted for API compatibility with HiddenQubitPair.
        """

    # ------------------------------------------------------------------
    # Evolution — extracts measurement angles from the waveforms
    # ------------------------------------------------------------------

    def evolve(
        self,
        t: npt.NDArray[np.float64],
        wave_q0: npt.NDArray[np.complex128],
        wave_q1: npt.NDArray[np.complex128],
        drive_freq_q0: float,
        drive_freq_q1: float,
    ) -> None:
        """Compute and accumulate the effective measurement angle from each waveform.

        The angle is the integrated area of the Q (σy) component of the drive,
        scaled by each qubit's ``_omega`` — the same quantity the quantum
        Hamiltonian uses.  Successive calls accumulate, matching the chaining
        semantics of ``HiddenQubitPair.evolve``.

        Drive frequencies are accepted for API compatibility but ignored: the
        LHV model is local and has no notion of resonance.

        Args:
            t:             Time grid [s], shape ``(N,)``.
            wave_q0:       Complex drive envelope for qubit 0, shape ``(N,)``.
            wave_q1:       Complex drive envelope for qubit 1, shape ``(N,)``.
            drive_freq_q0: Drive frequency for qubit 0 [Hz] (unused).
            drive_freq_q1: Drive frequency for qubit 1 [Hz] (unused).
        """
        self._theta0 += self._angle_from_wave(t, wave_q0, self.q0._omega)
        self._theta1 += self._angle_from_wave(t, wave_q1, self.q1._omega)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, shots: int = 1) -> npt.NDArray[np.int8]:
        """Sample classical outcomes using the stored angles and hidden variable.

        Each of the ``shots`` outcomes draws a fresh λ-offset so that the
        marginal distribution stays uniform, while the correlations between
        the two sides are governed by the LHV strategy.

        Args:
            shots: Number of two-qubit measurement shots.

        Returns:
            Integer array of shape ``(shots, 2)``; column 0 is qubit 0's
            outcome, column 1 is qubit 1's outcome.
        """
        # Each shot gets its own λ perturbation around the prepared value,
        # keeping marginals uniform across repeated calls.
        lam = (self._lambda + np.random.uniform(0, 2 * np.pi, size=shots)) % (2 * np.pi)

        b0, b1 = self._lhv_outcomes(self._theta0, self._theta1, lam)
        return np.stack([b0, b1], axis=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_from_wave(
        t: npt.NDArray[np.float64],
        wave: npt.NDArray[np.complex128],
        omega: float,
    ) -> float:
        """Compute the Y-rotation angle from the integrated Q component of a waveform.

        This mirrors the ``σy`` drive term in ``HiddenQubit.hamiltonian_terms``:
        θ = ∫ ω · wave.imag  dt.

        Args:
            t:     Time grid [s].
            wave:  Complex drive envelope.
            omega: Qubit drive calibration constant [rad / (V·s)].

        Returns:
            Rotation angle [rad].
        """
        return float(np.trapezoid(wave.imag * omega, t))

    def _lhv_outcomes(
        self,
        theta0: float,
        theta1: float,
        lam: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        """Apply the LHV strategy to produce per-shot bit outcomes.

        Args:
            theta0: Measurement angle for qubit 0 [rad].
            theta1: Measurement angle for qubit 1 [rad].
            lam:    Per-shot hidden variable values, shape ``(shots,)``.

        Returns:
            Pair of bit arrays ``(b0, b1)``, each shape ``(shots,)``.
        """
        if self.strategy == "optimal":
            # Each side votes on which side of the λ-line it falls.
            # Achieves the maximum classical CHSH value of |S| = 2.
            b0 = (np.cos(theta0 - lam) < 0).astype(np.int8)
            b1 = (np.cos(theta1 - lam) < 0).astype(np.int8)
        elif self.strategy == "random":
            # Independent fair coins — all correlators vanish, S = 0.
            b0 = (np.random.rand(len(lam)) < 0.5).astype(np.int8)
            b1 = (np.random.rand(len(lam)) < 0.5).astype(np.int8)
        else:
            raise ValueError(f"Unknown strategy {self.strategy!r}. Choose 'optimal' or 'random'.")
        return b0, b1
