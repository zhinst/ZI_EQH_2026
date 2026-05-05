from __future__ import annotations

import numpy as np
import numpy.typing as npt
import qutip as qt

from qubit import VirtualQubit

# Default time resolution for wait() free-evolution segments.
_WAIT_STEPS = 200


class HiddenQubitPair:
    """A two-qubit simulator that prepares a noisy Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2.

    Each qubit can be individually driven by a LabOne Q waveform.  The pair is
    coupled via a transverse (XX + YY) exchange interaction of strength J.

    The joint state is stored in ``self.state`` (a 4×4 density matrix) and
    updated in-place by :meth:`evolve`, :meth:`wait`, and :meth:`reset`.
    Calls chain:

        pair.reset()
        pair.evolve(t1, w0_1, w1_1, f0, f1)   # Bell → ρ₁
        pair.wait(500e-9)                       # ρ₁ → ρ₂  (free decay)
        pair.evolve(t2, w0_2, w1_2, f0, f1)   # ρ₂ → ρ₃
        bits = pair.measure(shots=4000)         # sample from ρ₃

    Call :meth:`reset` to return the pair to the initial Bell state before
    starting a new experiment.

    Args:
        q0:               First qubit.
        q1:               Second qubit.
        prep_visibility:  Mixing weight between the ideal Bell state and the
                          maximally mixed state (1 → pure Bell, 0 → fully mixed).
        J_coupling:       Exchange coupling strength [rad/s].
    """

    def __init__(
        self,
        q0: VirtualQubit,
        q1: VirtualQubit,
        prep_visibility: float = 0.97,
        J_coupling: float = 2 * np.pi * 3e6,
    ) -> None:
        self.q0 = q0
        self.q1 = q1
        self._visibility = prep_visibility
        self._J = J_coupling
        self.state: qt.Qobj = self._make_bell_state()

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the pair to the prepared Bell state |Φ+⟩.

        Call this at the beginning of each new experiment to discard any
        previously accumulated state.
        """
        self.state = self._make_bell_state()

    def _make_bell_state(self) -> qt.Qobj:
        """Prepare a visibility-weighted mixture of |Φ+⟩ and the maximally mixed state."""
        bell = (
            qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
            + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
        ).unit()
        maximally_mixed = qt.tensor(qt.qeye(2), qt.qeye(2)) / 4
        return self._visibility * bell.proj() + (1 - self._visibility) * maximally_mixed

    # ------------------------------------------------------------------
    # Two-qubit operator constructors
    # These replace the generic "embed" lambdas: every operator that acts
    # on q0 is tensor-producted with I₂ on q1, and vice versa.
    # ------------------------------------------------------------------

    @staticmethod
    def _on_q0(op: qt.Qobj) -> qt.Qobj:
        """Embed a single-qubit operator so that it acts on qubit 0."""
        return qt.tensor(op, qt.qeye(2))

    @staticmethod
    def _on_q1(op: qt.Qobj) -> qt.Qobj:
        """Embed a single-qubit operator so that it acts on qubit 1."""
        return qt.tensor(qt.qeye(2), op)

    # ------------------------------------------------------------------
    # Hamiltonian helpers
    # ------------------------------------------------------------------

    def _drive_hamiltonian(
        self,
        wave_q0: npt.NDArray[np.complex128],
        wave_q1: npt.NDArray[np.complex128],
        drive_freq_q0: float,
        drive_freq_q1: float,
    ) -> list:
        """Assemble the full two-qubit drive Hamiltonian.

        Each qubit contributes its own single-qubit terms (detuning + IQ drive),
        embedded into the two-qubit space via :meth:`_on_q0` / :meth:`_on_q1`.
        """
        return self.q0.hamiltonian_terms(
            wave_q0, drive_freq_q0, embed=self._on_q0
        ) + self.q1.hamiltonian_terms(wave_q1, drive_freq_q1, embed=self._on_q1)

    def _coupling_hamiltonian(self) -> list:
        """Build the time-dependent XX + YY exchange coupling between the qubits.

        In the rotating frame, the qubit–qubit detuning Δ = ω_q0 − ω_q1 causes
        the coupling to oscillate at Δ.  The result is expressed as two
        time-dependent terms (real and imaginary parts) in QuTiP format.

        Returns:
            QuTiP-format list of ``[operator, coeff_fn]`` pairs.
        """
        raise_q0_lower_q1 = self._on_q0(qt.sigmap()) * self._on_q1(qt.sigmam())
        lower_q0_raise_q1 = self._on_q0(qt.sigmam()) * self._on_q1(qt.sigmap())
        exchange = raise_q0_lower_q1 + lower_q0_raise_q1

        # Oscillation frequency = difference of qubit resonance frequencies
        delta = 2 * np.pi * (self.q0._fq - self.q1._fq)

        def coupling_cos(t: float, args: dict | None = None) -> float:
            return 0.5 * self._J * np.cos(delta * t)

        def coupling_sin(t: float, args: dict | None = None) -> float:
            return 0.5 * self._J * np.sin(delta * t)

        return [
            [exchange, coupling_cos],  # real part
            [1j * (raise_q0_lower_q1 - lower_q0_raise_q1), coupling_sin],  # imag part
        ]

    def _collapse_operators(self) -> list[qt.Qobj]:
        """Collect Lindblad collapse operators for both qubits, embedded in the 4D space."""
        return self.q0.collapse_operators(
            embed=self._on_q0
        ) + self.q1.collapse_operators(embed=self._on_q1)

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------

    def evolve(
        self,
        t: npt.NDArray[np.float64],
        wave_q0: npt.NDArray[np.complex128],
        wave_q1: npt.NDArray[np.complex128],
        drive_freq_q0: float,
        drive_freq_q1: float,
    ) -> None:
        """Propagate the joint state forward under two drive pulses, updating ``self.state``.

        The propagation starts from the current ``self.state``, so successive
        calls chain automatically.  Use :meth:`reset` to return to the Bell state.

        The time grid ``t`` should start at 0 — it is interpreted as relative
        time within the pulse, not absolute lab time.

        Args:
            t:             Time grid [s], shape ``(N,)``, starting at 0.
            wave_q0:       Complex drive envelope for qubit 0, shape ``(N,)``.
            wave_q1:       Complex drive envelope for qubit 1, shape ``(N,)``.
            drive_freq_q0: Drive frequency for qubit 0 [Hz].
            drive_freq_q1: Drive frequency for qubit 1 [Hz].
        """
        H = self._drive_hamiltonian(wave_q0, wave_q1, drive_freq_q0, drive_freq_q1)
        H += self._coupling_hamiltonian()
        c_ops = self._collapse_operators()
        result = qt.mesolve(H, self.state, t, c_ops=c_ops, e_ops=[])
        self.state = result.states[-1]

    def wait(self, duration: float, n_steps: int = _WAIT_STEPS) -> None:
        """Let both qubits decay freely for ``duration`` seconds under zero drive.

        No pulses are applied — only the T1 / T2 Lindblad channels of each qubit
        and the exchange coupling act.  This is useful for modelling idle time
        between two-qubit gate layers or entanglement generation latency.

        Like :meth:`evolve`, this updates ``self.state`` in-place so calls chain.

        Args:
            duration: Idle time [s].
            n_steps:  Number of time steps for the mesolve integration.
                      More steps give higher accuracy for long idles; the
                      default of 200 is sufficient for durations up to ~T1.
        """
        t = np.linspace(0, duration, n_steps)
        H_zero = qt.Qobj(np.zeros((4, 4)))
        c_ops = self._collapse_operators()
        result = qt.mesolve(H_zero, self.state, t, c_ops=c_ops, e_ops=[])
        self.state = result.states[-1]

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, shots: int = 1) -> npt.NDArray[np.int8]:
        """Sample computational-basis outcomes from the current state.

        Reads ``self.state`` at call time, so the result always reflects the
        most recent :meth:`evolve` or :meth:`wait`.  Can be called multiple
        times after a single evolution without side effects on the state.

        Each shot yields a pair of bits (b0, b1) ∈ {0, 1}², drawn from the
        Born-rule probabilities of the stored density matrix, then corrupted
        by each qubit's independent readout error model.

        Args:
            shots: Number of two-qubit measurement shots.

        Returns:
            Integer array of shape ``(shots, 2)``; column 0 is qubit 0's outcome,
            column 1 is qubit 1's outcome.
        """
        probs = self._outcome_probabilities()
        outcomes = np.random.choice(4, size=shots, p=probs)

        # Decode the 2-bit outcomes: outcome k → (k >> 1, k & 1)
        bits = np.stack([(outcomes >> 1) & 1, outcomes & 1], axis=1).astype(np.int8)
        bits[:, 0] = self.q0._apply_readout_error(bits[:, 0])
        bits[:, 1] = self.q1._apply_readout_error(bits[:, 1])
        return bits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _outcome_probabilities(self) -> npt.NDArray[np.float64]:
        """Compute Born-rule probabilities for all four computational-basis outcomes.

        Returns:
            Normalised probability array ``[P(00), P(01), P(10), P(11)]``.
        """
        projectors = [
            qt.tensor(qt.basis(2, a).proj(), qt.basis(2, b).proj())
            for a in (0, 1)
            for b in (0, 1)
        ]
        probs = np.array([abs((P * self.state).tr()) for P in projectors], dtype=float)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        return probs
