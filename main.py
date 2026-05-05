from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)
import numpy as np
import matplotlib.pyplot as plt

from laboneq.simple import *

from experiments import Experiments
from qrng import HierarchicalQRNG
from qubit import HiddenQubit
from qubit_pair import VirtualQubitPair

# ------------------------------------------------------------------
# Device setup
# ------------------------------------------------------------------
number_of_qubits = 2
device_setup, qubits = generate_device_setup_qubits(
    number_qubits=number_of_qubits,
    pqsc=[{"serial": "DEV10001"}],
    hdawg=[{"serial": "DEV8001", "zsync": 0, "number_of_channels": 8, "options": None}],
    shfqc=[
        {
            "serial": "DEV12001",
            "zsync": 1,
            "number_of_channels": 6,
            "readout_multiplex": 6,
            "options": None,
        }
    ],
    include_flux_lines=True,
    server_host="localhost",
    setup_name=f"my_{number_of_qubits}_fixed_qubit_setup",
)

session = Session(device_setup)
session.connect(do_emulation=True)

experiments = Experiments(device_setup, qubits)


def qubit_spectroscopy(qubit):
    # 100 kHz step: needs to be << J for the CPHASE gate to give clean Bell states.
    freqs = np.linspace(5.00e9, 5.20e9, 2001)
    P1 = []
    for f in freqs:
        exp = experiments.spec_experiment(drive_freq=f)
        compiled = session.compile(exp)
        t, wf = experiments.get_waveform(compiled, pulse_length=2e-6)
        qubit.reset()
        qubit.evolve(t, wf, drive_freq=f)
        bits = qubit.measure(shots=10000)
        P1.append(bits.mean())

    plt.plot(freqs / 1e9, P1)
    plt.xlabel("drive freq [GHz]")
    plt.ylabel("P(1)")
    plt.legend()
    plt.show()

    return freqs[np.argmax(P1)]


def amplitude_rabi(qubit, drive_freq):
    P1 = []
    amps = np.linspace(0, 1.0, 81)
    for amp in amps:
        exp = experiments.make_rabi_experiment([amp])
        compiled = session.compile(exp)
        t, wf = experiments.get_waveform(compiled)
        qubit.reset()
        qubit.evolve(t, wf, drive_freq=drive_freq)
        bits = qubit.measure(shots=1000)
        P1.append(bits.mean())

    plt.plot(amps, P1)
    plt.xlabel("Amplitude, arb.u.")
    plt.ylabel("P(1)")
    plt.legend()
    plt.show()

    return amps[np.argmax(P1)]


def gauss_exp_qrng(qubit, drive_freq, amp_pi):
    n_bits = 8  # 256 bins
    sampler = HierarchicalQRNG(
        experiments, qubit, session, drive_freq=drive_freq, amp_pi=amp_pi, n_bits=n_bits
    )

    # ---------- Gaussian on [-4, 4] ----------
    x_edges = np.linspace(-4, 4, sampler.N + 1)
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    gauss = np.exp(-(x_cent**2) / 2)

    sampler.load_distribution(gauss)
    idx_g = sampler.sample(200)
    samples_g = x_cent[idx_g]

    # ---------- Exponential on [0, 6] ----------
    x_edges = np.linspace(0, 6, sampler.N + 1)
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    expo = np.exp(-x_cent)

    sampler.load_distribution(expo)
    idx_e = sampler.sample(200)
    samples_e = x_cent[idx_e]

    # ---------- Plot ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].hist(samples_g, bins=40, density=True, alpha=0.6, label="QRNG")
    xx = np.linspace(-4, 4, 200)
    ax[0].plot(xx, np.exp(-(xx**2) / 2) / np.sqrt(2 * np.pi), "k-", label="target")
    ax[0].set_title("Gaussian")
    ax[0].legend()

    ax[1].hist(samples_e, bins=40, density=True, alpha=0.6, label="QRNG")
    xx = np.linspace(0, 6, 200)
    ax[1].plot(xx, np.exp(-xx), "k-", label="target")
    ax[1].set_title("Exponential")
    ax[1].legend()
    plt.show()


# CHSH-optimal angles for the prepared state |Phi-> = (|00> - |11>) / sqrt(2):
#   Alice (q0):  a = 0,        a' = pi/2
#   Bob   (q1):  b = -pi/4,    b' = +pi/4   (signs flipped relative to |Phi+>)
CHSH_SETTINGS = [
    (0.0, -np.pi / 4),       # E(a, b)
    (0.0, +np.pi / 4),       # E(a, b')
    (np.pi / 2, -np.pi / 4), # E(a', b)
    (np.pi / 2, +np.pi / 4), # E(a', b')
]


def run_bell_test(
    pair,
    experiments,
    session,
    f_drive_q0,
    f_drive_q1,
    amp_pi_q0,
    amp_pi_q1,
    j_coupling,
    shots_per_setting=4000,
):
    """Run the four CHSH settings. Returns S, correlators, raw bits.

    For each setting:
        1. reset()                          -- return to |00>
        2. evolve(Hadamard pulses)          -- |+>|+>
        3. cphase()                          -- entangling gate
        4. evolve(Rz + final Ry pulses)     -- rotate to Bell basis
        5. evolve(Ry(theta_i) pulses)       -- rotate into measurement basis
        6. measure()                         -- sample
    """
    # Pre-compile the two halves of the Bell preparation pulse sequence.
    prep_pre_exp = experiments.make_bell_prep_pulses_pre(
        f_drive_q0, f_drive_q1, amp_pi_q0, amp_pi_q1
    )
    prep_post_exp = experiments.make_bell_prep_pulses_post(
        f_drive_q0, f_drive_q1, amp_pi_q0, amp_pi_q1
    )
    prep_pre_compiled = session.compile(prep_pre_exp)
    prep_post_compiled = session.compile(prep_post_exp)
    t_pre, w_pre_q0, w_pre_q1 = experiments.get_two_waveforms(prep_pre_compiled, length=1e-6)
    t_post, w_post_q0, w_post_q1 = experiments.get_two_waveforms(prep_post_compiled, length=2e-6)

    E = np.zeros(4)
    raw = []
    for k, (theta0, theta1) in enumerate(CHSH_SETTINGS):
        exp = experiments.make_bell_basis_experiment(
            theta0, theta1, f_drive_q0, f_drive_q1, amp_pi_q0, amp_pi_q1
        )
        compiled = session.compile(exp)
        t, w0, w1 = experiments.get_two_waveforms(compiled)

        pair.reset()
        # Hadamard layer
        pair.evolve(t_pre, w_pre_q0, w_pre_q1,
                    drive_freq_q0=f_drive_q0, drive_freq_q1=f_drive_q1,
                    coupling_on=False)
        # CPHASE gate (coupling briefly turned on, no drive pulses)
        pair.cphase(j_coupling=j_coupling)
        # Rz corrections + final rotation -> Bell state
        pair.evolve(t_post, w_post_q0, w_post_q1,
                    drive_freq_q0=f_drive_q0, drive_freq_q1=f_drive_q1,
                    coupling_on=False)
        # Measurement basis rotation
        pair.evolve(t, w0, w1,
                    drive_freq_q0=f_drive_q0, drive_freq_q1=f_drive_q1,
                    coupling_on=False)
        bits = pair.measure(shots=shots_per_setting)
        a_signed = 1 - 2 * bits[:, 0].astype(np.int8)
        b_signed = 1 - 2 * bits[:, 1].astype(np.int8)
        E[k] = float(np.mean(a_signed * b_signed))
        raw.append(bits)

    S = E[0] + E[1] + E[2] - E[3]
    return S, E, raw


def min_entropy_per_pair(S):
    if S <= 2:
        return 0.0  # no certified randomness
    S = min(S, 2 * np.sqrt(2))  # cap at Tsirelson
    p_star = 0.5 * (1 + np.sqrt(2 - S**2 / 4))
    return -np.log2(p_star)  # bits / shot


def main():
    q0 = HiddenQubit(seed=42)
    q1 = HiddenQubit(seed=5)
    # f_drive_q0 = qubit_spectroscopy(q0)
    f_drive_q0 = q0._fq
    f_drive_q1 = q1._fq
    amp_pi_q0 = amplitude_rabi(q0, drive_freq=f_drive_q0)
    # f_drive_q1 = qubit_spectroscopy(q1)
    amp_pi_q1 = amplitude_rabi(q1, drive_freq=f_drive_q1)
    # print(f"Qubit freq: {f_drive_q0}\nAmplitude: {amp_pi_q0}")
    print(f"q0 actual : {q0._fq / 1e9:.6f} GHz")
    print(f"q0 drive  : {f_drive_q0 / 1e9:.6f} GHz")
    print(f"detuning  : {(q0._fq - f_drive_q0) / 1e6:+.3f} MHz")
    print(f"q1 actual : {q1._fq / 1e9:.6f} GHz")
    print(f"q1 drive  : {f_drive_q1 / 1e9:.6f} GHz")
    print(f"detuning  : {(q1._fq - f_drive_q1) / 1e6:+.3f} MHz")

    # gauss_exp_qrng(q0, f_drive_q0, amp_pi_q0)

    pair = VirtualQubitPair(q0, q1)
    # pair = ClassicalQubitPair(q0, q1)

    S, E, raw = run_bell_test(
        pair,
        experiments,
        session,
        f_drive_q0,
        f_drive_q1,
        amp_pi_q0,
        amp_pi_q1,
        j_coupling=2 * np.pi * 3e6,  # user must characterise this
        shots_per_setting=4000,
    )

    H_per_pair = min_entropy_per_pair(S)
    total_pairs = sum(b.shape[0] for b in raw)

    print(f"  E(a, b)   = {E[0]:+.3f}")
    print(f"  E(a, b')  = {E[1]:+.3f}")
    print(f"  E(a',b)   = {E[2]:+.3f}")
    print(f"  E(a',b')  = {E[3]:+.3f}")
    print(
        f"  CHSH S    = {S:.3f}    (classical ≤ 2,  Tsirelson = {2 * np.sqrt(2):.3f})"
    )
    print(f"  H_min     = {H_per_pair:.3f} bits / pair")
    print(
        f"  ≥ {H_per_pair * total_pairs:.0f} certified bits from {total_pairs} pairs"
        if S > 2
        else "  ❌ no Bell violation; bits not certified"
    )


if __name__ == "__main__":
    main()
