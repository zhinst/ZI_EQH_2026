from laboneq.core.types.enums.modulation_type import ModulationType
from laboneq.dsl.calibration.calibration import Calibration
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.signal_calibration import SignalCalibration
from laboneq.dsl.experiment import pulse_library
from laboneq.dsl.experiment.experiment import Experiment
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.parameter import SweepParameter
from laboneq.simulator.output_simulator import OutputSimulator
import numpy as np

PULSE_LEN = 300e-9
LO_FREQ = 5.0e9
IF_FREQ = 100e6
STEP_DELAY = 200e-9  # idle between sweep steps


class Experiments:
    def __init__(self, device_setup, qubits):
        self.device_setup = device_setup
        self.qubits = qubits
        self.q0 = qubits[0]
        self.q1 = qubits[1]

    def make_rabi_experiment(self, amplitudes, pulse_length=PULSE_LEN):
        exp = Experiment(signals=[ExperimentSignal("q0_drive")])
        amp = SweepParameter("amp", values=np.asarray(amplitudes))

        with exp.acquire_loop_rt(count=1):
            with exp.sweep(parameter=amp):
                exp.play(
                    signal="q0_drive",
                    pulse=pulse_library.gaussian(
                        uid="rabi", length=pulse_length, amplitude=1.0
                    ),
                    amplitude=amp,
                )
                exp.delay(signal="q0_drive", time=STEP_DELAY)

        exp.set_calibration(
            Calibration(
                {
                    "q0_drive": SignalCalibration(
                        oscillator=Oscillator(
                            frequency=IF_FREQ, modulation_type=ModulationType.HARDWARE
                        ),
                        local_oscillator=Oscillator(frequency=LO_FREQ),
                        range=10,
                    )
                }
            )
        )
        exp.set_signal_map({"q0_drive": self.qubits[0].signals["drive"]})
        return exp

    def spec_experiment(self, drive_freq, pulse_length=2e-6, amplitude=0.05):
        exp = Experiment(signals=[ExperimentSignal("q0_drive")])
        with exp.acquire_loop_rt(count=1):
            exp.play(
                "q0_drive",
                pulse_library.const(length=pulse_length, amplitude=1.0),
                amplitude=amplitude,
            )
        exp.set_calibration(
            Calibration(
                {
                    "q0_drive": SignalCalibration(
                        oscillator=Oscillator(
                            frequency=drive_freq - LO_FREQ,
                            modulation_type=ModulationType.HARDWARE,
                        ),
                        local_oscillator=Oscillator(frequency=LO_FREQ),
                        range=10,
                    )
                }
            )
        )
        exp.set_signal_map({"q0_drive": self.qubits[0].signals["drive"]})
        return exp

    def make_bell_basis_experiment(
        self,
        theta0,
        theta1,
        f_drive_q0,
        f_drive_q1,
        amp_pi_q0,
        amp_pi_q1,
        lo=5.0e9,
        pulse_length=300e-9,
    ):
        """Apply Ry(theta_i) to qubit i in parallel, in each qubit's own frame."""
        exp = Experiment(
            signals=[ExperimentSignal("q0_drive"), ExperimentSignal("q1_drive")]
        )
        pulse = pulse_library.gaussian(uid="ry", length=pulse_length, amplitude=1.0)

        with exp.acquire_loop_rt(count=1):
            exp.play(
                "q0_drive",
                pulse,
                amplitude=-(theta0 / np.pi) * amp_pi_q0,
                phase=np.pi / 2,
            )
            exp.play(
                "q1_drive",
                pulse,
                amplitude=-(theta1 / np.pi) * amp_pi_q1,
                phase=np.pi / 2,
            )
            exp.delay("q0_drive", time=200e-9)
            exp.delay("q1_drive", time=200e-9)

        cal = Calibration(
            {
                "q0_drive": SignalCalibration(
                    oscillator=Oscillator(
                        frequency=f_drive_q0 - lo,
                        modulation_type=ModulationType.HARDWARE,
                    ),
                    local_oscillator=Oscillator(frequency=lo),
                    range=10,
                ),
                "q1_drive": SignalCalibration(
                    oscillator=Oscillator(
                        frequency=f_drive_q1 - lo,
                        modulation_type=ModulationType.HARDWARE,
                    ),
                    local_oscillator=Oscillator(frequency=lo),
                    range=10,
                ),
            }
        )
        exp.set_calibration(cal)
        exp.set_signal_map(
            {
                "q0_drive": self.qubits[0].signals["drive"],
                "q1_drive": self.qubits[1].signals["drive"],
            }
        )
        return exp

    def make_bell_prep_pulses_pre(
        self,
        f_drive_q0,
        f_drive_q1,
        amp_pi_q0,
        amp_pi_q1,
        lo=5.0e9,
        pulse_length=PULSE_LEN,
    ):
        """Pulses played BEFORE the CPHASE gate: Hadamard on both qubits.

        Layer 1: Ry_q0(+pi/2),   Ry_q1(+pi/2)            [Hadamards]
        """
        return self._make_layer_experiment(
            [[(0, "y", +np.pi / 2), (1, "y", +np.pi / 2)]],
            f_drive_q0, f_drive_q1, amp_pi_q0, amp_pi_q1, lo, pulse_length,
        )

    def make_bell_prep_pulses_post(
        self,
        f_drive_q0,
        f_drive_q1,
        amp_pi_q0,
        amp_pi_q1,
        lo=5.0e9,
        pulse_length=PULSE_LEN,
    ):
        """Pulses played AFTER the CPHASE gate: Rz(-pi/2) on both + final Ry on q1.

        Layer 1: Rx_q0(-pi/2),   Rx_q1(-pi/2)            [Rz part 1]
        Layer 2: Ry_q0(-pi/2),   Ry_q1(-pi/2)            [Rz part 2]
        Layer 3: Rx_q0(+pi/2),   Rx_q1(+pi/2)            [Rz part 3]
        Layer 4: idle q0,        Ry_q1(-pi/2)            [final]

        Together with the pre-CPHASE Hadamards and the CPHASE wait, this
        produces |Phi-> = (|00> - |11>) / sqrt(2).  CHSH-optimal Bob angles
        are ``-pi/4, +pi/4`` (signs flipped relative to |Phi+>).
        """
        return self._make_layer_experiment(
            [
                [(0, "x", -np.pi / 2), (1, "x", -np.pi / 2)],
                [(0, "y", -np.pi / 2), (1, "y", -np.pi / 2)],
                [(0, "x", +np.pi / 2), (1, "x", +np.pi / 2)],
                [None,                  (1, "y", -np.pi / 2)],
            ],
            f_drive_q0, f_drive_q1, amp_pi_q0, amp_pi_q1, lo, pulse_length,
        )

    def _make_layer_experiment(
        self,
        layers,
        f_drive_q0,
        f_drive_q1,
        amp_pi_q0,
        amp_pi_q1,
        lo,
        pulse_length,
    ):
        """Build a 2-qubit pulse experiment from a list of parallel-gate layers."""
        exp = Experiment(
            signals=[ExperimentSignal("q0_drive"), ExperimentSignal("q1_drive")]
        )
        pulse = pulse_library.gaussian(uid="ry_half", length=pulse_length, amplitude=1.0)
        amp_pi = [amp_pi_q0, amp_pi_q1]
        signal_names = ["q0_drive", "q1_drive"]

        def _phase(axis):
            return 0.0 if axis == "x" else np.pi / 2

        with exp.acquire_loop_rt(count=1):
            for layer in layers:
                for q_gate in layer:
                    if q_gate is None:
                        continue
                    q_idx, axis, angle = q_gate
                    exp.play(
                        signal_names[q_idx],
                        pulse,
                        amplitude=-(angle / np.pi) * amp_pi[q_idx],
                        phase=_phase(axis),
                    )
                # Idle the qubit that didn't get a pulse this layer
                played = {g[0] for g in layer if g is not None}
                for q_idx in (0, 1):
                    if q_idx not in played:
                        exp.delay(signal_names[q_idx], time=pulse_length)

        cal = Calibration(
            {
                "q0_drive": SignalCalibration(
                    oscillator=Oscillator(
                        frequency=f_drive_q0 - lo,
                        modulation_type=ModulationType.HARDWARE,
                    ),
                    local_oscillator=Oscillator(frequency=lo),
                    range=10,
                ),
                "q1_drive": SignalCalibration(
                    oscillator=Oscillator(
                        frequency=f_drive_q1 - lo,
                        modulation_type=ModulationType.HARDWARE,
                    ),
                    local_oscillator=Oscillator(frequency=lo),
                    range=10,
                ),
            }
        )
        exp.set_calibration(cal)
        exp.set_signal_map(
            {
                "q0_drive": self.qubits[0].signals["drive"],
                "q1_drive": self.qubits[1].signals["drive"],
            }
        )
        return exp


    def get_waveform(self, compiled, pulse_length=1000.0):
        sim = OutputSimulator(compiled)
        drive_port = self.device_setup.logical_signal_by_uid(
            self.q0.uid + "/drive"
        ).physical_channel

        snip = sim.get_snippet(drive_port, start=0, output_length=pulse_length)
        return snip.time - snip.time[0], snip.wave

    def get_two_waveforms(self, compiled, length=400e-9):
        sim = OutputSimulator(compiled)
        p0 = self.device_setup.logical_signal_by_uid(
            self.q0.uid + "/drive"
        ).physical_channel
        p1 = self.device_setup.logical_signal_by_uid(
            self.q1.uid + "/drive"
        ).physical_channel
        s0 = sim.get_snippet(p0, start=0, output_length=length)
        s1 = sim.get_snippet(p1, start=0, output_length=length)
        t = s0.time - s0.time[0]
        return t, s0.wave, s1.wave
