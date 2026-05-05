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

    def make_bell_prep_experiment(
        self,
        f_drive_q0,
        amp_pi_q0,
        lo=5.0e9,
        pulse_length=300e-9,
    ):
        """Apply Ry(pi/2) to q0 only, leaving q1 idle.

        This is the first step of Bell state preparation: putting q0 into
        the superposition (|0> + |1>) / sqrt(2).  The second step is a wait
        whose duration equals pi / (2 * J_coupling), which lets the XX+YY
        exchange coupling act as an entangling gate.  The user must determine
        the correct wait time by characterising the coupling strength J.
        """
        exp = Experiment(
            signals=[ExperimentSignal("q0_drive"), ExperimentSignal("q1_drive")]
        )
        pulse = pulse_library.gaussian(uid="ry_half", length=pulse_length, amplitude=1.0)

        with exp.acquire_loop_rt(count=1):
            exp.play(
                "q0_drive",
                pulse,
                amplitude=-(0.5 / np.pi) * amp_pi_q0,  # Ry(pi/2): half a pi pulse
                phase=np.pi / 2,
            )
            exp.delay("q0_drive", time=200e-9)
            exp.delay("q1_drive", time=pulse_length + 200e-9)

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
                        frequency=f_drive_q0 - lo,  # idle; freq doesn't matter
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
