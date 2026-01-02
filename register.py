"""
Quantum register implementation for Shor's algorithm.

This module provides a simple quantum register that stores quantum state
amplitudes and supports measurement operations.

State Representation:
    A quantum register with n qubits is represented by 2^n complex amplitudes,
    one for each computational basis state |00...0⟩ through |11...1⟩.

Qubit Ordering Convention (Big-Endian):
    - Qubit 0 is the most significant bit (leftmost)
    - State index = q0*2^(n-1) + q1*2^(n-2) + ... + qn*2^0
    - Example: |q0 q1 q2⟩ = |101⟩ maps to index 5 = 1*4 + 0*2 + 1*1

Measurement:
    When measured, the register collapses to a definite classical state
    with probability |amplitude|^2 for each basis state. After measurement,
    the register is in a pure state with all probability on the measured value.

Example:
    >>> reg = Register(3)  # Create 3-qubit register in |000⟩
    >>> reg.amplitudes[0]  # Probability amplitude of |000⟩
    1.0
    >>> Hadamard().apply(reg, 0)  # Apply H to qubit 0 only
    >>> # Now in state (|000⟩ + |100⟩)/√2
    >>> # Only qubit 0 is in superposition; qubits 1,2 remain |0⟩
    >>> reg.measure()  # Collapse to definite state
    Register with measured value |000⟩ or |100⟩ (50% probability each)
"""

import numpy as np
from utils import extract_bits


class Register:
    def __init__(self, nq):
        self.nq = nq
        # a quantum state require 2^n amplitudes
        self.amplitudes = np.zeros(2**nq)
        # to begin with set the probability of |0> to 1
        self.amplitudes[0] = 1
        # is this state measured yet?
        # once it is, it will have a definite value
        self.value = None

    def __repr__(self):
        if not self.value:
            return self.amplitudes.__repr__()
        else:
            return f"|{self.value}>"

    def update(self, amps):
        assert len(amps) == len(self.amplitudes)
        self.amplitudes = amps

    def measure(self):
        if self.value:
            # because we back fill the amplitudes this isn't really needed
            # repeated measurements will return the same value
            return self.value
        else:
            # turn amplitudes into probs
            probs = np.absolute(self.amplitudes)**2

            results = list(range(len(probs)))

            # measure a value
            self.value = np.binary_repr(
                    np.random.choice(results, p=probs),
                    self.nq)
            # fill the amplitudes according to the measurement
            self.amplitudes.fill(0)
            self.amplitudes[int(self.value[::-1], base=2)] = 1

            return self

    def measure_qubits(self, qubit_indices):
        """
        Perform partial measurement on specified qubits.

        Measures only the specified qubits while leaving others in superposition.
        This is essential for Shor's algorithm where we measure the target
        register first, then the counting register.

        Args:
            qubit_indices: List of qubit positions to measure

        Returns:
            Integer value of the measured qubits

        Example:
            >>> reg = Register(3)
            >>> # Create state (|000⟩ + |101⟩)/√2
            >>> # Measure only qubit 0
            >>> value = reg.measure_qubits([0])
            >>> # Returns 0 or 1, leaves other qubits in superposition
        """
        # Sample from probability distribution
        probabilities = np.abs(self.amplitudes) ** 2
        measured_state = np.random.choice(len(probabilities), p=probabilities)

        # Extract value from measured qubits
        measured_value = extract_bits(measured_state, qubit_indices, self.nq)

        # Collapse state: zero out inconsistent amplitudes and renormalize
        new_amplitudes = np.copy(self.amplitudes)
        for state_idx in range(len(new_amplitudes)):
            state_value = extract_bits(state_idx, qubit_indices, self.nq)
            if state_value != measured_value:
                new_amplitudes[state_idx] = 0

        # Renormalize after collapse
        norm = np.sqrt(np.sum(np.abs(new_amplitudes) ** 2))
        if norm > 0:
            new_amplitudes /= norm

        self.update(new_amplitudes)
        return measured_value
