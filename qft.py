"""
Quantum Fourier Transform implementation.

The QFT is the quantum analogue of the discrete Fourier transform. It transforms
the computational basis states into a Fourier basis, which is crucial for
extracting period information in Shor's algorithm.

Mathematical Operation:
    The QFT maps basis state |j⟩ to:

    QFT|j⟩ = (1/√2^n) Σ(k=0 to 2^n-1) e^(2πijk/2^n) |k⟩

    For a superposition input, it transforms amplitudes via discrete Fourier transform.

Key Property for Shor's Algorithm:
    If the input is a periodic superposition with period r (e.g., states spaced
    by r: |0⟩, |r⟩, |2r⟩, |3r⟩, ...), the QFT output will have peaks at
    multiples of 2^n/r. Measuring then gives a value k where k/2^n ≈ s/r,
    allowing extraction of the period r using continued fractions.

Example:
    Input:  (|0⟩ + |4⟩ + |8⟩ + |12⟩)/2  [period r=4, n=4 qubits]
    Output: High probability at k ∈ {0, 4, 8, 12}
    Measure k=4: 4/16 = 1/4 → period r=4 via continued fractions
"""

import numpy as np
from gates import Hadamard, CP, SWAP


def apply_qft(register, start=0, n_qubits=None):
    """
    Apply Quantum Fourier Transform to specified qubits in the register.

    This function transforms the quantum state by applying the QFT operation,
    which performs a discrete Fourier transform on the quantum amplitudes.
    The transformation is done in-place, modifying the register directly.

    Args:
        register: QuantumRegister to transform (modified in-place)
        start: Starting qubit index (default: 0)
        n_qubits: Number of qubits to include in QFT (default: all from start)

    Returns:
        The same register object, now in the Fourier basis. The returned register
        has been modified such that if the input was:
            Σ α_j |j⟩
        the output is:
            Σ β_k |k⟩  where β_k = (1/√2^n) Σ α_j e^(2πijk/2^n)

    Implementation:
        Uses the efficient quantum circuit construction with O(n²) gates:
        1. Apply Hadamard and controlled phase rotations to each qubit
        2. Reverse qubit order with SWAP gates

    Example:
        >>> reg = Register(4)
        >>> # Create periodic state with period 4
        >>> reg.amplitudes = [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0]
        >>> apply_qft(reg, start=0, n_qubits=4)
        >>> # Now reg has peaks at multiples of 16/4 = 4
        >>> # High probability to measure k ∈ {0, 4, 8, 12}

    Note:
        The register is modified in-place. The return value is provided for
        convenience in method chaining, but the register passed in is the same
        object that is returned.
    """
    if n_qubits is None:
        n_qubits = register.nq - start

    # Phase 1: Apply Hadamard and controlled rotations
    for j in range(n_qubits):
        qubit_j = start + j

        # Apply Hadamard to current qubit
        Hadamard().apply(register, qubit_j)

        # Apply controlled phase rotations from all subsequent qubits
        for k in range(j + 1, n_qubits):
            qubit_k = start + k

            # Calculate rotation index: k - j + 1
            rotation_k = k - j + 1

            # Calculate phase angle: 2π / 2^rotation_k
            phi = 2 * np.pi / (2 ** rotation_k)

            # Apply controlled phase rotation
            CP(phi).apply(register, qt=qubit_j, qc=qubit_k)

    # Phase 2: Reverse qubit order
    for i in range(n_qubits // 2):
        qubit_low = start + i
        qubit_high = start + (n_qubits - 1 - i)
        SWAP().apply(register, q1=qubit_low, q2=qubit_high)

    return register
