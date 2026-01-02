"""
Quantum gate implementations for Shor's algorithm.

This module provides basic quantum gates (Hadamard, CNOT, etc.) and
specialized modular arithmetic gates needed for Shor's factoring algorithm.

Qubit Ordering Convention (Big-Endian):
    - Qubit 0 is the leftmost, most significant bit
    - For n qubits, state index is: q0*2^(n-1) + q1*2^(n-2) + ... + qn*2^0
    - Example: |q0 q1 q2⟩ = |101⟩ has index 5 = 1*4 + 0*2 + 1*1
"""

import numpy as np
from functools import reduce
from utils import extract_bits, replace_bits


__all__ = ['extract_bits', 'replace_bits', 'nkron', 'I', 'X', 'Hadamard',
           'P', 'CNOT', 'SWAP', 'CZ', 'CP', 'ModularMultiply',
           'ControlledModularMultiply']


def nkron(*args):
    """Compute Kronecker product of multiple matrices."""
    return reduce(np.kron, args)


class QuantumException(Exception):
    pass


class Gate:
    def __init__(self):
        """Initialize gate with matrix set to None (to be defined by subclasses)."""
        self.matrix = None

    def __repr__(self):
        return self.__class__.__name__

    def apply(self, register, iq=0):
        """
        Apply this gate to a single qubit in the register.

        Args:
            register: QuantumRegister to apply gate to
            iq: Index of qubit to apply gate to (0-indexed)

        Returns:
            The modified register

        Raises:
            QuantumException: If register has already been measured
        """
        if register.value:
            raise QuantumException("Cannot apply a Gate to a Collapsed State")

        n = register.nq
        gates = [I().matrix] * n
        gates[iq] = self.matrix

        gate_matrix = nkron(*gates)
        new_state = gate_matrix @ register.amplitudes

        register.update(new_state)
        return register


class I(Gate):
    """
    Identity gate (I).

    Matrix:
        [[1, 0],
         [0, 1]]
    """

    def __init__(self):
        super().__init__()
        self.matrix = np.eye(2)


class X(Gate):
    """
    Pauli-X gate (NOT gate).

    Flips the qubit state: X|0⟩ = |1⟩, X|1⟩ = |0⟩

    Matrix:
        [[0, 1],
         [1, 0]]
    """

    def __init__(self):
        super().__init__()
        self.matrix = np.array([[0, 1], [1, 0]])


class Hadamard(Gate):
    """
    Hadamard gate (H).

    Creates equal superposition:
        H|0⟩ = (|0⟩ + |1⟩)/√2
        H|1⟩ = (|0⟩ - |1⟩)/√2

    Matrix:
        (1/√2) * [[1,  1],
                   [1, -1]]
    """

    def __init__(self):
        super().__init__()
        self.matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


class P(Gate):
    """
    Phase rotation gate (P).

    Applies phase rotation to |1⟩ state:
        P(φ)|0⟩ = |0⟩
        P(φ)|1⟩ = e^(iφ)|1⟩

    Matrix:
        [[1,      0],
         [0, e^(iφ)]]

    Args:
        phi: Phase angle in radians (default: π/4)
    """

    def __init__(self, phi=2 * np.pi / 8):
        super().__init__()
        self.matrix = np.array([[1, 0], [0, np.e ** (phi * 1j)]])


class CNOT(Gate):
    """
    Controlled-NOT gate.

    Flips target qubit if control qubit is |1⟩:
        CNOT|00⟩ = |00⟩
        CNOT|01⟩ = |01⟩
        CNOT|10⟩ = |11⟩
        CNOT|11⟩ = |10⟩

    Implementation uses projector formula: (P0 ⊗ I) + (P1 ⊗ X)
    where P0 = |0⟩⟨0|, P1 = |1⟩⟨1|
    """

    def __init__(self):
        super().__init__()
        self.matrix = np.array([[0, 1], [1, 0]])  # X gate for target

    def apply(self, register, qt, qc):
        """
        Apply CNOT to register.

        Args:
            register: QuantumRegister to apply gate to
            qt: Index of target qubit (gets flipped)
            qc: Index of control qubit (controls the flip)
        """
        P0 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
        P1 = np.array([[0, 0], [0, 1]])  # |1⟩⟨1|
        n = register.nq

        # Build (P0 ⊗ I ⊗ ... ⊗ I)
        g0 = [I().matrix] * n
        g0[qc] = P0

        # Build (P1 ⊗ ... ⊗ X ⊗ ... ⊗ I)
        g1 = [I().matrix] * n
        g1[qc] = P1
        g1[qt] = self.matrix

        gate_matrix = nkron(*g0) + nkron(*g1)
        new_state = gate_matrix @ register.amplitudes
        register.update(new_state)
        return register


class SWAP(Gate):
    """
    SWAP gate.

    Exchanges the states of two qubits:
        SWAP|00⟩ = |00⟩
        SWAP|01⟩ = |10⟩
        SWAP|10⟩ = |01⟩
        SWAP|11⟩ = |11⟩

    Implemented as CNOT(a,b) • CNOT(b,a) • CNOT(a,b)
    """

    def __init__(self):
        super().__init__()

    def apply(self, register, q1, q2):
        """
        Apply SWAP to register.

        Args:
            register: QuantumRegister to apply gate to
            q1: Index of first qubit to swap
            q2: Index of second qubit to swap
        """
        c = CNOT()
        c.apply(register, qt=q1, qc=q2)
        c.apply(register, qt=q2, qc=q1)
        c.apply(register, qt=q1, qc=q2)
        return register


class CZ(Gate):
    """
    Controlled-Z gate.

    Applies Z gate to target if control is |1⟩:
        CZ|00⟩ = |00⟩
        CZ|01⟩ = |01⟩
        CZ|10⟩ = |10⟩
        CZ|11⟩ = -|11⟩

    Implemented as H • CNOT • H
    """

    def __init__(self):
        super().__init__()

    def apply(self, register, qt, qc):
        """
        Apply CZ to register.

        Args:
            register: QuantumRegister to apply gate to
            qt: Index of target qubit
            qc: Index of control qubit
        """
        c = CNOT()
        h = Hadamard()

        h.apply(register, qt)
        c.apply(register, qt=qt, qc=qc)
        h.apply(register, qt)

        return register


class CP(P):
    """
    Controlled-Phase gate.

    Applies phase rotation to target if control is |1⟩:
        CP(φ)|00⟩ = |00⟩
        CP(φ)|01⟩ = |01⟩
        CP(φ)|10⟩ = |10⟩
        CP(φ)|11⟩ = e^(iφ)|11⟩

    Args:
        phi: Phase angle in radians (default: π/2)
    """

    def __init__(self, phi=np.pi / 2):
        super().__init__(phi=phi)
        self.phi = phi

    def apply(self, register, qt, qc):
        """
        Apply CP to register.

        Args:
            register: QuantumRegister to apply gate to
            qt: Index of target qubit (receives phase)
            qc: Index of control qubit
        """
        P0 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
        P1 = np.array([[0, 0], [0, 1]])  # |1⟩⟨1|
        n = register.nq

        # Build (P0 ⊗ I ⊗ ... ⊗ I)
        g0 = [I().matrix] * n
        g0[qc] = P0

        # Build (P1 ⊗ ... ⊗ P(φ) ⊗ ... ⊗ I)
        g1 = [I().matrix] * n
        g1[qc] = P1
        g1[qt] = self.matrix

        gate_matrix = nkron(*g0) + nkron(*g1)

        new_state = gate_matrix @ register.amplitudes
        register.update(new_state)

        return register


class ModularMultiply(Gate):
    """
    Modular multiplication gate: |y⟩ → |k·y mod N⟩

    This gate performs modular multiplication as a permutation of basis states.
    It rearranges which computational basis state maps to which, leaving
    the quantum superposition structure intact.

    Example (k=7, N=15):
        |1⟩  → |7⟩   (7·1 mod 15 = 7)
        |7⟩  → |4⟩   (7·7 mod 15 = 4)
        |4⟩  → |13⟩  (7·4 mod 15 = 13)
        |13⟩ → |1⟩   (7·13 mod 15 = 1)  [cycle complete, period = 4]

    States with value ≥ N are left unchanged.

    Args:
        k: Multiplier (must be coprime to N for valid period finding)
        N: Modulus
    """

    def __init__(self, k, N):
        super().__init__()
        self.k = k
        self.N = N

    def apply(self, register, target_qubits):
        """
        Apply modular multiplication to specific qubits.

        Args:
            register: QuantumRegister to apply gate to
            target_qubits: List of qubit indices to operate on
                          (e.g., [4, 5, 6, 7] for a 4-qubit value)
        """
        n_total = register.nq

        # Build the permutation matrix
        size = 2 ** n_total
        perm_matrix = np.zeros((size, size), dtype=complex)

        # For each basis state of the full register
        for state_idx in range(size):
            # Extract the value stored in target qubits
            target_val = extract_bits(state_idx, target_qubits, n_total)

            # Apply modular multiplication (only if value < N)
            if target_val < self.N:
                new_target_val = (self.k * target_val) % self.N
            else:
                new_target_val = target_val  # Leave states >= N unchanged

            # Build new state index with multiplied value
            new_state_idx = replace_bits(state_idx, target_qubits, n_total, new_target_val)

            # Set permutation matrix entry
            # This is a permutation: exactly one 1 per row and column
            perm_matrix[new_state_idx, state_idx] = 1

        # Apply the permutation
        new_amplitudes = perm_matrix @ register.amplitudes
        register.update(np.asarray(new_amplitudes).flatten())
        return register


class ControlledModularMultiply(Gate):
    """
    Controlled modular multiplication gate.

    Performs modular multiplication only when control qubit is |1⟩:
        |0⟩|y⟩ → |0⟩|y⟩           (control=0: do nothing)
        |1⟩|y⟩ → |1⟩|k·y mod N⟩   (control=1: multiply)

    This is the key operation in Shor's quantum modular exponentiation.
    By applying this gate with different multipliers k controlled by
    different qubits, we compute a^x mod N for superposed values of x.

    Implementation uses projector formula: (P0 ⊗ I) + (P1 ⊗ U_mult)
    where P0 = |0⟩⟨0|, P1 = |1⟩⟨1|, U_mult is the multiplication permutation.

    Args:
        k: Multiplier
        N: Modulus
    """

    def __init__(self, k, N):
        super().__init__()
        self.k = k
        self.N = N

    def apply(self, register, control_qubit, target_qubits):
        """
        Apply controlled modular multiplication.

        Args:
            register: QuantumRegister to apply gate to
            control_qubit: Single qubit index that controls the operation
            target_qubits: List of qubit indices holding the value to multiply
        """
        n_total = register.nq
        size = 2 ** n_total

        # Build controlled permutation matrix
        controlled_matrix = np.zeros((size, size), dtype=complex)

        for state_idx in range(size):
            # Check control bit value
            # the bit-twiddling here is a little obscure, but it is needed because
            # the qubits are big-endian (Qubit 0 is stored in the highest bit)
            control_bit = (state_idx >> (n_total - 1 - control_qubit)) & 1

            # Extract target value
            target_val = extract_bits(state_idx, target_qubits, n_total)

            # Decide what to do based on control bit
            if control_bit == 0:
                # Control is |0⟩: identity operation
                new_target_val = target_val
            else:
                # Control is |1⟩: modular multiplication
                if target_val < self.N:
                    new_target_val = (self.k * target_val) % self.N
                else:
                    new_target_val = target_val

            # Build new state index
            new_state_idx = replace_bits(state_idx, target_qubits, n_total, new_target_val)

            controlled_matrix[new_state_idx, state_idx] = 1

        # Apply the controlled operation
        new_amplitudes = controlled_matrix @ register.amplitudes
        register.update(np.asarray(new_amplitudes).flatten())
        return register
