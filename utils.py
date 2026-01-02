"""
Utility functions for quantum state manipulation.
"""


def extract_bits(state_idx, qubit_indices, n_total_qubits):
    """
    Extract the value stored in specific qubits from a basis state index.

    This function reads out a subset of qubits from a full quantum state
    index, respecting big-endian ordering where qubit 0 is the MSB.

    Big-Endian Bit Extraction:
        Given state |q0 q1 q2 q3 q4⟩ with qubits [1, 3, 4]:
        - Extract bits from positions q1, q3, q4
        - Return value = q1*2^2 + q3*2^1 + q4*2^0

    Args:
        state_idx: Integer index of the basis state (0 to 2^n - 1)
        qubit_indices: List of qubit positions to extract [q1, q2, ...]
        n_total_qubits: Total number of qubits in the register

    Returns:
        Integer value formed by the extracted qubits

    Example:
        >>> # State index 13 = 0b01101 represents |01101⟩
        >>> # Extract qubits [1, 3] (positions counting from 0)
        >>> extract_bits(13, [1, 3], 5)
        3  # Binary: 11 (q1=1, q3=1)
    """
    value = 0
    n_extracted = len(qubit_indices)

    for i, qubit_pos in enumerate(qubit_indices):
        # Extract bit at position qubit_pos (big-endian)
        # Shift right by (n_total - 1 - qubit_pos) to move desired bit to LSB
        bit = (state_idx >> (n_total_qubits - 1 - qubit_pos)) & 1

        # Place this bit in position i of the output value
        # Most significant extracted bit goes to MSB of output
        value |= (bit << (n_extracted - 1 - i))

    return value


def replace_bits(state_idx, qubit_indices, n_total_qubits, new_value):
    """
    Replace specific qubit values in a basis state index.

    This function modifies a subset of qubits in a full quantum state
    index, leaving other qubits unchanged.

    Big-Endian Bit Replacement:
        Given state |q0 q1 q2 q3 q4⟩, qubits [1, 3], new_value=2 (binary 10):
        1. Clear bits at positions q1, q3
        2. Set q1 = 1, q3 = 0 (from binary 10)
        3. Other qubits q0, q2, q4 remain unchanged

    Args:
        state_idx: Integer index of the basis state (0 to 2^n - 1)
        qubit_indices: List of qubit positions to replace [q1, q2, ...]
        n_total_qubits: Total number of qubits in the register
        new_value: New value to write into the specified qubits

    Returns:
        New state index with updated qubit values

    Example:
        >>> # State |01101⟩ (index 13), replace qubits [1,3] with value 2 (binary 10)
        >>> # Original: q0=0, q1=1, q2=1, q3=1, q4=1
        >>> # New:      q0=0, q1=1, q2=1, q3=0, q4=1  (q1=1, q3=0 from value 10)
        >>> replace_bits(13, [1, 3], 5, 2)
        13  # Binary: 01101 → 01101 (q1=1, q3=0)
    """
    result = state_idx
    n_bits = len(qubit_indices)

    # Clear the bits we're about to replace, then set new values
    for i, qubit_pos in enumerate(qubit_indices):
        # Extract bit i from new_value (big-endian)
        new_bit = (new_value >> (n_bits - 1 - i)) & 1

        # Calculate position of this qubit in the full state
        bit_position = n_total_qubits - 1 - qubit_pos

        # Clear the bit at this position
        result &= ~(1 << bit_position)

        # Set the new bit value
        result |= (new_bit << bit_position)

    return result
