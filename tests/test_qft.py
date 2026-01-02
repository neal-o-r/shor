"""
Test QFT with states that clearly change under transformation.
"""

import numpy as np
from register import Register
from gates import X, Hadamard
from qft import apply_qft


def test_single_basis_state():
    """QFT|1⟩ → uniform magnitude with phase gradient."""
    print("\n" + "="*60)
    print("Test 1: Single Basis State |1⟩")
    print("="*60)

    register = Register(3)
    X().apply(register, 2)  # |001⟩

    print("\nInput: |001⟩")
    print(f"Amplitudes: {register.amplitudes}")

    apply_qft(register)

    print("\nAfter QFT:")
    magnitudes = np.abs(register.amplitudes)
    phases = np.angle(register.amplitudes)

    for k in range(8):
        print(f"|{k}⟩: magnitude={magnitudes[k]:.3f}, phase={phases[k]:.3f}")

    # Check all magnitudes equal
    assert np.allclose(magnitudes, 1/np.sqrt(8)), "All magnitudes should be 1/√8"

    # Check phase progression
    expected_phases = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * np.pi / 4
    # Normalize phases to [-π, π]
    expected_phases = (expected_phases + np.pi) % (2*np.pi) - np.pi
    phases_normalized = (phases + np.pi) % (2*np.pi) - np.pi
    assert np.allclose(phases_normalized, expected_phases, atol=1e-10)

    print("✓ Uniform magnitude with phase gradient")


def test_equal_superposition():
    """Equal superposition → |0⟩."""
    print("\n" + "="*60)
    print("Test 2: Equal Superposition")
    print("="*60)

    register = Register(3)
    for i in range(3):
        Hadamard().apply(register, i)

    print("\nInput: Equal superposition (all states)")
    print(f"Amplitudes: {np.abs(register.amplitudes)}")

    apply_qft(register)

    print("\nAfter QFT:")
    print(f"Amplitudes: {register.amplitudes}")

    expected = np.zeros(8)
    expected[0] = 1.0

    assert np.allclose(register.amplitudes, expected), "Should collapse to |0⟩"
    print("✓ Collapsed to |0⟩")


def test_two_adjacent_states():
    """(|0⟩ + |1⟩)/√2 → pattern with constructive/destructive interference."""
    print("\n" + "="*60)
    print("Test 3: Two Adjacent States")
    print("="*60)

    register = Register(3)
    register.amplitudes = np.zeros(8, dtype=complex)
    register.amplitudes[0] = 1/np.sqrt(2)
    register.amplitudes[1] = 1/np.sqrt(2)

    print("\nInput: (|000⟩ + |001⟩)/√2")
    print(f"Amplitudes: {register.amplitudes}")

    apply_qft(register)

    print("\nAfter QFT:")
    for k in range(8):
        mag = abs(register.amplitudes[k])
        phase = np.angle(register.amplitudes[k])
        print(f"|{k}⟩: magnitude={mag:.3f}, phase={phase:.3f}")

    # k=4 should be zero (destructive interference)
    assert abs(register.amplitudes[4]) < 1e-10, "k=4 should be zero"

    # k=0 should be 0.5 (constructive)
    assert abs(abs(register.amplitudes[0]) - 0.5) < 1e-10, "k=0 should be 0.5"

    print("✓ Shows constructive (k=0) and destructive (k=4) interference")


def test_two_qubit_bell_state():
    """Bell state (|00⟩ + |11⟩)/√2."""
    print("\n" + "="*60)
    print("Test 4: Bell State (2 qubits)")
    print("="*60)

    register = Register(2)
    register.amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

    print("\nInput: (|00⟩ + |11⟩)/√2")
    print(f"Amplitudes: {register.amplitudes}")

    apply_qft(register)

    print("\nAfter QFT:")
    for k in range(4):
        mag = abs(register.amplitudes[k])
        phase = np.angle(register.amplitudes[k])
        print(f"|{k:02b}⟩: magnitude={mag:.3f}, phase={phase:.3f}")

    # Just check it changed significantly
    input_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    assert not np.allclose(register.amplitudes, input_state), "Output should differ from input"

    print("✓ State transformed (different from input)")


def test_qft_is_unitary():
    """Verify QFT preserves norm (unitary)."""
    print("\n" + "="*60)
    print("Test 5: QFT Preserves Norm (Unitarity)")
    print("="*60)

    register = Register(3)
    # Random state
    register.amplitudes = np.random.randn(8) + 1j * np.random.randn(8)
    register.amplitudes /= np.linalg.norm(register.amplitudes)

    print("\nInput: Random normalized state")
    norm_before = np.linalg.norm(register.amplitudes)
    print(f"Norm before: {norm_before}")

    apply_qft(register)

    norm_after = np.linalg.norm(register.amplitudes)
    print(f"Norm after: {norm_after}")

    assert abs(norm_after - 1.0) < 1e-10, "Norm should be preserved"
    print("✓ Norm preserved (QFT is unitary)")


if __name__ == "__main__":
    print("="*60)
    print("QFT Transformation Tests")
    print("="*60)

    try:
        test_single_basis_state()
        test_equal_superposition()
        test_two_adjacent_states()
        test_two_qubit_bell_state()
        test_qft_is_unitary()

        print("\n" + "="*60)
        print("ALL QFT TESTS PASSED!")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
