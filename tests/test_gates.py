"""
Comprehensive test suite for controlled quantum gates.
Tests CNOT, SWAP, CP, and CZ gates with various qubit configurations.

Qubit ordering convention (big-endian):
- Qubit 0 is leftmost in tensor product, most significant in basis states
- For 2 qubits |q0 q1⟩: |00⟩ (idx 0), |01⟩ (idx 1), |10⟩ (idx 2), |11⟩ (idx 3)
- For 3 qubits |q0 q1 q2⟩: indices 0-7 map to |000⟩ through |111⟩
"""

import numpy as np
from register import Register
from gates import CNOT, SWAP, CP, CZ


def check_normalization(register, test_name=""):
    """Verify that the quantum state is normalized (probabilities sum to 1)."""
    prob_sum = np.sum(np.abs(register.amplitudes)**2)
    assert np.isclose(prob_sum, 1.0), f"{test_name}: State not normalized! Sum of probabilities = {prob_sum}"


def test_cnot_basic():
    """Test CNOT gate on basic 2-qubit states."""
    print("\n=== Testing CNOT Gate ===")

    # Test 1: |00⟩ → |00⟩ (control=0, target=1)
    print("\nTest 1: CNOT |00⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([1, 0, 0, 0])  # |00⟩
    CNOT().apply(r, qt=1, qc=0)
    check_normalization(r, "CNOT |00⟩")
    expected = np.array([1, 0, 0, 0])  # Should remain |00⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 2: |01⟩ → |01⟩ (control=1, target=0) - control is |1⟩ but in qubit 1
    print("\nTest 2: CNOT |01⟩ with control=1, target=0")
    r = Register(2)
    r.amplitudes = np.array([0, 1, 0, 0])  # |01⟩
    CNOT().apply(r, qt=0, qc=1)
    expected = np.array([0, 0, 0, 1])  # Should become |11⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 3: |10⟩ → |11⟩ (control=0, target=1) - control is |1⟩
    print("\nTest 3: CNOT |10⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CNOT().apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 0, 1])  # Should become |11⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 4: |11⟩ → |10⟩ (control=0, target=1) - control is |1⟩, flips target
    print("\nTest 4: CNOT |11⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 0, 1])  # |11⟩
    CNOT().apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 1, 0])  # Should become |10⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    print("\n✓ All CNOT basic tests passed!")


def test_cnot_nonadjacent():
    """Test CNOT with non-adjacent qubits."""
    print("\n=== Testing CNOT with Non-Adjacent Qubits ===")

    # Test: |100⟩ → |101⟩ (control=0, target=2, middle qubit unaffected)
    print("\nTest: CNOT |100⟩ with control=0, target=2")
    r = Register(3)
    r.amplitudes = np.array([0, 0, 0, 0, 1, 0, 0, 0])  # |100⟩
    CNOT().apply(r, qt=2, qc=0)
    expected = np.array([0, 0, 0, 0, 0, 1, 0, 0])  # Should become |101⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test: |001⟩ → |001⟩ (control=0 is |0⟩, target=2, should not flip)
    print("\nTest: CNOT |001⟩ with control=0, target=2")
    r = Register(3)
    r.amplitudes = np.array([0, 1, 0, 0, 0, 0, 0, 0])  # |001⟩
    CNOT().apply(r, qt=2, qc=0)
    expected = np.array([0, 1, 0, 0, 0, 0, 0, 0])  # Should remain |001⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    print("\n✓ All CNOT non-adjacent tests passed!")


def test_swap():
    """Test SWAP gate."""
    print("\n=== Testing SWAP Gate ===")

    # Test 1: SWAP |01⟩ → |10⟩
    print("\nTest 1: SWAP |01⟩ (swap qubits 0 and 1)")
    r = Register(2)
    r.amplitudes = np.array([0, 1, 0, 0])  # |01⟩
    SWAP().apply(r, q1=0, q2=1)
    expected = np.array([0, 0, 1, 0])  # Should become |10⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 2: SWAP |10⟩ → |01⟩
    print("\nTest 2: SWAP |10⟩ (swap qubits 0 and 1)")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    SWAP().apply(r, q1=0, q2=1)
    expected = np.array([0, 1, 0, 0])  # Should become |01⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 3: SWAP in 3-qubit system |100⟩ → |001⟩ (swap q0 and q2)
    print("\nTest 3: SWAP |100⟩ (swap qubits 0 and 2)")
    r = Register(3)
    r.amplitudes = np.array([0, 0, 0, 0, 1, 0, 0, 0])  # |100⟩
    SWAP().apply(r, q1=0, q2=2)
    expected = np.array([0, 1, 0, 0, 0, 0, 0, 0])  # Should become |001⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    print("\n✓ All SWAP tests passed!")


def test_cp():
    """Test CP (Controlled-Phase) gate."""
    print("\n=== Testing CP Gate ===")

    # Test 1: CP with φ=π on |11⟩ should flip sign
    print("\nTest 1: CP(π) on |11⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 0, 1])  # |11⟩
    CP(phi=np.pi).apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 0, -1])  # Should become -|11⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 2: CP with φ=π on |10⟩ should NOT change (control is |1⟩ but target is |0⟩)
    print("\nTest 2: CP(π) on |10⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CP(phi=np.pi).apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 1, 0])  # Should remain |10⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 3: CP with φ=π on |01⟩ should NOT change (control is |0⟩)
    print("\nTest 3: CP(π) on |01⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 1, 0, 0])  # |01⟩
    CP(phi=np.pi).apply(r, qt=1, qc=0)
    expected = np.array([0, 1, 0, 0])  # Should remain |01⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 4: CP with φ=π/2 on |11⟩ should apply phase e^(iπ/2) = i
    print("\nTest 4: CP(π/2) on |11⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 0, 1])  # |11⟩
    CP(phi=np.pi/2).apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 0, 1j])  # Should become i|11⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 5: CP on superposition (1/√2)(|10⟩ + |11⟩) with φ=π
    print("\nTest 5: CP(π) on superposition (1/√2)(|10⟩ + |11⟩)")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])
    CP(phi=np.pi).apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 1/np.sqrt(2), -1/np.sqrt(2)])
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    print("\n✓ All CP tests passed!")


def test_cz():
    """Test CZ (Controlled-Z) gate."""
    print("\n=== Testing CZ Gate ===")

    # Test 1: CZ on |11⟩ should flip sign
    print("\nTest 1: CZ on |11⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 0, 1])  # |11⟩
    CZ().apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 0, -1])  # Should become -|11⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 2: CZ on |10⟩ should NOT change
    print("\nTest 2: CZ on |10⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CZ().apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 1, 0])  # Should remain |10⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 3: CZ on |01⟩ should NOT change
    print("\nTest 3: CZ on |01⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([0, 1, 0, 0])  # |01⟩
    CZ().apply(r, qt=1, qc=0)
    expected = np.array([0, 1, 0, 0])  # Should remain |01⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 4: CZ on |00⟩ should NOT change
    print("\nTest 4: CZ on |00⟩ with control=0, target=1")
    r = Register(2)
    r.amplitudes = np.array([1, 0, 0, 0])  # |00⟩
    CZ().apply(r, qt=1, qc=0)
    expected = np.array([1, 0, 0, 0])  # Should remain |00⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    # Test 5: CZ on superposition (1/√2)(|10⟩ + |11⟩)
    print("\nTest 5: CZ on superposition (1/√2)(|10⟩ + |11⟩)")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])
    CZ().apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 1/np.sqrt(2), -1/np.sqrt(2)])
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    print("\n✓ All CZ tests passed!")


def test_superposition():
    """Test gates on superposition states."""
    print("\n=== Testing Gates on Superposition States ===")

    # Test CNOT on equal superposition
    print("\nTest: CNOT on (1/2)(|00⟩ + |01⟩ + |10⟩ + |11⟩)")
    r = Register(2)
    r.amplitudes = np.array([0.5, 0.5, 0.5, 0.5])
    CNOT().apply(r, qt=1, qc=0)
    # When control (q0) is |0⟩: |00⟩ stays |00⟩, |01⟩ stays |01⟩
    # When control (q0) is |1⟩: |10⟩ → |11⟩, |11⟩ → |10⟩
    expected = np.array([0.5, 0.5, 0.5, 0.5])  # After CNOT: |00⟩ + |01⟩ + |11⟩ + |10⟩
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: {r.amplitudes}")

    print("\n✓ All superposition tests passed!")


def test_normalization():
    """Test that quantum states remain normalized after gate operations."""
    print("\n=== Testing State Normalization ===")

    # Test 1: CNOT preserves normalization
    print("\nTest 1: CNOT preserves normalization")
    r = Register(2)
    r.amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
    CNOT().apply(r, qt=1, qc=0)
    prob_sum = np.sum(np.abs(r.amplitudes)**2)
    assert np.isclose(prob_sum, 1.0), f"Not normalized! Sum = {prob_sum}"
    print(f"✓ Pass: Probability sum = {prob_sum}")

    # Test 2: CP preserves normalization
    print("\nTest 2: CP preserves normalization")
    r = Register(2)
    r.amplitudes = np.array([0.5, 0.5, 0.5, 0.5])
    CP(phi=np.pi/4).apply(r, qt=1, qc=0)
    prob_sum = np.sum(np.abs(r.amplitudes)**2)
    assert np.isclose(prob_sum, 1.0), f"Not normalized! Sum = {prob_sum}"
    print(f"✓ Pass: Probability sum = {prob_sum}")

    # Test 3: SWAP preserves normalization
    print("\nTest 3: SWAP preserves normalization")
    r = Register(3)
    r.amplitudes = np.array([0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 0, 1/np.sqrt(3), 0, 0])
    SWAP().apply(r, q1=0, q2=2)
    prob_sum = np.sum(np.abs(r.amplitudes)**2)
    assert np.isclose(prob_sum, 1.0), f"Not normalized! Sum = {prob_sum}"
    print(f"✓ Pass: Probability sum = {prob_sum}")

    # Test 4: CZ preserves normalization
    print("\nTest 4: CZ preserves normalization")
    r = Register(2)
    r.amplitudes = np.array([0.6, 0, 0, 0.8])
    CZ().apply(r, qt=1, qc=0)
    prob_sum = np.sum(np.abs(r.amplitudes)**2)
    assert np.isclose(prob_sum, 1.0), f"Not normalized! Sum = {prob_sum}"
    print(f"✓ Pass: Probability sum = {prob_sum}")

    print("\n✓ All normalization tests passed!")


def test_cp_exotic_angles():
    """Test CP gate with exotic angles that will be used in QFT."""
    print("\n=== Testing CP with Exotic Angles ===")

    # Test 1: CP with φ=2π/3
    print("\nTest 1: CP(2π/3) on |11⟩")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 0, 1])  # |11⟩
    CP(phi=2*np.pi/3).apply(r, qt=1, qc=0)
    expected_phase = np.exp(1j * 2 * np.pi / 3)
    expected = np.array([0, 0, 0, expected_phase])
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: Phase = {r.amplitudes[3]}, expected = {expected_phase}")

    # Test 2: CP with φ=2π/8 (π/4)
    print("\nTest 2: CP(2π/8) on |11⟩")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 0, 1])  # |11⟩
    CP(phi=2*np.pi/8).apply(r, qt=1, qc=0)
    expected_phase = np.exp(1j * np.pi / 4)
    expected = np.array([0, 0, 0, expected_phase])
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: Phase = {r.amplitudes[3]}, expected = {expected_phase}")

    # Test 3: CP with φ=2π/3 on superposition
    print("\nTest 3: CP(2π/3) on superposition (1/√2)(|10⟩ + |11⟩)")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)])
    CP(phi=2*np.pi/3).apply(r, qt=1, qc=0)
    expected_phase = np.exp(1j * 2 * np.pi / 3)
    expected = np.array([0, 0, 1/np.sqrt(2), expected_phase/np.sqrt(2)])
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: Superposition with phase applied correctly")

    # Test 4: Verify CP with φ=2π/8 doesn't affect |10⟩
    print("\nTest 4: CP(2π/8) on |10⟩ should not change it")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CP(phi=2*np.pi/8).apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 1, 0])
    assert np.allclose(r.amplitudes, expected), f"Expected {expected}, got {r.amplitudes}"
    print(f"✓ Pass: |10⟩ unchanged")

    print("\n✓ All exotic angle CP tests passed!")


def test_cnot_commutativity():
    """Test that CNOT(control, target) ≠ CNOT(target, control) to catch argument order bugs."""
    print("\n=== Testing CNOT Non-Commutativity ===")

    # Test 1: CNOT(0,1) vs CNOT(1,0) on |10⟩
    print("\nTest 1: CNOT(qt=1,qc=0) vs CNOT(qt=0,qc=1) on |10⟩")

    r1 = Register(2)
    r1.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CNOT().apply(r1, qt=1, qc=0)  # Control on q0, target on q1

    r2 = Register(2)
    r2.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CNOT().apply(r2, qt=0, qc=1)  # Control on q1, target on q0

    print(f"  CNOT(qt=1,qc=0): |10⟩ → {r1.amplitudes}")
    print(f"  CNOT(qt=0,qc=1): |10⟩ → {r2.amplitudes}")
    assert not np.allclose(r1.amplitudes, r2.amplitudes), "CNOT should not be commutative!"
    print(f"✓ Pass: Different results as expected")

    # Test 2: Verify specific behavior of CNOT(0,1) on |10⟩
    print("\nTest 2: CNOT(qt=1,qc=0) on |10⟩ should give |11⟩")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CNOT().apply(r, qt=1, qc=0)
    expected = np.array([0, 0, 0, 1])  # |11⟩ (control q0=1, so flip target q1)
    assert np.allclose(r.amplitudes, expected), f"Expected |11⟩, got {r.amplitudes}"
    print(f"✓ Pass: |10⟩ → |11⟩")

    # Test 3: Verify specific behavior of CNOT(1,0) on |10⟩
    print("\nTest 3: CNOT(qt=0,qc=1) on |10⟩ should give |10⟩")
    r = Register(2)
    r.amplitudes = np.array([0, 0, 1, 0])  # |10⟩
    CNOT().apply(r, qt=0, qc=1)
    expected = np.array([0, 0, 1, 0])  # |10⟩ (control q1=0, so no flip)
    assert np.allclose(r.amplitudes, expected), f"Expected |10⟩, got {r.amplitudes}"
    print(f"✓ Pass: |10⟩ → |10⟩")

    # Test 4: CNOT on |01⟩ with swapped arguments
    print("\nTest 4: CNOT argument order on |01⟩")

    r1 = Register(2)
    r1.amplitudes = np.array([0, 1, 0, 0])  # |01⟩
    CNOT().apply(r1, qt=1, qc=0)  # q0 control, q1 target

    r2 = Register(2)
    r2.amplitudes = np.array([0, 1, 0, 0])  # |01⟩
    CNOT().apply(r2, qt=0, qc=1)  # q1 control, q0 target

    print(f"  CNOT(qt=1,qc=0): |01⟩ → {r1.amplitudes}")
    print(f"  CNOT(qt=0,qc=1): |01⟩ → {r2.amplitudes}")
    assert not np.allclose(r1.amplitudes, r2.amplitudes), "CNOT should not be commutative!"
    print(f"✓ Pass: Different results confirm correct argument handling")

    print("\n✓ All CNOT commutativity tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Gate Test Suite")
    print("=" * 60)

    try:
        test_cnot_basic()
        test_cnot_nonadjacent()
        test_swap()
        test_cp()
        test_cz()
        test_superposition()
        test_normalization()
        test_cp_exotic_angles()
        test_cnot_commutativity()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
