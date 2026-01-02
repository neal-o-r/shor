"""
Comprehensive tests for Shor's algorithm components.
"""

import numpy as np
from register import Register
from gates import X, Hadamard, ModularMultiply, ControlledModularMultiply
from shors import modular_exponentiation


def test_modular_multiply():
    """Test that modular multiplication works."""
    print("\n" + "="*60)
    print("Testing Modular Multiplication")
    print("="*60)

    register = Register(4)  # 4 qubits for values 0-15

    # Test 1: |1⟩ → |7⟩
    print("\nTest 1: Multiply |1⟩ by 7 mod 15")
    X().apply(register, 3)  # Initialize to |1⟩

    gate = ModularMultiply(7, 15)
    gate.apply(register, [0, 1, 2, 3])

    # Check result is |7⟩
    assert abs(register.amplitudes[7] - 1.0) < 0.01, "Should be |7⟩"
    print("  ✓ Result: |7⟩")

    # Test 2: |7⟩ → |4⟩ (since 7×7 = 49 mod 15 = 4)
    print("\nTest 2: Multiply |7⟩ by 7 mod 15")
    gate.apply(register, [0, 1, 2, 3])

    assert abs(register.amplitudes[4] - 1.0) < 0.01, "Should be |4⟩"
    print("  ✓ Result: |4⟩")

    # Test 3: Verify it forms cycles (period detection)
    print("\nTest 3: Verify cycling back to |1⟩")
    gate.apply(register, [0, 1, 2, 3])  # |4⟩ → |13⟩
    gate.apply(register, [0, 1, 2, 3])  # |13⟩ → |1⟩

    assert abs(register.amplitudes[1] - 1.0) < 0.01, "Should cycle back to |1⟩"
    print("  ✓ Cycle: |1⟩ → |7⟩ → |4⟩ → |13⟩ → |1⟩ (period 4)")

    print("\n✓ All modular multiplication tests passed!")


def test_controlled_modular_multiply():
    """Test controlled modular multiplication."""
    print("\n" + "="*60)
    print("Testing Controlled Modular Multiplication")
    print("="*60)

    # Test 1: Control = |0⟩, should do nothing
    print("\nTest 1: Control=|0⟩, Target=|1⟩")
    register = Register(5)  # 1 control + 4 target
    X().apply(register, 4)  # Target = |1⟩

    gate = ControlledModularMultiply(7, 15)
    gate.apply(register, control_qubit=0, target_qubits=[1, 2, 3, 4])

    assert abs(register.amplitudes[1] - 1.0) < 0.01
    print("  ✓ Target unchanged: |1⟩")

    # Test 2: Control = |1⟩, should multiply
    print("\nTest 2: Control=|1⟩, Target=|1⟩")
    register = Register(5)
    X().apply(register, 0)  # Control = |1⟩
    X().apply(register, 4)  # Target = |1⟩

    gate.apply(register, control_qubit=0, target_qubits=[1, 2, 3, 4])

    # State should be |10111⟩ = 23
    assert abs(register.amplitudes[23] - 1.0) < 0.01
    print("  ✓ Target multiplied: |7⟩")

    # Test 3: Superposition control (entanglement)
    print("\nTest 3: Control in superposition")
    register = Register(5)
    Hadamard().apply(register, 0)  # Control = (|0⟩+|1⟩)/√2
    X().apply(register, 4)          # Target = |1⟩

    gate.apply(register, control_qubit=0, target_qubits=[1, 2, 3, 4])

    # Should be (|0⟩|1⟩ + |1⟩|7⟩)/√2
    assert abs(register.amplitudes[1] - 1/np.sqrt(2)) < 0.01
    assert abs(register.amplitudes[23] - 1/np.sqrt(2)) < 0.01
    print("  ✓ Creates entanglement: (|0⟩|1⟩ + |1⟩|7⟩)/√2")

    print("\n✓ All controlled multiplication tests passed!")


def test_modular_exponentiation():
    """Test the full modular exponentiation."""
    print("\n" + "="*60)
    print("Testing Modular Exponentiation")
    print("="*60)

    # 6 qubits: 2 control, 4 target
    register = Register(6)

    # Initialize target to |1⟩
    X().apply(register, 5)

    # Put control in superposition
    Hadamard().apply(register, 0)
    Hadamard().apply(register, 1)

    print("\nInitial: (|00⟩+|01⟩+|10⟩+|11⟩)|1⟩ / 2")

    # Apply modular exponentiation: 7^x mod 15
    modular_exponentiation(
        register,
        control_qubits=[0, 1],
        target_qubits=[2, 3, 4, 5],
        a=7,
        N=15
    )

    print("\nExpected states:")
    print("  |00⟩|1⟩:  7^0 = 1")
    print("  |01⟩|7⟩:  7^1 = 7")
    print("  |10⟩|4⟩:  7^2 = 49 mod 15 = 4")
    print("  |11⟩|13⟩: 7^3 = 343 mod 15 = 13")

    # Verify expected states
    expected = {
        0b000001: 0.5,   # |00⟩|1⟩
        0b010111: 0.5,   # |01⟩|7⟩
        0b100100: 0.5,   # |10⟩|4⟩
        0b111101: 0.5,   # |11⟩|13⟩
    }

    print("\nVerifying amplitudes:")
    all_correct = True
    for state, expected_amp in expected.items():
        actual_amp = abs(register.amplitudes[state])
        control = state >> 4
        target = state & 0b1111
        match = abs(actual_amp - expected_amp) < 0.01
        symbol = "✓" if match else "✗"
        print(f"  {symbol} |{control:02b}⟩|{target:04b}⟩: {actual_amp:.3f} (expected {expected_amp:.3f})")
        all_correct = all_correct and match

    assert all_correct, "Some amplitudes incorrect"
    print("\n✓ All modular exponentiation tests passed!")


if __name__ == "__main__":
    print("="*60)
    print("Shor's Algorithm Component Tests")
    print("="*60)

    try:
        test_modular_multiply()
        test_controlled_modular_multiply()
        test_modular_exponentiation()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
