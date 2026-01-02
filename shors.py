import numpy as np
from math import gcd
from register import Register
from gates import X, Hadamard, ControlledModularMultiply
from qft import apply_qft


def modular_exponentiation(register, control_qubits, target_qubits, a, N):
    """
    Computes: |x⟩|1⟩ → |x⟩|a^x mod N⟩ using repeated squaring.

    Each control qubit triggers multiplication by a^(2^k) mod N.
    """
    # Precompute powers: a^1, a^2, a^4, a^8, ... (mod N)
    powers = []
    power = a % N
    for k in range(len(control_qubits)):
        powers.append(power)
        power = (power * power) % N

    # Apply controlled multiplications (big-endian: qubit 0 controls highest power)
    n_control = len(control_qubits)
    for k, control_q in enumerate(control_qubits):
        power_index = n_control - 1 - k
        multiplier = powers[power_index]
        gate = ControlledModularMultiply(multiplier, N)
        gate.apply(register, control_q, target_qubits)

    return register


def continued_fractions_convergents(fraction, max_denominator):
    """
    [CLASSICAL] Find convergents of a fraction using continued fractions.

    This algorithm approximates a fraction with simpler fractions (smaller
    denominators). Used to extract the period r from the QFT measurement.

    Algorithm:
        1. Extract integer part: a₀ = floor(x)
        2. Take reciprocal of remainder: x ← 1/(x - a₀)
        3. Repeat to get terms [a₀, a₁, a₂, ...]
        4. Compute convergents using recurrence:
           h_n = a_n * h_{n-1} + h_{n-2}
           k_n = a_n * k_{n-1} + k_{n-2}
           where h_n/k_n is the nth convergent

    Example:
        fraction = 0.25 = 1/4
        Step 1: a₀ = 0, remainder = 0.25
        Step 2: 1/0.25 = 4, a₁ = 4
        Terms: [0, 4]
        Convergents: (0,1), (1,4) → approximations 0/1, 1/4

    Args:
        fraction: Decimal fraction to approximate
        max_denominator: Maximum allowed denominator

    Returns:
        List of (numerator, denominator) convergent pairs
    """
    if fraction == 0:
        return [(0, 1)]

    # Continued fraction expansion: extract terms [a₀, a₁, a₂, ...]
    x, terms = fraction, []
    for _ in range(20):
        if x < 1e-10:
            break
        a = int(x)  # Integer part
        terms.append(a)
        x = x - a  # Remainder
        if x < 1e-10:
            break
        x = 1 / x  # Reciprocal for next iteration

    if not terms:
        return [(0, 1)]

    # Compute convergents using recurrence relation
    # h_n/k_n where h_n = a_n * h_{n-1} + h_{n-2}
    h_prev2, h_prev1 = 0, 1
    k_prev2, k_prev1 = 1, 0
    convergents = [(h_prev1, k_prev1)]

    for a in terms:
        h = a * h_prev1 + h_prev2  # Numerator recurrence
        k = a * k_prev1 + k_prev2  # Denominator recurrence

        if k > max_denominator:
            break

        convergents.append((h, k))
        h_prev2, h_prev1 = h_prev1, h
        k_prev2, k_prev1 = k_prev1, k

    return convergents


def find_period_from_measurement(measured, n_qubits, N, a):
    """
    [CLASSICAL] Extract period from QFT measurement using continued fractions.

    Returns the period r, or None if not found.
    """
    if measured == 0:
        return None

    fraction = measured / (2**n_qubits)
    convergents = continued_fractions_convergents(fraction, N)

    # Try each convergent as a potential period
    for num, denom in convergents:
        if denom <= 1 or denom >= N:
            continue
        if pow(a, denom, N) == 1:
            return denom

    return None


def find_period_classically(a, N, max_period=None):
    """
    [CLASSICAL] Find period of a^x mod N using classical computation.

    This is a drop-in replacement for quantum period-finding. Computes
    a^1, a^2, a^3, ... mod N until finding smallest r where a^r ≡ 1 (mod N).

    Args:
        a: Base for exponentiation
        N: Modulus
        max_period: Maximum period to check (default: N)

    Returns:
        The period r, or None if not found within max_period
    """
    if max_period is None:
        max_period = N

    value = a % N
    for r in range(1, max_period):
        if value == 1:
            return r
        value = (value * a) % N

    return None


def shors_algorithm(N, a=None, n_count_qubits=None, use_quantum=True):
    """
    Shor's algorithm to factor N.

    Combines period-finding (quantum or classical) with classical pre- and post-processing.

    Args:
        N: Number to factor (should be composite and odd)
        a: Base for exponentiation (chosen randomly if None)
        n_count_qubits: Counting register size (auto: 2*ceil(log2(N)) if None)
        use_quantum: If True, use quantum period-finding; if False, use classical

    Returns:
        (factor1, factor2) if successful, None otherwise
    """
    print(f"\n{'='*60}\nShor's Algorithm: Factoring N={N}\n{'='*60}")

    # ========== CLASSICAL PREPROCESSING ==========
    a = a if a is not None else np.random.randint(2, N)
    print(f"Base: a={a}")

    # Check for trivial factor
    g = gcd(a, N)
    if g > 1:
        print(f"Lucky! gcd({a}, {N}) = {g}")
        return (g, N // g)

    # Initialize quantum registers
    n_target = int(np.ceil(np.log2(N)))
    n_count_qubits = n_count_qubits if n_count_qubits else 2 * n_target
    n_total = n_count_qubits + n_target

    # ========== PERIOD FINDING ==========
    if use_quantum:
        print(f"Using QUANTUM period-finding")
        print(f"Qubits: {n_count_qubits} (counting) + {n_target} (target) = {n_total} total")

        register = Register(n_total)
        control_qubits = list(range(n_count_qubits))
        target_qubits = list(range(n_count_qubits, n_total))

        # Prepare initial state: |0...0⟩|1⟩
        X().apply(register, target_qubits[-1])

        # Create superposition in counting register
        for q in control_qubits:
            Hadamard().apply(register, q)
        print(f"Created superposition: (Σ|x⟩)|1⟩ / √{2**n_count_qubits}")

        # Quantum modular exponentiation: |x⟩|1⟩ → |x⟩|a^x mod N⟩
        print(f"Computing a^x mod N quantum mechanically...")
        modular_exponentiation(register, control_qubits, target_qubits, a, N)

        # Measure target register (collapses to periodic superposition)
        measured_target = register.measure_qubits(target_qubits)
        print(f"Measured target: {measured_target} → periodic superposition in counting register")

        # Apply QFT to extract period
        apply_qft(register, start=0, n_qubits=n_count_qubits)
        measured_count = register.measure_qubits(control_qubits)
        print(f"QFT measurement: {measured_count}")

        # Extract period from QFT measurement
        period = find_period_from_measurement(measured_count, n_count_qubits, N, a)
    else:
        print(f"Using CLASSICAL period-finding")
        period = find_period_classically(a, N)
        if period:
            print(f"Found period: r = {period}")

    # ========== CLASSICAL POST-PROCESSING ==========

    if period is None:
        print("Failed to extract period. Try again.")
        return None

    print(f"Found period: r = {period}")

    # Verify period is even
    if period % 2 != 0:
        print("Period is odd, cannot use. Try again.")
        return None

    # Compute factors
    half_power = pow(a, period // 2, N)

    if half_power == N - 1:
        print(f"{a}^({period}//2) ≡ -1 (mod {N}), cannot use. Try again.")
        return None

    factor1 = gcd(half_power - 1, N)
    factor2 = gcd(half_power + 1, N)

    # Check for non-trivial factors
    if factor1 > 1 and factor1 < N:
        print(f"SUCCESS! {N} = {factor1} × {N//factor1}")
        return (factor1, N // factor1)

    if factor2 > 1 and factor2 < N:
        print(f"SUCCESS! {N} = {factor2} × {N//factor2}")
        return (factor2, N // factor2)

    print("Found trivial factors. Try again.")
    return None


if __name__ == "__main__":
    N = 15

    # Demonstrate classical approach
    print("\n" + "="*60)
    print("CLASSICAL APPROACH")
    print("="*60)
    result_classical = shors_algorithm(N, use_quantum=False)
    if result_classical:
        print(f"\n✓ SUCCESS! {N} = {result_classical[0]} × {result_classical[1]}")
    else:
        print(f"\n✗ Failed to factor {N}")

    # Demonstrate quantum approach
    print("\n" + "="*60)
    print("QUANTUM APPROACH")
    print("="*60)
    result_quantum = shors_algorithm(N, n_count_qubits=4, use_quantum=True)
    if result_quantum:
        print(f"\n✓ SUCCESS! {N} = {result_quantum[0]} × {result_quantum[1]}")
    else:
        print(f"\n✗ Failed this run (probabilistic algorithm - try again)")
