# Shor's Algorithm:

This is a simple implementation of Shor's algorithm, that simulates the behaviour of a set of qubits as they are operated on by a set of quantum gates. 

## What Does This Do?

Shor's algorithm factors composite numbers by finding the period of modular exponentiation. The speed-up comes from using superposition and the Quantum Fourier Transform (QFT) to find this period efficiently.

For example, to factor `N=15`:
- Classical approach: Try dividing by 2, 3, 4, 5... until you find factors
- Shor's approach: Use QFT to find the period of `a^x mod 15`, then extract factors based on this period

This implementation implements both approaches side-by-side.

```
shor/
├── shors.py          # Main algorithm
├── gates.py          # Quantum gates (X, Hadamard, CNOT, etc.)
├── qft.py            # Quantum Fourier Transform
├── register.py       # Quantum register with measurement
├── utils.py          # Bit manipulation helpers
└── tests/            # Comprehensive test suite (written by Claude)
    ├── test_gates.py
    ├── test_qft.py
    └── test_shors.py
```

## How It Works

Shor's algorithm has three phases:

### 1. Classical Preprocessing
- Pick a random number `a` coprime to `N`
- Check if we got lucky with `gcd(a, N)` giving us a factor directly

### 2. Quantum Period-Finding
- Create superposition: Put the counting register in a superposition of all possible states
- Modular exponentiation: Compute `a^x mod N` for all x simultaneously 
- Measure the target register: This collapses the state to a periodic superposition
- Apply QFT: The Quantum Fourier Transform reveals the period
- Measure the result: Extract a value related to the period

### 3. Classical Post-Processing
- Use continued fractions to extract the period from the QFT measurement
- Verify the period is even (if not, try again with a different `a`)
- Compute factors using `gcd(a^(r/2) ± 1, N)`

## Quantum vs Classical

The code lets you toggle between quantum and classical period-finding:

**Quantum approach** (`use_quantum=True`):
- Uses superposition to test all values of x simultaneously
- Uses QFT to extract the period from the interference pattern
- Probabilistic: might need multiple runs

**Classical approach** (`use_quantum=False`):
- Computes a^1, a^2, a^3, ... mod N sequentially
- Stops when it finds a^r ≡ 1 (mod N)
- Deterministic: always finds the period

Both feed into the same classical post-processing, demonstrating that the quantum speedup comes purely from the period-finding step.

## Key Concepts Implemented

### Big-Endian Qubit Ordering
We use big-endian convention where qubit 0 is the most significant bit:
```
|q0 q1 q2⟩ = |101⟩ → state index 5 = 1×2² + 0×2¹ + 1×2⁰
```

### Partial Measurement
The algorithm measures the target register first (collapsing it to a specific value) while leaving the counting register in superposition.

### Quantum Fourier Transform
The QFT transforms a periodic superposition with period `r` into a state with peaks at multiples of `2^n/r`. Measuring this state gives us information about the period.

### Continued Fractions
After measuring the QFT output, we use continued fractions to approximate the measured value as a fraction `s/r`, where `r` is our desired period.

## Example Run

```
============================================================
CLASSICAL APPROACH
============================================================
Base: a=7
Using CLASSICAL period-finding
Found period: r = 4
✓ SUCCESS! 15 = 3 × 5

============================================================
QUANTUM APPROACH
============================================================
Base: a=11
Using QUANTUM period-finding
Qubits: 4 (counting) + 4 (target) = 8 total
Created superposition: (Σ|x⟩)|1⟩ / √16
Computing a^x mod N quantum mechanically...
Measured target: 1 → periodic superposition in counting register
QFT measurement: 4
Found period: r = 2
✓ SUCCESS! 15 = 5 × 3
```
