# Clean imports — remove duplicates and unused ones
from qiskit import QuantumCircuit, Aer, execute
from typing import List, Tuple  # Keep only if needed later

class ThreeQubitCircuit:
    """Three-qubit quantum circuit with measurement."""

    def __init__(self):
        self.circuit = QuantumCircuit(3, 3)
        self._build_circuit()

    def _build_circuit(self):
        """Builds the three-qubit entangled circuit."""
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)

    def measure_all(self):
        """Measure all qubits."""
        self.circuit.measure([0, 1, 2], [0, 1, 2])

    def run_circuit(self, shots=1024):
        """Run the circuit on a simulator."""
        self.measure_all()
        simulator = Aer.get_backend("qasm_simulator")
        result = execute(self.circuit, simulator, shots=shots).result()
        return result.get_counts()

    def quantum_cyber_tunnel(self, shots=1024):
        """Quantum-cyber tunnel example."""
        # Example new operations
        self.circuit.x(0)
        self.circuit.h(1)
        self.circuit.cx(0, 2)
        self.measure_all()
        simulator = Aer.get_backend("qasm_simulator")
        result = execute(self.circuit, simulator, shots=shots).result()
        return result.get_counts()

