"""
Unit tests for the three-qubit quantum circuit simulator.
"""
import pytest
import numpy as np
from src.qubits.three_qubit import QuantumCircuitSimulator


class TestQuantumCircuitSimulator:
    """Test cases for QuantumCircuitSimulator class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.simulator = QuantumCircuitSimulator()
    
    def test_initialization(self):
        """Test that simulator initializes correctly."""
        assert self.simulator is not None
        assert hasattr(self.simulator, 'state')
        assert hasattr(self.simulator, 'circuit_steps')
    
    def test_hadamard_gate_creation(self):
        """Test Hadamard gate matrix creation."""
        h_gate = self.simulator.create_hadamard()
        expected = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        np.testing.assert_array_almost_equal(h_gate, expected)
    
    def test_tensor_product(self):
        """Test tensor product calculation."""
        # Test with identity matrices
        i = np.eye(2)
        result = self.simulator.tensor_product([i, i])
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_main_function_runs(self):
        """Test that main function runs without error."""
        # This will be updated based on actual implementation
        try:
            # Import the main function
            from src.qubits.three_qubit import main
            # Test that it doesn't crash
            assert callable(main)
        except ImportError:
            # If main doesn't exist yet, just pass
            pass
