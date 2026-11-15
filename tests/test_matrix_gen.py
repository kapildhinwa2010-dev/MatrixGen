"""
Tests for MatrixGen module
"""

import math
import pytest
import torch
import torch.nn as nn
from matrix_gen import (
    MatrixGenerator,
    MatrixGenWrapper,
    MatrixGenConfig,
    create_matrix_generator
)


class TestMatrixGeneratorInitialization:
    """Test MatrixGenerator initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        gen = MatrixGenerator()
        assert gen.hidden_size == 128
        assert gen.num_layers == 4
        assert gen.chunk_size == 16384
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        gen = MatrixGenerator(
            hidden_size=64,
            num_layers=3,
            activation='relu',
            chunk_size=8192
        )
        assert gen.hidden_size == 64
        assert gen.num_layers == 3
        assert gen.chunk_size == 8192
    
    def test_network_architecture(self):
        """Test network architecture"""
        gen = MatrixGenerator(hidden_size=64, num_layers=3)
        # Count layers in net
        linear_layers = sum(1 for l in gen.net if isinstance(l, nn.Linear))
        assert linear_layers == 3  # 3 layers (input + 2 hidden + output)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        gen = MatrixGenerator(hidden_size=64, num_layers=3)
        count = gen.count_parameters()
        # 2 input -> 64 + 64 -> 64 + 64 -> 1
        expected = 2*64 + 64 + 64*64 + 64 + 64*1 + 1
        assert count == expected


class TestMatrixGeneratorGeneration:
    """Test matrix generation"""
    
    def test_generate_small_matrix(self):
        """Test generating small matrix"""
        gen = MatrixGenerator()
        W = gen.generate_matrix(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_generate_square_matrix(self):
        """Test generating square matrix"""
        gen = MatrixGenerator()
        W = gen.generate_matrix(100, 100)
        assert W.shape == (100, 100)
    
    def test_generate_rectangular_matrix(self):
        """Test generating rectangular matrix"""
        gen = MatrixGenerator()
        W = gen.generate_matrix(200, 100)
        assert W.shape == (200, 100)
    
    def test_generate_matrix_device(self):
        """Test weight generation on CPU"""
        gen = MatrixGenerator()
        W = gen.generate_matrix(10, 10, device=torch.device('cpu'))
        assert W.device.type == 'cpu'
    
    def test_weight_matrix_properties(self):
        """Test weight matrix statistical properties"""
        gen = MatrixGenerator()
        W = gen.generate_matrix(100, 100)
        # Check that weights are not all zeros
        assert W.abs().sum() > 0
        # Check that weights have reasonable scale
        assert W.std() > 0


class TestMatrixGeneratorChunkedProcessing:
    """Test chunked processing"""
    
    def test_chunked_vs_unchunked(self):
        """Test that chunked and unchunked produce consistent results"""
        gen1 = MatrixGenerator(chunk_size=10000)
        gen2 = MatrixGenerator(chunk_size=100)
        
        # Copy parameters from gen1 to gen2
        gen2.net.load_state_dict(gen1.net.state_dict())
        
        W1 = gen1.generate_matrix(100, 100)
        W2 = gen2.generate_matrix(100, 100)
        
        # Should be close (may differ slightly due to chunking)
        assert torch.allclose(W1, W2, rtol=1e-4, atol=1e-4)
    
    def test_large_matrix_chunking(self):
        """Test that large matrices are processed correctly"""
        gen = MatrixGenerator(chunk_size=100)
        W = gen.generate_matrix(500, 500)
        assert W.shape == (500, 500)
        # Check consistency
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


class TestMatrixGeneratorBatch:
    """Test batch generation"""
    
    def test_batch_generation(self):
        """Test batch generation of multiple matrices"""
        gen = MatrixGenerator()
        shapes = [
            ('W1', 10, 10),
            ('W2', 20, 10),
            ('W3', 10, 20),
        ]
        weights = gen.generate_batch(shapes)
        
        assert 'W1' in weights
        assert 'W2' in weights
        assert 'W3' in weights
        assert weights['W1'].shape == (10, 10)
        assert weights['W2'].shape == (20, 10)
        assert weights['W3'].shape == (10, 20)


class TestMatrixGenWrapper:
    """Test MatrixGenWrapper"""
    
    def test_layer_weights_generation(self):
        """Test generating layer weights"""
        wrapper = MatrixGenWrapper()
        layer_config = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = wrapper.generate_layer_weights(layer_config)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)
    
    def test_model_weights_generation(self):
        """Test generating model weights"""
        wrapper = MatrixGenWrapper()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        weights = wrapper.generate_model_weights(model)
        # Should have weights for both linear layers
        assert len(weights) >= 2


class TestCreateMatrixGenerator:
    """Test create_matrix_generator factory function"""
    
    def test_default_creation(self):
        """Test default generator creation"""
        gen = create_matrix_generator()
        assert isinstance(gen, MatrixGenerator)
        assert gen.hidden_size == 128
        assert gen.num_layers == 4
    
    def test_custom_creation(self):
        """Test custom generator creation"""
        gen = create_matrix_generator(
            hidden_size=64,
            num_layers=3,
            activation='gelu',
            chunk_size=8192
        )
        assert gen.hidden_size == 64
        assert gen.num_layers == 3


class TestMatrixGenConfig:
    """Test MatrixGenConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = MatrixGenConfig()
        assert config.hidden_size == 128
        assert config.num_layers == 4
        assert config.activation == "silu"
        assert config.chunk_size == 16384
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MatrixGenConfig(
            hidden_size=64,
            num_layers=3,
            activation='relu'
        )
        assert config.hidden_size == 64
        assert config.num_layers == 3
        assert config.activation == 'relu'


class TestActivationFunctions:
    """Test different activation functions"""
    
    def test_silu_activation(self):
        """Test SiLU activation"""
        gen = MatrixGenerator(activation='silu')
        W = gen.generate_matrix(10, 10)
        assert W.shape == (10, 10)
    
    def test_relu_activation(self):
        """Test ReLU activation"""
        gen = MatrixGenerator(activation='relu')
        W = gen.generate_matrix(10, 10)
        assert W.shape == (10, 10)
    
    def test_gelu_activation(self):
        """Test GELU activation"""
        gen = MatrixGenerator(activation='gelu')
        W = gen.generate_matrix(10, 10)
        assert W.shape == (10, 10)
    
    def test_tanh_activation(self):
        """Test Tanh activation"""
        gen = MatrixGenerator(activation='tanh')
        W = gen.generate_matrix(10, 10)
        assert W.shape == (10, 10)


class TestGradientFlow:
    """Test gradient flow through generator"""
    
    def test_gradients_exist(self):
        """Test that gradients flow through generator"""
        gen = MatrixGenerator()
        
        # Generate weights with gradient tracking
        W = gen.generate_matrix(10, 10)
        loss = W.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for param in gen.net.parameters():
            assert param.grad is not None


class TestDeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        gen = MatrixGenerator()
        W = gen.generate_matrix(10, 10)
        for param in gen.net.parameters():
            assert param.device == W.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        gen = MatrixGenerator()
        W = gen.generate_matrix(10, 10)
        for param in gen.net.parameters():
            assert param.dtype == W.dtype


class TestXavierScaling:
    """Test Xavier initialization scaling"""
    
    def test_xavier_scaling(self):
        """Test that weights are properly scaled"""
        gen = MatrixGenerator()
        
        # Generate matrix
        W = gen.generate_matrix(100, 100)
        
        # Check that variance is reasonable for Xavier initialization
        # For Xavier uniform, variance should be ~2/(fan_in + fan_out)
        expected_std = math.sqrt(2.0 / (100 + 100))
        actual_std = W.std().item()
        
        # Allow some tolerance
        assert abs(actual_std - expected_std) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
