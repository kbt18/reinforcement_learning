import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch, MagicMock

from dqn_gymnasium import DQN

# Import the DQN class
# Assuming it's in a file called dqn.py
# If it's in a different file, adjust this import
# from dqn import DQN

class TestDQN(unittest.TestCase):
    def test_initialization(self):
        """Test that the DQN model initializes correctly with different input shapes."""
        # Test with standard Atari input shape (4, 84, 84)
        input_shape = (4, 84, 84)
        num_actions = 6
        
        model = DQN(input_shape, num_actions)
        
        # Check that all expected layers exist
        self.assertIsInstance(model.conv1, nn.Conv2d)
        self.assertIsInstance(model.conv2, nn.Conv2d)
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.fc2, nn.Linear)
        
        # Check that layer dimensions are correct
        self.assertEqual(model.conv1.in_channels, 4)
        self.assertEqual(model.conv1.out_channels, 16)
        self.assertEqual(model.conv2.in_channels, 16)
        self.assertEqual(model.conv2.out_channels, 32)
        self.assertEqual(model.fc2.out_features, num_actions)
        
        # Test with a different input shape
        input_shape = (3, 64, 64)
        model = DQN(input_shape, num_actions)
        self.assertEqual(model.conv1.in_channels, 3)

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        # Standard Atari input shape
        input_shape = (4, 84, 84)
        num_actions = 6
        model = DQN(input_shape, num_actions)
        
        # Test single sample
        x = torch.randn(1, *input_shape)
        output = model(x)
        self.assertEqual(output.shape, (1, num_actions))
        
        # Test batch of samples
        batch_size = 32
        x = torch.randn(batch_size, *input_shape)
        output = model(x)
        self.assertEqual(output.shape, (batch_size, num_actions))

    def test_get_conv_output(self):
        """Test that _get_conv_output correctly calculates the flattened size."""
        input_shape = (4, 84, 84)
        num_actions = 6
        model = DQN(input_shape, num_actions)
        
        # Calculate expected output size manually
        # First conv layer: (84-8)/4 + 1 = 20.25 -> 20
        # Second conv layer: (20-4)/2 + 1 = 9.5 -> 9
        # Output shape should be 32 x 9 x 9 = 2592
        expected_size = 32 * 9 * 9
        
        # Get the actual size using the method
        actual_size = model._get_conv_output((4, 84, 84))
        
        self.assertEqual(actual_size, expected_size)

    def test_forward_conv(self):
        """Test that _forward_conv correctly processes input through convolutional layers."""
        input_shape = (4, 84, 84)
        num_actions = 6
        model = DQN(input_shape, num_actions)
        
        # Create an input tensor
        x = torch.randn(1, 4, 84, 84)
        
        # Get output from _forward_conv
        conv_output = model._forward_conv(x)
        
        # Expected shape: (1, 32, 9, 9)
        expected_shape = (1, 32, 9, 9)
        self.assertEqual(conv_output.shape, expected_shape)
        
        # Test with batch
        batch_size = 16
        x = torch.randn(batch_size, 4, 84, 84)
        conv_output = model._forward_conv(x)
        self.assertEqual(conv_output.shape, (batch_size, 32, 9, 9))

    def test_device_compatibility(self):
        """Test that the model can be moved to different devices (if available)."""
        input_shape = (4, 84, 84)
        num_actions = 6
        model = DQN(input_shape, num_actions)
        
        # Test on CPU
        model = model.to('cpu')
        x = torch.randn(1, *input_shape, device='cpu')
        output = model(x)
        self.assertEqual(output.device.type, 'cpu')
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            x = torch.randn(1, *input_shape, device='cuda')
            output = model(x)
            self.assertEqual(output.device.type, 'cuda')

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        input_shape = (4, 84, 84)
        num_actions = 6
        model = DQN(input_shape, num_actions)
        
        # Forward pass
        x = torch.randn(1, *input_shape, requires_grad=True)
        output = model(x)
        
        # Randomly select an action
        target_action = torch.randint(0, num_actions, (1,))
        loss = output[0, target_action]
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            # Check gradient is not all zeros
            self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)), 
                            f"Gradient for {name} is all zeros")

    def test_output_range(self):
        """Test that the output values are in a reasonable range for Q-values."""
        input_shape = (4, 84, 84)
        num_actions = 6
        model = DQN(input_shape, num_actions)
        
        # Create a batch of random observations
        batch_size = 32
        x = torch.randn(batch_size, *input_shape)
        
        # Get Q-values
        q_values = model(x)
        
        # Check that the values are within a reasonable range
        # DQN typically produces unbounded values before training,
        # but they shouldn't be extreme
        self.assertTrue(q_values.abs().max() < 100)

    def test_reproducibility(self):
        """Test that the network produces the same output given the same input and seed."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        input_shape = (4, 84, 84)
        num_actions = 6
        model1 = DQN(input_shape, num_actions)
        
        # Create a fixed input
        x = torch.randn(1, *input_shape)
        output1 = model1(x)
        
        # Reset seeds and create a new model
        torch.manual_seed(42)
        np.random.seed(42)
        model2 = DQN(input_shape, num_actions)
        output2 = model2(x)
        
        # Check that outputs are identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_model_save_load(self):
        """Test that the model can be saved and loaded correctly."""
        input_shape = (4, 84, 84)
        num_actions = 6
        model = DQN(input_shape, num_actions)
        
        # Create a fixed input
        x = torch.randn(1, *input_shape)
        output_before = model(x)
        
        # Save model
        torch.save(model.state_dict(), 'test_dqn_model.pt')
        
        # Load model into a new instance
        loaded_model = DQN(input_shape, num_actions)
        loaded_model.load_state_dict(torch.load('test_dqn_model.pt'))
        
        # Check output is the same
        output_after = loaded_model(x)
        self.assertTrue(torch.allclose(output_before, output_after))
        
        # Clean up
        import os
        if os.path.exists('test_dqn_model.pt'):
            os.remove('test_dqn_model.pt')


if __name__ == '__main__':
    unittest.main()