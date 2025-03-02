import unittest
import numpy as np
import torch
from collections import deque
from dqn import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        # Create a buffer with a small capacity for testing
        self.buffer_capacity = 5
        self.buffer = ReplayBuffer(self.buffer_capacity)
        
        # Create some sample data
        self.sample_state = torch.zeros((1, 4, 84, 84))  # Example frame stack
        self.sample_action = 1
        self.sample_reward = 1.0
        self.sample_next_state = torch.ones((1, 4, 84, 84))
        self.sample_done = False
    
    def test_push_and_length(self):
        """Test that pushing items increases the buffer length correctly"""
        initial_len = len(self.buffer)
        
        # Push a single item
        self.buffer.push(
            self.sample_state, 
            self.sample_action, 
            self.sample_reward, 
            self.sample_next_state, 
            self.sample_done
        )
        
        self.assertEqual(len(self.buffer), initial_len + 1)
        
        # Push multiple items
        for i in range(3):
            self.buffer.push(
                self.sample_state, 
                self.sample_action + i, 
                self.sample_reward + i, 
                self.sample_next_state, 
                self.sample_done
            )
        
        self.assertEqual(len(self.buffer), initial_len + 4)
    
    def test_capacity_limit(self):
        """Test that the buffer respects its capacity limit"""
        # Fill the buffer beyond capacity
        for i in range(self.buffer_capacity + 5):
            self.buffer.push(
                self.sample_state, 
                self.sample_action, 
                self.sample_reward, 
                self.sample_next_state, 
                self.sample_done
            )
        
        # Buffer length should be equal to its capacity
        self.assertEqual(len(self.buffer), self.buffer_capacity)
    
    def test_fifo_behavior(self):
        """Test that the buffer follows First-In-First-Out behavior when capacity is reached"""
        # First, fill the buffer with distinct actions to track
        for i in range(self.buffer_capacity):
            self.buffer.push(
                self.sample_state, 
                i,  # Use i as a unique action identifier
                self.sample_reward, 
                self.sample_next_state, 
                self.sample_done
            )
        
        # Get the first experience's action
        first_action = self.buffer.memory[0][1]
        self.assertEqual(first_action, 0)  # Should be the first action we pushed
        
        # Add one more item to push out the oldest
        self.buffer.push(
            self.sample_state, 
            self.buffer_capacity,  # New unique action
            self.sample_reward, 
            self.sample_next_state, 
            self.sample_done
        )
        
        # Now the first item should have changed
        new_first_action = self.buffer.memory[0][1]
        self.assertEqual(new_first_action, 1)  # Should be the second action we pushed
        
    def test_sample_size(self):
        """Test that sample returns the correct batch size"""
        # Fill the buffer with more than the requested sample size
        for i in range(10):  # Assuming buffer capacity >= 10
            self.buffer.push(
                self.sample_state, 
                self.sample_action, 
                self.sample_reward, 
                self.sample_next_state, 
                self.sample_done
            )
        
        # Test various batch sizes
        for batch_size in [1, 2, 5]:
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            
            self.assertEqual(len(actions), batch_size)
            self.assertEqual(states.shape[0], batch_size)
            self.assertEqual(next_states.shape[0], batch_size)
            self.assertEqual(rewards.shape[0], batch_size)
            self.assertEqual(dones.shape[0], batch_size)
    
    def test_sample_randomness(self):
        """Test that samples are drawn randomly"""
        # Fill buffer with experiences that have unique actions
        for i in range(100):
            self.buffer.push(
                self.sample_state,
                i,  # Each experience has a unique action index
                self.sample_reward,
                self.sample_next_state,
                self.sample_done
            )
        
        # Take two samples and check they're not identical
        sample1_actions = self.buffer.sample(10)[1]
        sample2_actions = self.buffer.sample(10)[1]
        
        # The probability of two random samples being identical is extremely low
        self.assertFalse(torch.all(sample1_actions == sample2_actions))
    
    def test_tensor_conversion(self):
        """Test that the samples are properly converted to tensors"""
        # Fill buffer
        for i in range(10):
            self.buffer.push(
                self.sample_state,
                self.sample_action,
                self.sample_reward + i,  # Different rewards for testing
                self.sample_next_state,
                i % 2 == 0  # Alternating done flags
            )
        
        states, actions, rewards, next_states, dones = self.buffer.sample(5)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        
        # Check dtypes
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(dones.dtype, torch.bool)

    def test_empty_buffer_sample(self):
        """Test error handling when trying to sample from an empty buffer"""
        # Buffer starts empty
        empty_buffer = ReplayBuffer(self.buffer_capacity)
        
        # Trying to sample from an empty buffer should raise an exception
        with self.assertRaises(ValueError):
            empty_buffer.sample(1)
    
    def test_insufficient_items_sample(self):
        """Test error handling when trying to sample more items than available"""
        # Add just 2 items
        for i in range(2):
            self.buffer.push(
                self.sample_state,
                self.sample_action,
                self.sample_reward,
                self.sample_next_state,
                self.sample_done
            )
        
        # Trying to sample more than available should raise an exception
        with self.assertRaises(ValueError):
            self.buffer.sample(3)


if __name__ == '__main__':
    import random
    unittest.main()