import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define hyperparameters
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 32  # Batch size for training
BUFFER_SIZE = 100000  # Replay buffer size
MIN_REPLAY_SIZE = 10000  # Minimum replay buffer size before training
EPSILON_START = 1.0  # Starting epsilon value
EPSILON_END = 0.1  # Final epsilon value
EPSILON_DECAY = 1000000  # Number of frames to decay epsilon
TARGET_UPDATE_FREQ = 10000  # How often to update target network
LEARNING_RATE = 0.00025  # Learning rate for optimizer

# Define DQN model architecture
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Initialize Deep Q-Network
        
        Parameters:
            input_shape (tuple): The shape of the input (C, H, W)
            num_actions (int): Number of possible actions
        """
        super(DQN, self).__init__()
        
        # TODO: Define your convolutional neural network architecture
        # Hint: Standard architecture for Atari is:
        # - Conv layers with nonlinearities
        # - Flatten
        # - Fully connected layers
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)

        out_dims = self._get_conv_output((c, h, w))
        self.fc1 = nn.Linear(out_dims, 256)
        self.fc2 = nn.Linear(256, num_actions)
        
        # Calculate the size of the features after convolutions
        # TODO: Replace this placeholder calculation with the correct one for your architecture
    
    def _get_conv_output(self, shape):
        """
        Calculate the output size of the convolutional layers
        """
        o = self._forward_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _forward_conv(self, x):
        """
        Forward pass through convolutional layers
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
    
    def forward(self, x):
        """
        Forward pass through the entire network
        """
        # TODO: Implement the forward pass
        # Hint: Remember to handle batch dimension correctly

        x = self._forward_conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Replay Buffer for storing experiences
class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Initialize Replay Buffer
        
        Parameters:
            buffer_size (int): Maximum size of the replay buffer
        """
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to the buffer
        """
        # TODO: Implement experience addition to buffer
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        """
        # TODO: Implement sampling of batch_size experiences

    
    def __len__(self):
        """
        Return the current size of the buffer
        """
        return len(self.buffer)


# Preprocessing for Atari environments
class AtariPreprocessing:
    def __init__(self, env, frame_skip=4, frame_size=84, stack_frames=4):
        """
        Preprocessing wrapper for Atari environments
        
        Parameters:
            env (gym.Env): Gym environment
            frame_skip (int): Number of frames to skip
            frame_size (int): Size to resize frames to (frame_size x frame_size)
            stack_frames (int): Number of frames to stack
        """
        self.env = env
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.stack_frames = stack_frames
        
        # Initialize frame stack
        self.frames = deque(maxlen=stack_frames)
        
        # Get action space and observation space
        self.action_space = env.action_space
        
        # Define new observation shape (stack_frames, frame_size, frame_size)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(stack_frames, frame_size, frame_size),
            dtype=np.uint8
        )
    
    def reset(self):
        """Reset environment and preprocess initial observation"""
        # TODO: Implement reset functionality
        # 1. Reset the environment
        # 2. Preprocess the initial observation
        # 3. Stack the initial frame the required number of times
        
    
    def step(self, action):
        """Execute action and preprocess the resulting observation"""
        # TODO: Implement step functionality
        # 1. Skip frames
        # 2. Preprocess the observation
        # 3. Update the frame stack
    
    def _preprocess_observation(self, observation):
        """
        Preprocess a single observation
        Convert to grayscale, resize, etc.
        """
        # TODO: Implement preprocessing
        # 1. Convert to grayscale (take mean across channels)
        # 2. Resize to frame_size x frame_size
        # 3. Convert to numpy array of appropriate type
        
        # Note: In a real implementation, you would use libraries like OpenCV
        # For the skeleton, we'll use a placeholder
        
        # Simplified placeholder
        processed_obs = np.mean(observation, axis=2).astype(np.uint8)
        
        # Here you would resize to frame_size x frame_size
        # For example: processed_obs = cv2.resize(processed_obs, (self.frame_size, self.frame_size))
        
        return processed_obs
    
    def _get_observation(self):
        """
        Get the current observation from the frame stack
        """
        # Convert the frame stack to a numpy array
        return np.array(self.frames)
    
    def render(self):
        """Render the environment"""
        return self.env.render()


# Main DQN agent class
class DQNAgent:
    def __init__(self, env, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize DQN Agent
        
        Parameters:
            env: Gymnasium environment
            device (str): Device to run the models on (cuda or cpu)
        """
        self.env = env
        self.device = device
        
        # Get state and action space
        self.num_actions = env.action_space.n
        self.state_shape = env.observation_space.shape
        
        # Initialize Q-networks (online and target)
        self.online_net = DQN(self.state_shape, self.num_actions).to(device)
        self.target_net = DQN(self.state_shape, self.num_actions).to(device)
        
        # Copy weights from online to target network
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        
        # Initialize step counter
        self.steps = 0
    
    def select_action(self, state, epsilon):
        """
        Select an action using epsilon-greedy policy
        
        Parameters:
            state: Current state
            epsilon (float): Probability of selecting a random action
        
        Returns:
            int: Selected action
        """
        # TODO: Implement epsilon-greedy action selection
        # 1. With probability epsilon, select a random action
        # 2. Otherwise, select the action with the highest Q-value
        
    
    def train_step(self):
        """
        Perform a single training step
        
        Returns:
            float: Loss value
        """
        # TODO: Implement DQN training step
        # 1. Sample a batch from the replay buffer
        # 2. Calculate the current Q-values
        # 3. Calculate the target Q-values
        # 4. Calculate the loss
        # 5. Perform backpropagation and optimization
        
    
    def update_target_network(self):
        """
        Update the target network with the parameters from the online network
        """
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def get_epsilon(self):
        """
        Get epsilon value based on the current step
        
        Returns:
            float: Current epsilon value
        """
        # Linear annealing from EPSILON_START to EPSILON_END over EPSILON_DECAY steps
        return max(EPSILON_END, EPSILON_START - (self.steps / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
    
    def train(self, num_frames):
        """
        Train the agent
        
        Parameters:
            num_frames (int): Total number of frames to train for
        
        Returns:
            list: Episode rewards
        """
        # Lists to store metrics
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        # Initialize environment
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Fill replay buffer with random actions before training
        print("Filling replay buffer...")
        while len(self.replay_buffer) < MIN_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                state = self.env.reset()
        
        print("Training...")
        for frame in tqdm(range(num_frames)):
            # Get epsilon for current step
            epsilon = self.get_epsilon()
            
            # Select action
            action = self.select_action(state, epsilon)
            
            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Add experience to replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Update episode statistics
            episode_reward += reward
            episode_length += 1
            
            # Train the network
            loss = self.train_step()
            losses.append(loss)
            
            # Update target network if needed
            if self.steps % TARGET_UPDATE_FREQ == 0:
                self.update_target_network()
            
            # Reset if episode is done
            if done:
                state = self.env.reset()
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Print progress
                if len(episode_rewards) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode: {len(episode_rewards)}, Avg. Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
                
                # Reset episode statistics
                episode_reward = 0
                episode_length = 0
            
            # Increment step counter
            self.steps += 1
        
        return episode_rewards
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the agent
        
        Parameters:
            num_episodes (int): Number of episodes to evaluate
            render (bool): Whether to render the environment
        
        Returns:
            float: Average reward over episodes
        """
        # TODO: Implement evaluation
        # 1. Run the agent for num_episodes with epsilon=0
        # 2. Return the average reward
    
        print(f"Average Evaluation Reward: {avg_reward:.2f}")
    
    def save(self, path):
        """
        Save the agent's models
        
        Parameters:
            path (str): Path to save the models to
        """
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
    
    def load(self, path):
        """
        Load the agent's models
        
        Parameters:
            path (str): Path to load the models from
        """
        checkpoint = torch.load(path)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']


# Main training loop
def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create Atari environment
    # For example: "PongNoFrameskip-v4", "BreakoutNoFrameskip-v4"
    env_name = "PongNoFrameskip-v4"
    env = gym.make(env_name)
    
    # Apply preprocessing
    env = AtariPreprocessing(env)
    
    # Create DQN agent
    agent = DQNAgent(env)
    
    # Train the agent
    num_frames = 1000000  # Set to a smaller value for testing
    rewards = agent.train(num_frames)
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f"DQN Training on {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"{env_name}_rewards.png")
    plt.show()
    
    # Save the agent
    agent.save(f"{env_name}_dqn.pt")
    
    # Evaluate the agent
    agent.evaluate(num_episodes=5, render=True)


if __name__ == "__main__":
    main()