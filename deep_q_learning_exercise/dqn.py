import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gym.wrappers import FrameStack, AtariPreprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Hyperparameters - feel free to experiment with these
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000  # Number of frames to decay epsilon over
MEMORY_SIZE = 100000  # Experience replay buffer size
TARGET_UPDATE = 10000  # Update target network every N frames
LEARNING_RATE = 0.0001
NUM_FRAMES = 4  # Input frame stack size
TOTAL_FRAMES = 10000000  # Total training frames

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Deep Q-Network architecture
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        
        # TODO: Implement the DQN architecture here
        # Hint: For Atari, you'll want convolutional layers followed by fully connected layers
        # Example structure (from the original DQN paper):
        # - Conv layers to process the visual input
        # - Fully connected layers to produce action values
        
        # Placeholder implementation (you should replace this):
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the feature maps after convolutions
        self.fc_input_dims = 7 * 7 * 64  # This depends on input size and conv layers
        
        self.fc1 = nn.Linear(self.fc_input_dims, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Hint: Apply activation functions between layers (typically ReLU)
        
        # Placeholder implementation (you should improve this):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # TODO: Implement the storage of experience tuples
        # Hint: Consider storing experiences as (state, action, reward, next_state, done)
        pass
    
    def sample(self, batch_size):
        # TODO: Implement random sampling of experience tuples
        # Hint: Return a batch of (state, action, reward, next_state, done) tuples
        pass
    
    def __len__(self):
        return len(self.memory)

# Epsilon-greedy action selection
def select_action(state, policy_net, epsilon, num_actions):
    # TODO: Implement epsilon-greedy action selection
    # Hint:
    # - With probability epsilon, select a random action
    # - Otherwise, select the action with the highest Q-value from the policy network
    pass

# Preprocess the environment
def make_atari_env(env_name):
    # TODO: Set up the Atari environment with appropriate wrappers
    # Hint: Use AtariPreprocessing for resizing, grayscaling, frame skipping
    # and FrameStack to stack multiple frames as input to the DQN
    env = gym.make(env_name)
    
    # Example preprocessing (improve or modify as needed):
    env = AtariPreprocessing(
        env,
        frame_skip=4,  # Skip frames for faster training
        grayscale_obs=True,  # Convert to grayscale
        scale_obs=True,  # Scale observations to [0, 1]
        terminal_on_life_loss=False  # End episode on life loss
    )
    env = FrameStack(env, NUM_FRAMES)  # Stack frames for temporal information
    
    return env

# Training function
def train_dqn(env_name="PongNoFrameskip-v4"):
    # Create environment
    env = make_atari_env(env_name)
    
    # Get state and action dimensions
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Create networks
    policy_net = DQN(state_shape[0], num_actions).to(device)
    target_net = DQN(state_shape[0], num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is only used for inference
    
    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    # Create replay buffer
    memory = ReplayBuffer(MEMORY_SIZE)
    
    # Track progress
    rewards = []
    episode_reward = 0
    episode_count = 0
    
    # Initialize state
    state = env.reset()
    
    # Main training loop
    for frame_idx in tqdm(range(1, TOTAL_FRAMES + 1)):
        # Calculate epsilon for exploration
        epsilon = max(EPSILON_END, EPSILON_START - (frame_idx / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
        
        # Select and perform an action
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        action = select_action(state_tensor, policy_net, epsilon, num_actions)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # Store transition in replay memory
        memory.push(state, action, reward, next_state, done)
        state = next_state
        
        # Reset environment if done
        if done:
            state = env.reset()
            rewards.append(episode_reward)
            episode_reward = 0
            episode_count += 1
            print(f"Episode {episode_count}, Avg Reward: {np.mean(rewards[-100:]):.2f}, Epsilon: {epsilon:.2f}")
        
        # Skip update if buffer doesn't have enough samples
        if len(memory) < BATCH_SIZE:
            continue
        
        # Perform experience replay and update the network
        # TODO: Implement the training step
        # Hint:
        # 1. Sample a batch from the replay buffer
        # 2. Compute Q-values and target Q-values
        # 3. Compute loss (typically MSE or Huber loss)
        # 4. Perform backpropagation and optimization
        
        # Periodically update the target network
        if frame_idx % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Periodically save the model and plot results
        if frame_idx % 100000 == 0:
            torch.save(policy_net.state_dict(), f"{env_name}_{frame_idx}.pth")
            # Optionally plot learning curve
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.title(f"DQN Training - {env_name}")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig(f"{env_name}_learning_curve_{frame_idx}.png")
    
    # Final save
    torch.save(policy_net.state_dict(), f"{env_name}_final.pth")
    env.close()
    return policy_net

# Function to watch the trained agent play
def watch_agent(env_name, model_path, num_episodes=5):
    # TODO: Implement a function to visualize the trained agent's performance
    # Hint:
    # 1. Load the trained model
    # 2. Create the environment
    # 3. Run episodes with the trained policy (no exploration)
    # 4. Render the environment to see the agent in action
    pass

if __name__ == "__main__":
    # Train the agent
    trained_model = train_dqn()
    
    # Watch the trained agent play
    watch_agent("PongNoFrameskip-v4", "PongNoFrameskip-v4_final.pth")