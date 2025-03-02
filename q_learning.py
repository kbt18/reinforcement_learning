import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorld:
    def __init__(self):
        self.grid_size = 10
        
        # Initialize grid with zeros
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Place obstacles (1), coins (2), and traps (3)
        # Obstacles
        self.obstacles = [
            (1, 1), (1, 2), (1, 3),
            (3, 3), (3, 4), (3, 5),
            (5, 1), (5, 2),
            (7, 7), (7, 8),
            (8, 3), (8, 4), (8, 5)
        ]
        
        # Coins
        self.coins = [
            (0, 4), (2, 2),
            (4, 4), (4, 7),
            (6, 1), (6, 8),
            (9, 2)
        ]
        
        # Traps
        self.traps = [
            (2, 7), (2, 8),
            (5, 5), (7, 2),
            (8, 8)
        ]
        
        # Place elements on grid
        for x, y in self.obstacles:
            self.grid[x, y] = 1
        for x, y in self.coins:
            self.grid[x, y] = 2
        for x, y in self.traps:
            self.grid[x, y] = 3
            
        self.start = (0, 0)
        self.goal = (9, 9)
        self.agent_pos = self.start
        
        # Keep track of collected coins
        self.collected_coins = set()
        
        # Action space: up, right, down, left
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
    def reset(self):
        self.agent_pos = self.start
        self.collected_coins = set()
        return self.agent_pos
    
    def step(self, action):
        """
        Execute action and return new state, reward, and whether episode is done
        """
        new_pos = (
            self.agent_pos[0] + self.actions[action][0],
            self.agent_pos[1] + self.actions[action][1]
        )
        
        # Check if new position is valid
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            
            # Check different cell types
            cell_type = self.grid[new_pos]
            
            if cell_type == 1:  # Hit obstacle
                return self.agent_pos, -50, True
            
            self.agent_pos = new_pos
            
            if self.agent_pos == self.goal:  # Reached goal
                return self.agent_pos, 10000, True
            
            elif cell_type == 2 and new_pos not in self.collected_coins:  # Collect coin
                self.collected_coins.add(new_pos)
                return self.agent_pos, 10, False
                
            elif cell_type == 3:  # Hit trap
                return self.agent_pos, -20, False
                
            return self.agent_pos, 2, False  # Normal move
            
        return self.agent_pos, -50, True  # Hit wall
    
    def render(self):
        """
        Visualize the grid world
        """
        plt.figure(figsize=(8, 8))
        plt.grid(True)
        
        # Plot grid elements
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:  # Obstacles
                    plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'gray')
                elif self.grid[i, j] == 2:  # Coins
                    if (i, j) not in self.collected_coins:
                        plt.plot(j+0.5, i+0.5, 'yo', markersize=15)
                elif self.grid[i, j] == 3:  # Traps
                    plt.plot(j+0.5, i+0.5, 'rx', markersize=15)
        
        # Plot agent
        plt.plot(self.agent_pos[1]+0.5, self.agent_pos[0]+0.5, 'bo', markersize=15)
        
        # Plot goal
        plt.plot(self.goal[1]+0.5, self.goal[0]+0.5, 'g*', markersize=20)
        
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.show()

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, state_size, action_size))
        
        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.97  # Increased because path is longer
        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # Slower decay for more exploration
        self.epsilon_min = 0.01
        
    def get_action(self, state):
        take_random_action = random.uniform(0, 1)
        if take_random_action <= self.epsilon:
            return self.random_action()
        return self.greedy_action(state)

    def random_action(self):
        return random.randint(0, 3)
    
    def greedy_action(self, state):
        _, action = self.max_q(state)
        return action
    
    # returns max q value and action given state s 
    def max_q(self, state):
        q_values = self.q_table[state[0]][state[1]]  # Access using both coordinates

        max_index = 0 
        max_q = q_values[max_index]
        for i in range(1, len(q_values)):   
            if q_values[i] > max_q:
                max_index = i
                max_q = q_values[max_index]

        return max_q, max_index

    def update(self, state, action, reward, next_state):
        current_q_value = self.q_table[state[0]][state[1]][action]
        max_q_next_state, _ = self.max_q(next_state)
        
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor*max_q_next_state - current_q_value)
        self.q_table[state[0]][state[1]][action] = new_q_value

def plot_q_table(q_table, episode):
    """
    Visualize the Q-table for each action at the current episode
    """
    action_names = ['Up', 'Right', 'Down', 'Left']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Q-values at Episode {episode}')
    
    for idx, (action, ax) in enumerate(zip(action_names, axes.flat)):
        im = ax.imshow(q_table[:, :, idx], cmap='viridis')
        ax.set_title(f'Action: {action}')
        plt.colorbar(im, ax=ax)
        ax.grid(True)
        
    plt.tight_layout()
    plt.show()

def train(episodes=5000):  # Increased episodes for better convergence
    env = GridWorld()
    agent = QLearningAgent(env.grid_size, len(env.actions))
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
           action = agent.get_action(state)
           s_t_plus_1, r_t_plus_1, done = env.step(action)
           total_reward += r_t_plus_1
           agent.update(state, action, r_t_plus_1, s_t_plus_1)
           state = s_t_plus_1
           steps+=1
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        rewards_history.append(total_reward)
        
        if episode % 1000 == 0:
            print(f"\nEpisode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
            env.render()  # Visualize training progress
            
            # Print and visualize Q-table
            print("\nQ-table statistics:")
            print(f"Max Q-value: {np.max(agent.q_table):.2f}")
            print(f"Min Q-value: {np.min(agent.q_table):.2f}")
            print(f"Mean Q-value: {np.mean(agent.q_table):.2f}")
            # plot_q_table(agent.q_table, episode)
            
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
            
    return agent

if __name__ == "__main__":
    trained_agent = train()