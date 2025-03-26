import sqlite3
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

# Database path
DB_PATH = "d:/hand/db/poker.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to the SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def reset_database():
    """Drop all tables in the database."""
    try:
        cursor.executescript("DROP TABLE IF EXISTS Bot_Performance;")
        cursor.executescript("DROP TABLE IF EXISTS Training_Metrics;")
        conn.commit()
        logging.info("Database reset successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error resetting database: {e}")

def create_tables():
    """Create all necessary tables if they don't exist."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Bot_Performance (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,  
                hand_id INTEGER,
                player_id INTEGER,      
                decision_quality TEXT,      
                expected_value REAL,        
                actual_result REAL,         
                deviation_from_GTO REAL,    
                exploitability_gain REAL,   
                adjustment_made TEXT,       
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (hand_id) REFERENCES Hands (hand_id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Training_Metrics (
                episode INTEGER PRIMARY KEY,
                avg_reward REAL,
                avg_regret REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logging.info("All tables created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating tables: {e}")

class DecisionQualityNN(nn.Module):
    def __init__(self):
        super(DecisionQualityNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input: expected_value, actual_result, exploitability_score
        self.fc2 = nn.Linear(64, 32)  # Hidden layer
        self.fc3 = nn.Linear(32, 3)   # Output: 3 classes (Optimal, Suboptimal, Blunder)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DecisionQualityNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_decision_quality_model(X, y):
    """Train the neural network model for decision quality prediction."""
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            logging.info(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

def predict_decision_quality(features):
    """Predict decision quality using the trained model."""
    features = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
    return predicted.numpy()

def populate_bot_performance():
    """Populate the Bot_Performance table with only the best solutions after training."""
    try:
        cursor.execute("""
            SELECT 
                h.hand_id,
                p.player_id,
                r.net_result AS actual_result,
                gto.expected_value,
                e.exploitability_score
            FROM Hands h
            JOIN Players p ON h.hand_id = p.hand_id
            JOIN Results r ON h.hand_id = r.hand_id AND p.player_id = r.player_id
            JOIN GTO_Solutions gto ON h.hand_id = gto.hand_id
            JOIN Exploitability e ON p.player_id = e.player_id
        """)
        data = cursor.fetchall()

        X = []
        y = []
        for row in data:
            hand_id, player_id, actual_result, expected_value, exploitability_score = row
            X.append([expected_value, actual_result, exploitability_score])
            if expected_value >= actual_result:
                y.append(0)  # Optimal
            elif expected_value * 0.8 <= actual_result < expected_value:
                y.append(1)  # Suboptimal
            else:
                y.append(2)  # Blunder

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        train_decision_quality_model(X, y)

        for row in data:
            hand_id, player_id, actual_result, expected_value, exploitability_score = row
            features = np.array([[expected_value, actual_result, exploitability_score]], dtype=np.float32)
            features = scaler.transform(features)

            decision_quality = predict_decision_quality(features)[0]
            decision_quality_map = {0: "Optimal", 1: "Suboptimal", 2: "Blunder"}
            decision_quality = decision_quality_map[decision_quality]

            # Only log "Optimal" decisions
            if decision_quality == "Optimal":
                deviation_from_GTO = abs(expected_value - actual_result)
                exploitability_gain = exploitability_score * 0.5
                adjustment_made = "Yes" if deviation_from_GTO > 0.1 else "No"

                cursor.execute("""
                    INSERT INTO Bot_Performance (hand_id, player_id, decision_quality, expected_value, actual_result, deviation_from_GTO, exploitability_gain, adjustment_made)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (hand_id, player_id, decision_quality, expected_value, actual_result, deviation_from_GTO, exploitability_gain, adjustment_made))

        conn.commit()
        logging.info("Bot_Performance table populated with optimal solutions.")
    except sqlite3.Error as e:
        logging.error(f"Error populating Bot_Performance table: {e}")

# Define the Monte Carlo CFR class
class MonteCarloCFR:
    def __init__(self, input_size, output_size):
        self.regret_network = self.build_network(input_size, output_size)
        self.strategy_network = self.build_network(input_size, output_size)
        self.cumulative_regrets = np.zeros(output_size)
        self.cumulative_strategy = np.zeros(output_size)

    def build_network(self, input_size, output_size):
        """Build a neural network using PyTorch."""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def train(self, poker_env, episodes=10000):
        """Train the Monte Carlo CFR model."""
        regret_optimizer = optim.Adam(self.regret_network.parameters(), lr=0.001)
        strategy_optimizer = optim.Adam(self.strategy_network.parameters(), lr=0.001)

        for episode in range(episodes):
            state = poker_env.reset()
            regrets = []
            actions = []
            rewards = []

            # Monte Carlo CFR: Sample a single path through the game tree
            while not poker_env.is_done():
                # Predict strategy and select action
                state_tensor = torch.tensor(state, dtype=torch.float32)
                strategy = torch.softmax(self.strategy_network(state_tensor), dim=-1).detach().numpy()
                action = np.random.choice(len(strategy), p=strategy)
                next_state, reward, done = poker_env.step(action)

                # Record state-action pair and reward
                regrets.append((state, action, reward))
                actions.append(action)
                rewards.append(reward)
                state = next_state

            # Update regrets and strategy based on the sampled path
            for state, action, reward in regrets:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                strategy = torch.softmax(self.strategy_network(state_tensor), dim=-1).detach().numpy()
                regret_values = self.compute_regrets(action, reward, strategy)  # Remove state argument

                # Update regret network
                regret_optimizer.zero_grad()
                regret_pred = self.regret_network(state_tensor)
                regret_loss = torch.mean((regret_pred - torch.tensor(regret_values, dtype=torch.float32)) ** 2)
                regret_loss.backward()
                regret_optimizer.step()

                # Update strategy network
                strategy_optimizer.zero_grad()
                strategy_pred = self.strategy_network(state_tensor)
                strategy_loss = torch.mean((strategy_pred - torch.tensor(strategy, dtype=torch.float32)) ** 2)
                strategy_loss.backward()
                strategy_optimizer.step()

                # Update cumulative regrets and strategy
                self.cumulative_regrets += regret_values
                self.cumulative_strategy += strategy

            # Log training metrics to the database
            self.log_training_metrics(episode, rewards, regrets)

    def log_training_metrics(self, episode, rewards, regrets):
        """Log training metrics to the database, rounding values to 4 decimal places."""
        try:
            # Calculate average reward and regret for the episode
            avg_reward = round(np.mean(rewards), 4)  # Round to 4 decimal places
            avg_regret = round(np.mean([regret[2] for regret in regrets]), 4)  

            # Insert metrics into the database
            cursor.execute("""
                INSERT INTO Training_Metrics (episode, avg_reward, avg_regret, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (episode, avg_reward, avg_regret))

            conn.commit()
            logging.info(f"Logged training metrics for episode {episode}.")
        except sqlite3.Error as e:
            logging.error(f"Error logging training metrics: {e}")


    def compute_regrets(self, action, reward, strategy):
        """Compute regrets for the given action, reward, and strategy."""
        # Debugging: Print shapes and types
        print(f"Shape of reward: {np.shape(reward)}")
        print(f"Type of reward: {type(reward)}")
        print(f"Shape of strategy: {np.shape(strategy)}")
        print(f"Type of strategy: {type(strategy)}")
        print(f"Action: {action}")

        # Ensure reward is a scalar
        if isinstance(reward, (np.ndarray, list)):
            reward = np.sum(reward)  # Sum all elements to get a scalar

        # Ensure strategy is a 1D array
        if isinstance(strategy, (np.float32, np.float64, float, int)):
            raise ValueError("Strategy must be a 1D array, not a scalar.")

        # Compute regret values
        regret_values = np.zeros(len(strategy))
        regret_values[action] = reward - np.sum(strategy * reward)

  # Regret for the chosen action

        return regret_values


    # Define a simple poker environment for testing
class PokerEnv:
    def __init__(self):
        self.state_size = 50  # Size of the state vector
        self.action_space = ["Fold", "Call", "Raise", "All-in"]  # Possible actions
        self.action_map = {0: "Fold", 1: "Call", 2: "Raise", 3: "All-in"}  # Map integer actions to strings
        self.done = False
        self.pot_size = 0  # Initialize the pot size
        self.current_bet = 0  # Track the current bet amount
        self.player_stack = 1000  # Example player stack size
        self.opponent_stack = 1000  # Example opponent stack size

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = np.random.rand(self.state_size)  # Random state vector
        self.done = False
        self.pot_size = 0  # Reset the pot size
        self.current_bet = 0  # Reset the current bet
        self.player_stack = 1000  # Reset player stack
        self.opponent_stack = 1000  # Reset opponent stack
        return self.state

    def step(self, action):
        """
        Simulate a step in the poker game with betting.
        Args:
            action: The action taken by the agent (e.g., 0 for Fold, 1 for Call, 2 for Raise, 3 for All-in).
        Returns:
            next_state: The next state of the game.
            reward: The reward for the action taken (must be a scalar).
            done: Whether the game is over.
        """
        if self.done:
            raise ValueError("The game is already over. Call reset() to start a new game.")

        # Map the integer action to the corresponding string action
        if action not in self.action_map:
            raise ValueError(f"Invalid action: {action}")
        action_str = self.action_map[action]

        reward = 0  # Initialize reward

        if action_str == "Fold":
            # Player folds, opponent wins the pot
            reward = -self.pot_size
            self.done = True
        elif action_str == "Call":
            # Player matches the current bet
            bet_amount = self.current_bet
            if bet_amount > self.player_stack:
                bet_amount = self.player_stack  # Can't bet more than the stack
            self.player_stack -= bet_amount
            self.pot_size += bet_amount
            reward = self.simulate_round_outcome()  # Simulate the outcome of the round
        elif action_str == "Raise":
            # Player raises the bet
            raise_amount = np.random.randint(10, 100)  # Random raise amount for simulation
            if raise_amount > self.player_stack:
                raise_amount = self.player_stack  # Can't raise more than the stack
            self.player_stack -= raise_amount
            self.pot_size += raise_amount
            self.current_bet += raise_amount
            reward = self.simulate_round_outcome()  # Simulate the outcome of the round
        elif action_str == "All-in":
            # Player goes all-in
            all_in_amount = self.player_stack
            self.player_stack = 0
            self.pot_size += all_in_amount
            self.current_bet += all_in_amount
            reward = self.simulate_round_outcome()  # Simulate the outcome of the round
        else:
            raise ValueError(f"Invalid action: {action_str}")

        # Update the state to a new random vector
        next_state = np.random.rand(self.state_size)

        # Randomly decide if the game is over
        self.done = np.random.choice([True, False], p=[0.1, 0.9])  # 10% chance to end the game

        # Ensure reward is a scalar
        if isinstance(reward, (np.ndarray, list)):
            reward = reward[0]  # Take the first element if reward is an array or list

        return next_state, reward, self.done

    def simulate_round_outcome(self):
        """
        Simulate the outcome of the round (e.g., win, lose, or draw).
        Returns:
            reward: The reward for the round.
        """
        # Randomly decide if the player wins, loses, or draws
        outcome = np.random.choice(["win", "lose", "draw"], p=[0.4, 0.4, 0.2])
        if outcome == "win":
            reward = self.pot_size  # Player wins the pot
            self.player_stack += self.pot_size
        elif outcome == "lose":
            reward = -self.pot_size  # Player loses the pot
            self.opponent_stack += self.pot_size
        else:
            reward = 0  # Draw, no change in stacks

        self.pot_size = 0  # Reset the pot after the round
        return reward

    def is_done(self):
        """Check if the game is over."""
        return self.done

    def is_all_in(self):
        """Check if the player is all-in."""
        return self.player_stack == 0

# Main execution
try:
    reset_database()
    create_tables()
    populate_bot_performance()

    # Define or import the poker environment
    poker_env = PokerEnv()  # Use the enhanced poker environment with betting

    # Train the Monte Carlo CFR model
    input_size = 50  # Size of the state vector
    output_size = 4  # Number of actions: Fold, Call, Raise, All-in
    monte_carlo_cfr = MonteCarloCFR(input_size, output_size)
    monte_carlo_cfr.train(poker_env)
finally:
    conn.close()