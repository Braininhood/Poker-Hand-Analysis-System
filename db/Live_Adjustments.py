import sqlite3
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        cursor.executescript("""
            DROP TABLE IF EXISTS Live_AI_Adjustments;
        """)
        conn.commit()
        logging.info("Database reset successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error resetting database: {e}")

reset_database()

def create_live_adjustments_table():
    """Create the Live_AI_Adjustments table if it doesn't exist."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Live_AI_Adjustments (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,  
                hand_id INTEGER,
                player_id INTEGER,      
                current_strategy TEXT,     -- "GTO", "Exploitative", "Hybrid"
                recommended_action TEXT,   -- AI's real-time best move
                adjustment_reason TEXT,    -- Why AI changed strategy (e.g., "Opponent overfolds")
                opponent_tendency TEXT,    -- "Loose-Passive", "TAG", "Maniac"
                new_bet_size REAL,         -- AI-adjusted bet size
                new_frequency REAL,        -- AI-adjusted action frequency
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,  -- Added timestamp column
                FOREIGN KEY (hand_id) REFERENCES Hands (hand_id)
            );
        """)
        conn.commit()
        logging.info("Live_AI_Adjustments table created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating Live_AI_Adjustments table: {e}")

# Define a neural network for strategy adjustment
class StrategyAdjustmentNN(nn.Module):
    def __init__(self):
        super(StrategyAdjustmentNN, self).__init__()
        self.fc1 = nn.Linear(8, 32)  # Input layer: 8 features (e.g., VPIP, PFR, etc.)
        self.fc2 = nn.Linear(32, 16)  # Hidden layer
        self.fc3 = nn.Linear(16, 2)   # Output layer: 2 outputs (new_bet_size, new_frequency)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network, loss function, and optimizer
model = StrategyAdjustmentNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(X, y):
    """Train the neural network model."""
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            logging.info(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

def predict_adjustment(features):
    """Predict new bet size and frequency using the trained model."""
    features = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
    return outputs.numpy()

def determine_adjustment_reason(aggression_factor, fold_to_cbet, showdown_win):
    """Determine the adjustment reason based on opponent tendencies."""
    if fold_to_cbet > 0.7 and aggression_factor < 1.5:
        return "Opponent overfolds"
    elif aggression_factor > 2.0 and showdown_win < 0.5:
        return "Opponent is aggressive but weak at showdown"
    elif aggression_factor > 2.0:
        return "Opponent is aggressive"
    else:
        return "Opponent is balanced"

def calculate_new_bet_size(aggression_factor, fold_to_cbet):
    """Calculate new bet size based on opponent tendencies."""
    if aggression_factor > 2.0:
        return 2.0  # Increase bet size against aggressive opponents
    elif fold_to_cbet > 0.7:
        return 1.0  # Decrease bet size against tight opponents
    else:
        return 1.5  # Default bet size

def calculate_new_frequency(aggression_factor, fold_to_cbet):
    """Calculate new frequency based on opponent tendencies."""
    if aggression_factor > 2.0:
        return 0.8  # Increase frequency against aggressive opponents
    elif fold_to_cbet > 0.7:
        return 0.6  # Decrease frequency against tight opponents
    else:
        return 0.7  # Default frequency

def populate_live_adjustments():
    """Populate the Live_AI_Adjustments table with data from Games, Players, Exploitability, and GTO_Solutions."""
    try:
        # Fetch data from relevant tables
        cursor.execute("""
            SELECT g.hand_id, p.player_id, e.vpip, e.pfr, e.aggression_factor, e.cbet_flop, e.cbet_turn, e.cbet_river, e.fold_to_cbet, e.showdown_win, e.playstyle, gto.recommended_action, gto.current_strategy
            FROM Games g
            JOIN Players p ON g.hand_id = p.hand_id
            JOIN Exploitability e ON p.player_id = e.player_id
            JOIN GTO_Solutions gto ON g.hand_id = gto.hand_id
        """)
        data = cursor.fetchall()

        # Prepare training data
        X = []
        y = []
        for row in data:
            hand_id, player_id, vpip, pfr, aggression_factor, cbet_flop, cbet_turn, cbet_river, fold_to_cbet, showdown_win, playstyle, recommended_action, current_strategy = row
            features = [vpip, pfr, aggression_factor, cbet_flop, cbet_turn, cbet_river, fold_to_cbet, showdown_win]
            X.append(features)
            y.append([calculate_new_bet_size(aggression_factor, fold_to_cbet), calculate_new_frequency(aggression_factor, fold_to_cbet)])

        # Train the model
        train_model(X, y)

        # Populate the Live_AI_Adjustments table
        for row in data:
            hand_id, player_id, vpip, pfr, aggression_factor, cbet_flop, cbet_turn, cbet_river, fold_to_cbet, showdown_win, playstyle, recommended_action, current_strategy = row
            features = [vpip, pfr, aggression_factor, cbet_flop, cbet_turn, cbet_river, fold_to_cbet, showdown_win]
            outputs = predict_adjustment(features)

            # Ensure outputs are floats
            new_bet_size = round(float(outputs[0]), 4)  # Convert to float
            new_frequency = round(float(outputs[1]), 4)  # Convert to float

            adjustment_reason = determine_adjustment_reason(aggression_factor, fold_to_cbet, showdown_win)

            # Insert into Live_AI_Adjustments table
            cursor.execute("""
                INSERT INTO Live_AI_Adjustments (
                    hand_id, player_id, current_strategy, recommended_action, 
                    adjustment_reason, opponent_tendency, new_bet_size, new_frequency
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hand_id, player_id, current_strategy, recommended_action, 
                adjustment_reason, playstyle, new_bet_size, new_frequency
            ))

        conn.commit()
        logging.info("Live_AI_Adjustments table populated successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error populating Live_AI_Adjustments table: {e}")

# Create and populate the Live_AI_Adjustments table
create_live_adjustments_table()
populate_live_adjustments()

# Close the connection
conn.close()