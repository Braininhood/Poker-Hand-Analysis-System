import sqlite3
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

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
            DROP TABLE IF EXISTS Opponent_Profiles;
        """)
        conn.commit()
        logging.info("Database reset successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error resetting database: {e}")

reset_database()

def create_opponent_profiles_table():
    """Create the Opponent_Profiles table if it doesn't exist."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Opponent_Profiles (
                player_id INTEGER PRIMARY KEY,
                total_hands INTEGER,       
                avg_vpip REAL,           
                avg_pfr REAL,           
                avg_agg_factor REAL,    
                avg_cbet REAL,         
                avg_fold_to_cbet REAL,  
                avg_showdown_win REAL,  
                playstyle TEXT,         
                exploitability_index REAL,  -- Ensure this is REAL
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        logging.info("Opponent_Profiles table created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating Opponent_Profiles table: {e}")

# Define an improved neural network for exploitability prediction
class ImprovedExploitabilityNN(nn.Module):
    def __init__(self):
        super(ImprovedExploitabilityNN, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # Input layer: 6 features
        self.dropout1 = nn.Dropout(0.2)  # Dropout for regularization
        self.fc2 = nn.Linear(64, 32)  # Hidden layer
        self.dropout2 = nn.Dropout(0.2)  # Dropout for regularization
        self.fc3 = nn.Linear(32, 16)  # Hidden layer
        self.fc4 = nn.Linear(16, 1)   # Output layer: exploitability index

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the neural network, loss function, and optimizer
model = ImprovedExploitabilityNN()
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

def predict_exploitability(features):
    """Predict exploitability index using the trained model."""
    features = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
    return outputs.numpy()

def populate_opponent_profiles():
    """Populate the Opponent_Profiles table with data from Exploitability, Results, and Players."""
    try:
        # Fetch data from relevant tables
        cursor.execute("""
            SELECT 
                p.player_id,
                COUNT(DISTINCT p.hand_id) AS total_hands,
                AVG(e.vpip) AS avg_vpip,
                AVG(e.pfr) AS avg_pfr,
                AVG(e.aggression_factor) AS avg_agg_factor,
                AVG(e.cbet_flop) AS avg_cbet,
                AVG(e.fold_to_cbet) AS avg_fold_to_cbet,
                AVG(r.showdown) AS avg_showdown_win
            FROM Players p
            LEFT JOIN Exploitability e ON p.player_id = e.player_id
            LEFT JOIN Results r ON p.player_id = r.player_id
            GROUP BY p.player_id
        """)
        data = cursor.fetchall()

        # Prepare data for training
        X = []
        y = []
        for row in data:
            player_id, total_hands, avg_vpip, avg_pfr, avg_agg_factor, avg_cbet, avg_fold_to_cbet, avg_showdown_win = row
            X.append([avg_vpip, avg_pfr, avg_agg_factor, avg_cbet, avg_fold_to_cbet, avg_showdown_win])
            y.append([avg_vpip + avg_pfr + avg_agg_factor])  # Simple target for training

        # Convert to numpy arrays for scaling
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Scale features and target
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)

        # Train the model
        train_model(X, y)

        for row in data:
            player_id, total_hands, avg_vpip, avg_pfr, avg_agg_factor, avg_cbet, avg_fold_to_cbet, avg_showdown_win = row

            # Determine playstyle based on metrics
            if avg_vpip > 0.4 and avg_pfr > 0.3:
                playstyle = "Aggressive LAG"
            elif avg_vpip > 0.3 and avg_pfr > 0.2:
                playstyle = "LAG"
            elif avg_vpip < 0.2 and avg_pfr < 0.15:
                playstyle = "Nit"
            elif avg_vpip < 0.25 and avg_pfr < 0.2:
                playstyle = "Tight Passive"
            elif avg_vpip > 0.25 and avg_pfr < 0.15:
                playstyle = "Loose Passive"
            else:
                playstyle = "TAG"

            # Predict exploitability index using the neural network
            features = np.array([[avg_vpip, avg_pfr, avg_agg_factor, avg_cbet, avg_fold_to_cbet, avg_showdown_win]], dtype=np.float32)
            features = scaler_X.transform(features)
            exploitability_index = predict_exploitability(features)
            exploitability_index = scaler_y.inverse_transform(exploitability_index)[0][0]

            # Round all numeric values to 4 decimal places
            avg_vpip = round(avg_vpip, 4)
            avg_pfr = round(avg_pfr, 4)
            avg_agg_factor = round(avg_agg_factor, 4)
            avg_cbet = round(avg_cbet, 4)
            avg_fold_to_cbet = round(avg_fold_to_cbet, 4)
            avg_showdown_win = round(avg_showdown_win, 4)
            exploitability_index = round(exploitability_index, 4)

            # Debugging: Log the exploitability_index before insertion
            logging.info(f"Debug: Inserting exploitability_index for player {player_id}: {exploitability_index}")

            # Insert into Opponent_Profiles table
            cursor.execute("""
                INSERT INTO Opponent_Profiles (player_id, total_hands, avg_vpip, avg_pfr, avg_agg_factor, avg_cbet, avg_fold_to_cbet, avg_showdown_win, playstyle, exploitability_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (player_id, total_hands, avg_vpip, avg_pfr, avg_agg_factor, avg_cbet, avg_fold_to_cbet, avg_showdown_win, playstyle, round(float(exploitability_index), 4)))  # Ensure exploitability_index is a float

        conn.commit()
        logging.info("Opponent_Profiles table populated successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error populating Opponent_Profiles table: {e}")

def make_real_time_adjustments():
    """Make real-time adjustments for all players in the Opponent_Profiles table."""
    try:
        cursor.execute("SELECT player_id FROM Opponent_Profiles")
        player_ids = cursor.fetchall()

        for (player_id,) in player_ids:
            cursor.execute("""
                SELECT playstyle, exploitability_index FROM Opponent_Profiles WHERE player_id = ?
            """, (player_id,))
            profile = cursor.fetchone()

            if profile:
                playstyle, exploitability_index = profile
                # Ensure exploitability_index is a float
                if isinstance(exploitability_index, bytes):
                    try:
                        # If the value is stored as bytes, decode it to a float
                        exploitability_index = float(exploitability_index.decode('utf-8'))
                    except (UnicodeDecodeError, ValueError) as e:
                        logging.error(f"Error decoding exploitability_index for player {player_id}. Value: {exploitability_index}. Error: {e}")
                        exploitability_index = 0.5  # Default value for exploitability_index
                elif isinstance(exploitability_index, (int, float)):
                    exploitability_index = float(exploitability_index)
                else:
                    logging.error(f"Unexpected type for exploitability_index for player {player_id}: {type(exploitability_index)}")
                    exploitability_index = 0.5  # Default value for exploitability_index
                
                # Debugging: Check the data type of exploitability_index
                logging.info(f"Debug: Player {player_id} exploitability_index type: {type(exploitability_index)}")

                # Log the player's playstyle and exploitability index
                logging.info(f"Player {player_id} has playstyle: {playstyle} and exploitability index: {exploitability_index:.4f}")

                # Adjust bet size based on exploitability index
                if exploitability_index > 0.7:
                    logging.info("Opponent is highly exploitable. Consider increasing bet size significantly to maximize profit.")
                elif exploitability_index > 0.5:
                    logging.info("Opponent is moderately exploitable. A slight increase in bet size may be beneficial.")
                elif exploitability_index < 0.3:
                    logging.info("Opponent is not very exploitable. Decreasing bet size could help minimize losses.")
                else:
                    logging.info("Opponent's exploitability is average. Maintain current bet size for balanced play.")
            else:
                logging.warning(f"No profile found for player_id {player_id}. Unable to make adjustments.")
    except sqlite3.Error as e:
        logging.error(f"Error making real-time adjustments: {e}")

# Create and populate the Opponent_Profiles table
create_opponent_profiles_table()
populate_opponent_profiles()

# Make real-time adjustments for all players
make_real_time_adjustments()

# Close the connection
conn.close()