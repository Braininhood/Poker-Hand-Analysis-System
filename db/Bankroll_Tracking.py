import sqlite3
import logging
from datetime import datetime

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
            DROP TABLE IF EXISTS Bankroll_Tracking;
        """)
        conn.commit()
        logging.info("Database reset successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error resetting database: {e}")

reset_database()

def create_bankroll_tracking_table():
    """Create the Bankroll_Tracking table if it doesn't exist."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Bankroll_Tracking (
                player_id INTEGER,      
                starting_balance REAL,  
                ending_balance REAL,    
                net_profit_loss REAL,  
                total_hands INTEGER,    
                bb_per_100_hands REAL,  
                total_rake_paid REAL,   
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id),
                FOREIGN KEY (player_id) REFERENCES Players (player_id)
            );
        """)
        conn.commit()
        logging.info("Bankroll_Tracking table created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error creating Bankroll_Tracking table: {e}")

def populate_bankroll_tracking():
    """Populate the Bankroll_Tracking table with data from Results, Players, and Games, focusing only on the hero."""
    try:
        # Fetch data from relevant tables, filtering for the hero (is_hero = 1)
        cursor.execute("""
            SELECT 
                p.player_id,
                MIN(p.stack_size) AS starting_balance,
                MAX(p.stack_size) AS ending_balance,
                SUM(r.net_result) AS net_profit_loss,
                COUNT(DISTINCT g.hand_id) AS total_hands,
                SUM(g.big_blind * 0.1) AS total_rake_paid,
                (SUM(r.net_result) / (COUNT(DISTINCT g.hand_id) / 100.0)) AS bb_per_100_hands
            FROM Players p
            JOIN Results r ON p.player_id = r.player_id
            JOIN Games g ON p.hand_id = g.hand_id
            WHERE p.is_hero = 1  -- Filter for the hero
            GROUP BY p.player_id  -- Group by player_id
        """)
        data = cursor.fetchall()

        for row in data:
            player_id, starting_balance, ending_balance, net_profit_loss, total_hands, total_rake_paid, bb_per_100_hands = row

            # Insert into Bankroll_Tracking table
            cursor.execute("""
                INSERT INTO Bankroll_Tracking (player_id, starting_balance, ending_balance, net_profit_loss, total_hands, bb_per_100_hands, total_rake_paid)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (player_id, starting_balance, ending_balance, net_profit_loss, total_hands, bb_per_100_hands, total_rake_paid))

        conn.commit()
        logging.info("Bankroll_Tracking table populated successfully.")
    except sqlite3.Error as e:
        logging.error(f"Error populating Bankroll_Tracking table: {e}")

# Create and populate the Bankroll_Tracking table
create_bankroll_tracking_table()
populate_bankroll_tracking()

# Close the connection
conn.close()