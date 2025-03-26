import os
import sqlite3
import re
import logging
from treys import Evaluator, Card, Deck
import random
import time

# Database setup
DB_PATH = "d:/hand/db/poker.db"
FOLDER_PATH = "d:/hand/5/"

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Treys Evaluator
evaluator = Evaluator()

# Monte Carlo simulation parameters
NUM_SIMULATIONS = 10000  # Number of simulations for equity calculation

def reset_database():
    """ Drop all tables in the database """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            DROP TABLE IF EXISTS Results;
            DROP TABLE IF EXISTS Actions;
            DROP TABLE IF EXISTS Hands;
            DROP TABLE IF EXISTS Players;
            DROP TABLE IF EXISTS Games;
        """)
        conn.commit()

def create_tables():
    """ Create necessary tables in the database """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS Games (
                hand_id INTEGER PRIMARY KEY,
                stakes TEXT,
                game_type TEXT,
                num_players INTEGER,
                small_blind REAL,
                big_blind REAL
            );

            CREATE TABLE IF NOT EXISTS Players (
                hand_id INTEGER,
                player_id INTEGER,
                position TEXT,
                stack_size REAL,
                bb_size REAL,
                is_hero INTEGER,
                PRIMARY KEY (hand_id, player_id),
                FOREIGN KEY (hand_id) REFERENCES Games (hand_id)
            );

            CREATE TABLE IF NOT EXISTS Hands (
                hand_id INTEGER PRIMARY KEY,
                hero_position TEXT,
                hero_cards TEXT,
                board_cards TEXT,
                pot_preflop REAL,
                pot_flop REAL,
                pot_turn REAL,
                pot_river REAL,
                FOREIGN KEY (hand_id) REFERENCES Games (hand_id)
            );

            CREATE TABLE IF NOT EXISTS Actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER,
                player_id INTEGER,
                position TEXT,
                street TEXT,
                action_type TEXT,
                amount REAL,
                is_all_in INTEGER,
                effective_stack REAL,
                pot_before REAL,
                pot_after REAL,
                bet_to_pot_ratio REAL,
                FOREIGN KEY (hand_id) REFERENCES Hands (hand_id)
            );

            CREATE TABLE IF NOT EXISTS Results (
                hand_id INTEGER,
                player_id INTEGER,
                position TEXT,
                cards TEXT,
                net_result REAL,
                won_pot REAL,
                showdown INTEGER,
                final_equity REAL,
                preflop_equity REAL,
                flop_equity REAL,
                turn_equity REAL,
                river_equity REAL,
                PRIMARY KEY (hand_id, player_id),
                FOREIGN KEY (hand_id) REFERENCES Hands (hand_id)
            );
        """)
        conn.commit()

def convert_amount(amount_str):
    """ Convert a string amount (e.g., '$1.24') to a float """
    if amount_str:
        return float(amount_str.replace('$', '').replace(',', ''))
    return 0.0

def parse_hand_file(file_path):
    """ Parse a poker hand history file and extract details """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    hand_id = int(re.search(r"hand_(\d+)\.txt", file_path).group(1))
    stakes_match = re.search(r"\$(\d+(?:\.\d+)?)/\$(\d+(?:\.\d+)?)", lines[0])

    if not stakes_match:
        logging.warning(f"Invalid stakes format in file {file_path}. Skipping.")
        return None

    small_blind, big_blind = map(convert_amount, stakes_match.groups())
    game_type = "NLH" if "NLH" in lines[0] else "Other"
    players_match = re.search(r"(\d+)\s+Players", lines[0])
    num_players = int(players_match.group(1) if players_match else 0)

    # Map player names to unique IDs
    player_id_map = {}
    player_id_counter = 1
    players = []
    hero_position = None

    for line in lines:
        player_match = re.match(r"^Hero\s*\((\w+)\):\s*\$(\d+(?:\.\d+)?)\s*\((\d+[,.]?\d*) bb\)|^(\w+):\s*\$(\d+(?:\.\d+)?)\s*\((\d+[,.]?\d*) bb\)", line)
        if player_match:
            if player_match.group(1):
                is_hero = 1
                position = player_match.group(1)
                stack_size = convert_amount(player_match.group(2))
                bb_size = convert_amount(player_match.group(3))
                hero_position = position
            else:
                is_hero = 0
                position = player_match.group(4)
                stack_size = convert_amount(player_match.group(5))
                bb_size = convert_amount(player_match.group(6))

            if position not in player_id_map:
                player_id_map[position] = player_id_counter
                player_id_counter += 1

            player_id = player_id_map[position]
            players.append((hand_id, player_id, position, stack_size, bb_size, is_hero))

    # Parse hero cards
    hero_cards = None
    for line in lines:
        if "Preflop" in line and "Hero" in line:
            hero_cards_match = re.search(r"Hero\s+([2-9TJQKA][cdhs])\s+([2-9TJQKA][cdhs])", line)
            if hero_cards_match:
                hero_cards = f"{hero_cards_match.group(1)} {hero_cards_match.group(2)}"
            else:
                logging.warning(f"No hero cards found in hand_id {hand_id}.")

    # Parse board cards and pot sizes
    board_cards = {"Flop": None, "Turn": None, "River": None}
    pot_sizes = {"Preflop": 0, "Flop": 0, "Turn": 0, "River": 0}
    current_street = "Preflop"

    for line in lines:
        # Update current_street when a new street begins
        if "Preflop:" in line:
            current_street = "Preflop"
        elif "Flop:" in line:
            current_street = "Flop"
        elif "Turn:" in line:
            current_street = "Turn"
        elif "River:" in line:
            current_street = "River"

        # Parse flop cards and pot size
        flop_match = re.search(r"Flop:\s+\(\$?(\d+(?:\.\d+)?)\)\s+([2-9TJQKA][cdhs]\s+[2-9TJQKA][cdhs]\s+[2-9TJQKA][cdhs])", line)
        if flop_match and not board_cards["Flop"]:  # Only parse flop cards once
            pot_sizes["Flop"] = convert_amount(flop_match.group(1))
            board_cards["Flop"] = flop_match.group(2)

        # Parse turn card and pot size
        turn_match = re.search(r"Turn:\s+\(\$?(\d+(?:\.\d+)?)\)\s+([2-9TJQKA][cdhs])", line)
        if turn_match and not board_cards["Turn"]:  # Only parse turn card once
            pot_sizes["Turn"] = convert_amount(turn_match.group(1))
            board_cards["Turn"] = turn_match.group(2)

        # Parse river card and pot size
        river_match = re.search(r"River:\s+\(\$?(\d+(?:\.\d+)?)\)\s+([2-9TJQKA][cdhs])", line)
        if river_match and not board_cards["River"]:  # Only parse river card once
            pot_sizes["River"] = convert_amount(river_match.group(1))
            board_cards["River"] = river_match.group(2)

    # Parse actions
    actions = []
    pot_size = small_blind + big_blind  # Initialize pot size with blinds
    acted_players = set()  # Track players who have acted
    last_bet_amount = 0  # Track the last bet amount for raises

    for line in lines:
        if "Preflop:" in line:
            current_street = "Preflop"
            last_bet_amount = 0  # Reset last_bet_amount for Preflop
        elif "Flop:" in line:
            current_street = "Flop"
            last_bet_amount = 0  # Reset last_bet_amount for Flop
        elif "Turn:" in line:
            current_street = "Turn"
            last_bet_amount = 0  # Reset last_bet_amount for Turn
        elif "River:" in line:
            current_street = "River"
            last_bet_amount = 0  # Reset last_bet_amount for River

        # Handle folds
        fold_match = re.match(r"(\d+)\s+folds", line)
        if fold_match:
            num_folds = int(fold_match.group(1))
            for player in players:
                if player[2] not in acted_players:  # Check if player has not acted
                    # Skip if the player is the Hero and has taken actions
                    if player[5] == 1:  # Hero (is_hero = 1)
                        continue  # Hero cannot fold if they have taken actions
                    # Add fold action for this player
                    actions.append((hand_id, player[1], player[2], current_street, "Fold", 0.0, 0, 
                                float(player[3]), pot_size, pot_size, None))
                    acted_players.add(player[2])
                    num_folds -= 1
                    logging.warning(f"Player {player[2]} folded in hand_id {hand_id}.")  # Log the position
                    if num_folds == 0:
                        break
        elif line.strip() == "All fold":
            for player in players:
                if player[2] not in acted_players:
                    # Skip if the player is the Hero and has taken actions
                    if player[5] == 1:  # Hero (is_hero = 1)
                        continue  # Hero cannot fold if they have taken actions
                    # Add fold action for this player
                    actions.append((hand_id, player[1], player[2], current_street, "Fold", 0.0, 0, 
                                float(player[3]), pot_size, pot_size, None))
                    acted_players.add(player[2])
                    logging.warning(f"Player {player[2]} folded in hand_id {hand_id}.")  # Log the position

        # Handle other actions (raises, calls, bets, checks)
        action_match = re.match(r"([\w\s]+)\s+(raises to|calls|bets|folds|checks)\s+\$?(\d+(?:\.\d+)?)?", line)
        if action_match:
            position, action, amount = action_match.groups()
            amount = convert_amount(amount) if amount else 0.0
            action_map = {
                "bets": "Bet",
                "calls": "Call",
                "raises to": "Raise",
                "folds": "Fold",
                "checks": "Check"
            }

            # Replace "Hero" with the correct position
            if position == "Hero":
                position = hero_position

            # Replace numeric positions with "Unknown"
            if position.isdigit():
                logging.warning(f"Replaced numeric position {position} with 'Unknown' in hand_id {hand_id}.")
                position = "Unknown"

            # Check if position exists in player_id_map
            if position not in player_id_map:
                logging.warning(f"Unknown player position: {position} in hand_id {hand_id}. Skipping action.")
                continue

            player_id = player_id_map[position]
            is_all_in = 1 if "all-in" in line else 0

            # Calculate pot_before and pot_after
            pot_before = pot_size
            if action_map[action] in ["Bet", "Raise", "Call"]:
                if action_map[action] == "Raise":
                    # For raises, subtract the previous bet amount
                    pot_size += (amount - last_bet_amount)
                elif action_map[action] == "Bet":
                    # For bets, add the full amount
                    pot_size += amount
                elif action_map[action] == "Call":
                    # For calls, add the amount needed to call
                    pot_size += amount
            pot_after = pot_size

            # Calculate bet_to_pot_ratio
            if action_map[action] in ["Bet", "Raise"]:
                if action_map[action] == "Bet":
                    bet_to_pot_ratio = round(amount / pot_before, 4) if pot_before > 0 else None
                    last_bet_amount = amount  # Update last_bet_amount for bets
                elif action_map[action] == "Raise":
                    bet_to_pot_ratio = round((amount - last_bet_amount) / pot_before, 4) if pot_before > 0 else None
                    last_bet_amount = amount  # Update last_bet_amount for raises
            else:
                bet_to_pot_ratio = None

            # Assign action to the correct street
            actions.append((hand_id, player_id, position, current_street, action_map[action], amount, is_all_in, 
                        float(players[player_id - 1][3]), pot_before, pot_after, 
                        bet_to_pot_ratio))
            acted_players.add(position)

        
    # Parse results
    results = []
    result_pattern = re.compile(
        r"(\w+)\s+"
        r"(showed|mucked)\s*"
        r"([2-9TJQKA][cdhs]\s+[2-9TJQKA][cdhs])?\s*"
        r"and\s+(won|lost)\s+"
        r"(?:\$?(\d+(?:\.\d+)?))?\s*"
        r"(?:\((-?\$?\d+(?:\.\d+)?)\s+net\))?"
    )
    for line in lines:
        result_match = result_pattern.search(line)
        if result_match:
            position = result_match.group(1)
            action = result_match.group(2)
            cards = result_match.group(3) if result_match.group(3) else ""
            outcome = result_match.group(4)
            won_amount = convert_amount(result_match.group(5)) if result_match.group(5) else 0.0
            net_result = convert_amount(result_match.group(6)) if result_match.group(6) else 0.0

            showdown = 1 if outcome == "won" else 0

            # Replace "Hero" with the correct position
            if position == "Hero":
                position = hero_position

            # Replace numeric positions with "Unknown"
            if position.isdigit():
                position = "Unknown"
                logging.warning(f"Replaced numeric position with 'Unknown' in hand_id {hand_id}.")

            # Check if position exists in player_id_map
            if position not in player_id_map:
                logging.warning(f"Unknown player position: {position} in hand_id {hand_id}. Skipping result.")
                continue

            player_id = player_id_map[position]
            results.append((hand_id, player_id, position, cards, net_result, won_amount, showdown))

    parsed_data = {
        "hand_id": hand_id,
        "stakes": f"{small_blind}/{big_blind}",
        "game_type": game_type,
        "num_players": num_players,
        "players": players,
        "hero_position": hero_position,
        "hero_cards": hero_cards,
        "board_cards": board_cards,
        "pot_sizes": pot_sizes,
        "actions": actions,
        "results": results
    }

    return parsed_data

def calculate_equity_monte_carlo(players, board_cards, evaluator, num_simulations=NUM_SIMULATIONS):
    """
    Calculate equity using Monte Carlo simulations.
    :param players: List of players with their hole cards.
    :param board_cards: List of community cards on the board.
    :param evaluator: Treys Evaluator instance.
    :param num_simulations: Number of simulations to run.
    :return: Dictionary of player_id to equity.
    """
    if not players:
        return {}

    # Convert known cards to Treys format
    known_cards = []
    for player in players:
        if "hero_cards" in player and player["hero_cards"]:  # Check if hero_cards is not empty
            hero_cards = player["hero_cards"].split()
            if len(hero_cards) == 2:
                known_cards.extend([Card.new(card) for card in hero_cards])
    known_cards.extend([Card.new(card) for card in board_cards])

    # Initialize equity counters
    equity_counts = {player[1]: 0 for player in players}

    for _ in range(num_simulations):
        # Create a new deck and remove known cards
        deck = Deck()
        for card in known_cards:
            deck.cards.remove(card)

        # Deal remaining board cards
        remaining_board_cards = []
        if len(board_cards) < 5:
            remaining_board_cards = deck.draw(5 - len(board_cards))

        # Deal unknown opponent cards
        opponent_cards = []
        for player in players:
            if "hero_cards" not in player or not player["hero_cards"]:  # If hole cards are unknown
                opponent_cards.append(deck.draw(2))

        # Evaluate hands
        best_hand = None
        best_player = None
        for player in players:
            if "hero_cards" in player and player["hero_cards"]:  # If hole cards are known
                hero_cards = player["hero_cards"].split()
                if len(hero_cards) == 2:
                    hole_cards = [Card.new(card) for card in hero_cards]
                    hand_strength = evaluator.evaluate(hole_cards, known_cards + remaining_board_cards)
                    if best_hand is None or hand_strength < best_hand:
                        best_hand = hand_strength
                        best_player = player[1]
            else:  # If hole cards are unknown
                hole_cards = opponent_cards.pop(0)
                hand_strength = evaluator.evaluate(hole_cards, known_cards + remaining_board_cards)
                if best_hand is None or hand_strength < best_hand:
                    best_hand = hand_strength
                    best_player = player[1]

        # Update equity counts
        if best_player is not None:
            equity_counts[best_player] += 1

    # Normalize equity counts to probabilities
    equities = {player_id: count / num_simulations for player_id, count in equity_counts.items()}
    return equities

def main():
    reset_database()
    create_tables()

    # Create a single database connection
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        process_all_files_in_batches(conn)

def process_all_files_in_batches(conn, batch_size=100):
    """ Process all poker hand files in the directory in batches """
    files = [f for f in os.listdir(FOLDER_PATH) if f.startswith("hand_") and f.endswith(".txt")]

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        for file_name in batch_files:
            file_path = os.path.join(FOLDER_PATH, file_name)
            parsed_data = parse_hand_file(file_path)
            if parsed_data:
                insert_into_db(conn, parsed_data)
        conn.commit()  # Commit after each batch

def insert_into_db(conn, data):
    """ Insert or update parsed data into the database with validation and error handling """
    if data is None:
        return

    try:
        cursor = conn.cursor()

        # Insert or update Games table
        small_blind, big_blind = map(float, data["stakes"].split("/"))
        cursor.execute("""
            INSERT OR REPLACE INTO Games (hand_id, stakes, game_type, num_players, small_blind, big_blind)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (data["hand_id"], data["stakes"], data["game_type"], data["num_players"], small_blind, big_blind))

        # Insert or update Players table
        for player in data["players"]:
            cursor.execute("""
                INSERT OR REPLACE INTO Players (hand_id, player_id, position, stack_size, bb_size, is_hero)
                VALUES (?, ?, ?, ?, ?, ?)
            """, player)

        # Insert or update Hands table
        cursor.execute("""
            INSERT OR REPLACE INTO Hands (hand_id, hero_position, hero_cards, board_cards, pot_preflop, pot_flop, pot_turn, pot_river)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (data["hand_id"], data["hero_position"], data["hero_cards"],
             " ".join(filter(None, data["board_cards"].values())),
             *data["pot_sizes"].values()))

        # Insert or update Actions table
        for action in data["actions"]:
            cursor.execute("""
                INSERT OR REPLACE INTO Actions (hand_id, player_id, position, street, action_type, amount, is_all_in, effective_stack, pot_before, pot_after, bet_to_pot_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, action)

        # Insert or update Results table
        for result in data["results"]:
            cursor.execute("""
                INSERT OR REPLACE INTO Results (hand_id, player_id, position, cards, net_result, won_pot, showdown, final_equity, preflop_equity, flop_equity, turn_equity, river_equity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*result, 0.0, 0.0, 0.0, 0.0, 0.0))
        
        # Update equities in the Results table
        update_equities_in_db(conn, data, evaluator)
        
        conn.commit()

    except Exception as e:
        logging.error(f"Error inserting/updating data into the database: {e}")
        conn.rollback()

def update_equities_in_db(conn, data, evaluator):
    """
    Update the equity values in the database for a given hand.
    :param conn: Database connection object.
    :param data: Parsed hand data.
    :param evaluator: Treys Evaluator instance.
    """
    if not data:
        return

    try:
        cursor = conn.cursor()

        # Get players and their cards
        players = data["players"]
        hero_cards = data["hero_cards"]
        board_cards = data["board_cards"]

        # Calculate equities at each street
        preflop_equity = calculate_equity_monte_carlo(players, [], evaluator)
        flop_equity = calculate_equity_monte_carlo(players, board_cards["Flop"].split() if board_cards["Flop"] else [], evaluator)
        turn_equity = calculate_equity_monte_carlo(players, board_cards["Turn"].split() if board_cards["Turn"] else [], evaluator)
        river_equity = calculate_equity_monte_carlo(players, board_cards["River"].split() if board_cards["River"] else [], evaluator)

        # Update equities in the Results table
        for result in data["results"]:
            hand_id, player_id, position, cards, net_result, won_pot, showdown = result
            cursor.execute("""
                UPDATE Results
                SET final_equity = ?,
                    preflop_equity = ?,
                    flop_equity = ?,
                    turn_equity = ?,
                    river_equity = ?
                WHERE hand_id = ? AND player_id = ?
            """, (
                river_equity.get(player_id, 0.0),
                preflop_equity.get(player_id, 0.0),
                flop_equity.get(player_id, 0.0),
                turn_equity.get(player_id, 0.0),
                river_equity.get(player_id, 0.0),
                hand_id,
                player_id
            ))

        conn.commit()

    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            logging.warning("Database is locked. Retrying...")
            time.sleep(1)  # Wait for 1 second before retrying
            update_equities_in_db(conn, data, evaluator)  # Retry
        else:
            logging.error(f"Error updating equities in the database: {e}")
            conn.rollback()
    except Exception as e:
        logging.error(f"Error updating equities in the database: {e}")
        conn.rollback()

if __name__ == "__main__":
    main()