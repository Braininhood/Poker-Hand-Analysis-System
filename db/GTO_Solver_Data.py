from deuces import Card, Evaluator
import sqlite3
import logging
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Database setup
DB_PATH = "d:/hand/db/poker.db"

class MCCFRPlus:
    def __init__(self):
        self.regret = defaultdict(lambda: np.zeros(5))  # Regret for fold, call, raise, bet, check
        self.strategy = defaultdict(lambda: np.ones(5) / 5)  # Current strategy
        self.cumulative_strategy = defaultdict(lambda: np.zeros(5))  # Cumulative strategy
        self.num_iterations = 0
        self.exploration_rate = 1.0  # Start with high exploration
        self.exploration_decay = 0.9999  # Slowly reduce exploration

    def update_strategy(self, info_set: str, strategy: np.ndarray):
        """
        Update the cumulative strategy and current strategy for an information set.
        Args:
            info_set: The information set identifier
            strategy: The current strategy probabilities
        """
        # Update cumulative strategy
        self.cumulative_strategy[info_set] += strategy * (self.num_iterations + 1)
        
        # Update current strategy
        self.strategy[info_set] = strategy

    def get_strategy(self, info_set: str) -> np.ndarray:
        """Get current strategy with exploration."""
        regret_pos = np.maximum(self.regret[info_set], 0)
        if np.sum(regret_pos) > 0:
            strategy = regret_pos / np.sum(regret_pos)
        else:
            strategy = np.ones(len(regret_pos)) / len(regret_pos)
        
        # Add exploration
        exploration = self.exploration_rate * (np.ones(len(strategy)) / len(strategy))
        return (1 - self.exploration_rate) * strategy + exploration

    def run_iteration(self, hand_id: int, street: str, num_players: int):
        """Run one iteration of MCCFR+ with improved exploration."""
        info_set = f"{hand_id}_{street.lower()}_{num_players}"
        strategy = self.get_strategy(info_set)
        
        # Sample action with exploration
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(5)  # Random exploration (now 5 actions)
        else:
            # Use a more sophisticated action selection based on strategy
            action = np.random.choice(5, p=strategy)  # Exploitation

        # Calculate counterfactual value with more realistic modeling
        cf_value = self._calculate_counterfactual_value(hand_id, street.lower(), num_players, action)

        # Update regrets with more aggressive updates
        for a in range(5):  # Now 5 actions
            regret = cf_value[a] - cf_value[action]
            self.update_regret(info_set, a, regret)

        # Update strategy with the new regrets
        self.update_strategy(info_set, self.get_strategy(info_set))

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.num_iterations += 1

    def _calculate_counterfactual_value(self, hand_id: int, street: str, num_players: int, action: int) -> np.ndarray:
        """Calculate counterfactual values with more realistic modeling."""
        # Get actual hand data from database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Get hero's cards and board cards
            cursor.execute("""
                SELECT hero_cards, board_cards
                FROM Hands
                WHERE hand_id = ?
            """, (hand_id,))
            hand_result = cursor.fetchone()
            
            # Get both pot_before and pot_after for the current street
            cursor.execute("""
                SELECT pot_before, pot_after
                FROM Actions
                WHERE hand_id = ? AND street = ?
                ORDER BY action_id DESC
                LIMIT 1
            """, (hand_id, street.lower()))
            pot_result = cursor.fetchone()
            
            # Get action amounts for the street
            cursor.execute("""
                SELECT amount
                FROM Actions
                WHERE hand_id = ? AND street = ?
                ORDER BY action_id
            """, (hand_id, street.lower()))
            action_amounts = [row[0] for row in cursor.fetchall()]
            
            # Get opponent fold frequency
            cursor.execute("""
                SELECT COUNT(*) as total_actions,
                       SUM(CASE WHEN action_type = 'fold' THEN 1 ELSE 0 END) as fold_count
                FROM Actions
                WHERE hand_id = ? AND street = ? AND player_id != (
                    SELECT player_id FROM Players WHERE hand_id = ? AND is_hero = 1
                )
            """, (hand_id, street.lower(), hand_id))
            opp_fold_result = cursor.fetchone()
            
        if hand_result and pot_result and opp_fold_result and action_amounts:
            hero_cards, board_cards = hand_result
            pot_before, pot_after = pot_result
            total_actions, fold_count = opp_fold_result
            
            # Calculate pot change and opponent fold frequency
            pot_change = pot_after - pot_before
            opponent_fold_frequency = fold_count / total_actions if total_actions > 0 else 0.2  # Default to 20% if no data
            
            # Calculate average action amount
            avg_action_amount = np.mean(action_amounts) if action_amounts else 0
            
            # Convert cards to deuces format and evaluate hand strength
            evaluator = Evaluator()
            hero_hand = [Card.new(card) for card in hero_cards.split()]
            board = [Card.new(card) for card in board_cards.split()] if board_cards else []
            hand_strength = evaluator.evaluate(hero_hand, board)
            
            # Calculate equity
            max_strength = 7462
            equity = 1 - (hand_strength / max_strength)
            
            # Calculate counterfactual values using the improved formula
            cf_values = self._calculate_cf_values(hand_strength, pot_change, equity, opponent_fold_frequency, avg_action_amount)
            
            return cf_values
        else:
            # Default values if no hand data
            return np.array([0.1, 0.8, 0.5, 0.6, 0.5])

    def _calculate_cf_values(self, hand_strength: float, pot_size: float, equity: float, 
                           opponent_fold_frequency: float, avg_action_amount: float) -> np.ndarray:
        """
        Calculate counterfactual values based on hand strength, pot size, equity, 
        opponent tendencies, and average action amount.
        """
        # Calculate base value based on equity and pot size
        equity = self.calculate_equity(hand_id) if self.calculate_equity(hand_id) is not None else 0.5
        pot_size = self._calculate_counterfactual_value(hand_strength, pot_size, equity, opponent_fold_frequency, avg_action_amount)  # Calculate pot_size using the method
        opponent_fold_frequency = self.calculate_opponent_fold_frequency(hand_id) if self.calculate_opponent_fold_frequency(hand_id) is not None else 0.2
        base_value = equity * pot_size
        #logging.info(f"Base value: {base_value}, pot_size: {pot_size}, equity: {equity}, opponent_fold_frequency: {opponent_fold_frequency}, avg_action_amount: {avg_action_amount}")

        # Adjust base value based on average action amount
        if avg_action_amount > 0:
            base_value *= (1 + (avg_action_amount / pot_size))

        # Adjust CFVs based on opponent tendencies and action-specific logic
        cf_values = np.array([
            base_value * 0.1,  # Fold (low value, as folding forfeits the pot)
            base_value * (1 - opponent_fold_frequency),  # Call (scaled by opponent's calling frequency)
            base_value * (1 + opponent_fold_frequency),  # Raise (scaled by opponent's folding frequency)
            base_value * 1.0,  # Bet (neutral value, adjust based on hand strength)
            base_value * 0.5  # Check (neutral value, adjust based on hand strength)
        ])
        
        # Scale values based on hand strength (stronger hands get higher values)
        hand_strength_factor = 1 - (hand_strength / 7462)
        cf_values *= hand_strength_factor
        
        return cf_values

    def update_regret(self, info_set: str, action: int, regret: float):
        """
        Update the regret for a specific information set and action.
        Args:
            info_set: The information set identifier
            action: The action index (0-4)
            regret: The regret value to add
        """
        # Add regret with a learning rate
        learning_rate = 1.0 / (1.0 + self.num_iterations)
        self.regret[info_set][action] += learning_rate * regret

    def get_average_strategy(self, info_set: str) -> np.ndarray:
        """
        Calculate the average strategy over all iterations.
        Args:
            info_set: The information set identifier
        Returns:
            The average strategy probabilities as a numpy array
        """
        if self.num_iterations == 0:
            return np.ones(5) / 5  # Uniform strategy if no iterations
        
        # Calculate average strategy by normalizing cumulative strategy
        cumulative = self.cumulative_strategy[info_set]
        if np.sum(cumulative) > 0:
            return cumulative / np.sum(cumulative)
        else:
            return np.ones(5) / 5  # Uniform strategy if no cumulative strategy

class GTOSolverData:
    def __init__(self):
        try:
            self.conn = sqlite3.connect(DB_PATH)
            logging.info("Database connection successful.")
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
        self.mccfr = MCCFRPlus()  # Initialize MCCFR+

    def reset_database(self):
        """Drop all tables in the database."""
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.executescript("""
                    DROP TABLE IF EXISTS GTO_Solutions;
                """)
                logging.info("Database reset successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error resetting database: {e}")

    def create_tables(self):
        """Create GTO Solutions table if it doesn't exist."""
        with self.conn:
            cursor = self.conn.cursor()

            # Create GTO_Solutions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS GTO_Solutions (
                    hand_id INTEGER,
                    player_id INTEGER,
                    position TEXT,
                    street TEXT,
                    action TEXT,
                    frequency REAL,
                    expected_value REAL,
                    optimal_range TEXT,
                    nash_equilibrium TEXT,
                    poker_matrix TEXT,
                    current_strategy TEXT,
                    recommended_action TEXT,
                    PRIMARY KEY (hand_id, player_id, street, action)
                );
            """)

    def get_hero_player_info(self, hand_id: int) -> Tuple[int, str, float, float]:
        """
        Get the hero player's information for a given hand.
        Args:
            hand_id: The hand ID.
        Returns:
            A tuple containing (player_id, position, stack_size, bb_size).
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT player_id, position, stack_size, bb_size
                FROM Players
                WHERE hand_id = ? AND is_hero = 1
            """, (hand_id,))
            hero_info = cursor.fetchone()
            if not hero_info:
                logging.warning(f"No hero player found for hand_id {hand_id}")
                return None
            #logging.info(f"Hero player info: {hero_info}")
            return hero_info

    def get_hero_actions(self, hand_id: int, player_id: int) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get all actions for the hero player in a given hand, grouped by street.
        Args:
            hand_id: The hand ID.
            player_id: The hero's player ID.
        Returns:
            A dictionary where keys are streets (e.g., 'preflop', 'flop') and values are lists of (action_type, is_all_in).
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT street, action_type, is_all_in
                FROM Actions
                WHERE hand_id = ? AND player_id = ?
                ORDER BY action_id
            """, (hand_id, player_id))
            actions = cursor.fetchall()
            if not actions:
                logging.warning(f"No actions found for hero in hand_id {hand_id}")
                return {}
            #logging.info(f"Hero actions: {actions}")
            # Group actions by street
            actions_by_street = defaultdict(list)
            for street, action_type, is_all_in in actions:
                #logging.info(f"Street: {street}, Action type: {action_type}, Is all in: {is_all_in}")
                actions_by_street[street.lower()].append((action_type.lower(), is_all_in))
            return actions_by_street

    def get_active_players_by_street(self, hand_id: int) -> Dict[str, int]:
        """
        Get the number of active players for each street.
        Args:
            hand_id: The hand ID.
        Returns:
            A dictionary mapping streets to the number of active players.
        """
        with self.conn:
            cursor = self.conn.cursor()
            
            # Get all actions for the hand
            cursor.execute("""
                SELECT street, player_id, action_type
                FROM Actions
                WHERE hand_id = ?
                ORDER BY action_id
            """, (hand_id,))
            all_actions = cursor.fetchall()
            
            if not all_actions:
                logging.warning(f"No actions found for hand_id {hand_id}")
                return {}

            # Initialize active players with preflop count
            cursor.execute("SELECT num_players FROM Games WHERE hand_id = ?", (hand_id,))
            num_players_result = cursor.fetchone()
            num_players_preflop = num_players_result[0] if num_players_result else 0
            
            active_players = set(range(1, num_players_preflop + 1))  # Player IDs start from 1
            players_by_street = {'preflop': num_players_preflop}
            
            # Track folded players
            folded_players = set()
            
            # Process actions street by street
            current_street = 'preflop'
            for street, player_id, action_type in all_actions:
                street = street.lower()
                action_type = action_type.lower()
                
                # If we've moved to a new street, update the active players count
                if street != current_street:
                    players_by_street[current_street] = len(active_players)
                    current_street = street
                
                # Update folded players
                if action_type == 'fold':
                    folded_players.add(player_id)
                    active_players.discard(player_id)
            
            # Add the final street's player count
            if current_street not in players_by_street:
                players_by_street[current_street] = len(active_players)
                
            return players_by_street

    def calculate_action_frequency(self, actions: List[Tuple[str, int]], street: str, position: str) -> Dict[str, float]:
        """
        Calculate the frequency of actions for a given street and position.
        Args:
            actions: List of (action_type, is_all_in) tuples.
            street: The street (e.g., 'preflop', 'flop').
            position: The player's position (e.g., 'BTN', 'SB').
        Returns:
            A dictionary of action frequencies, normalized to sum to 1.
        """
        if not actions:
            #logging.warning(f"No actions found for street {street} and position {position}")
            return {'fold': 0.0, 'call': 0.0, 'raise': 0.0, 'bet': 0.0, 'check': 0.0}

        # Count actions, weighted by their importance
        action_counts = defaultdict(float)
        for action, is_all_in in actions:
            weight = 1.0  # Default weight
            if is_all_in:
                weight = 2.0  # All-in actions are more significant
            action_counts[action] += weight

        # Normalize frequencies to sum to 1
        total_weight = sum(action_counts.values())
        if total_weight > 0:
            frequencies = {action: round(count / total_weight, 4) for action, count in action_counts.items()}
            #logging.info(f"Action frequencies for {street} and {position}: {frequencies}")
            return frequencies
        else:
            logging.warning(f"No valid actions found for street {street} and position {position}")
            return {action: 0.0 for action in action_counts.keys()}

    def process_hand(self, hand_id: int):
        try:
            logging.info(f"Starting to process hand_id: {hand_id}")

            # Get hero player info
            hero_info = self.get_hero_player_info(hand_id)
            if not hero_info:
                logging.warning(f"Skipping hand_id {hand_id} because no hero player found.")
                return
            player_id, position, stack_size, bb_size = hero_info

            # Get number of active players by street
            players_by_street = self.get_active_players_by_street(hand_id)
            if not players_by_street:
                logging.warning(f"No player data found for hand_id {hand_id}")
                return

            # Get hero actions grouped by street
            actions_by_street = self.get_hero_actions(hand_id, player_id)
            if not actions_by_street:
                logging.warning(f"No actions found for hero in hand_id {hand_id}")
                return

            # Process each street
            for street, actions in actions_by_street.items():
                # Get number of players for current street
                num_players = players_by_street.get(street, 0)

                # Run MCCFR+ iterations for the current street
                for _ in range(1000):  # Run 1000 iterations per street
                    self.mccfr.run_iteration(hand_id, street, num_players)

                # Calculate poker matrix using MCCFR+ strategy
                info_set = f"{hand_id}_{street.lower()}_{num_players}"
                mccfr_strategy = self.mccfr.get_average_strategy(info_set)
                poker_matrix = self.calculate_poker_matrix(hand_id, actions, mccfr_strategy)
                #logging.info(f"Poker matrix: {poker_matrix}")
                
                # Calculate action frequencies
                action_frequencies = self.calculate_action_frequency(actions, street, position)

                # Check if hero is all-in on this street
                is_hero_all_in = any(is_all_in for _, is_all_in in actions)

                # Calculate expected value
                expected_value = self.calculate_ev(hand_id, player_id).get(street.lower(), 0.0)

                # Calculate Nash Equilibrium for the street
                nash_equilibrium = self.calculate_nash_equilibrium(hand_id, street.lower(), num_players)

                # Calculate current strategy
                current_strategy = self.analyze_current_strategy(
                    actions, position, street, stack_size * bb_size, stack_size,
                    poker_matrix, nash_equilibrium,
                    self.calculate_equity(hand_id),
                    self.calculate_optimal_range(position, street, len(actions)),
                    is_hero_all_in
                )

                # Get recommended action based on analysis
                recommended_action = self.get_recommended_action(actions, self.calculate_equity(hand_id), current_strategy)

                # Insert GTO solution into the database
                self.insert_gto_solution(
                    hand_id, player_id, position, street, actions,
                    action_frequencies,
                    expected_value,
                    self.calculate_optimal_range(position, street, len(actions)),
                    nash_equilibrium, poker_matrix, current_strategy, recommended_action
                )

        except Exception as e:
            logging.error(f"Error processing hand {hand_id}: {e}", exc_info=True)

    def insert_gto_solution(self, hand_id: int, player_id: int, position: str, street: str, actions: List[Tuple[str, int]],
                        frequency: Dict[str, float], expected_value: Dict[str, float], optimal_range: List[str],
                        nash_equilibrium: Dict[str, float], poker_matrix: Dict[str, float],
                        current_strategy: Dict[str, float], recommended_action: str):
        """
        Insert a GTO solution into the database.
        """
        with self.conn:
            cursor = self.conn.cursor()
            for action, _ in actions:
                #logging.info(f"Inserting GTO solution for hand_id={hand_id}, player_id={player_id}, street={street}, action={action}")
                cursor.execute("""
                    INSERT OR REPLACE INTO GTO_Solutions (
                        hand_id, player_id, position, street, action,
                        frequency, expected_value, optimal_range,
                        nash_equilibrium, poker_matrix, current_strategy,
                        recommended_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hand_id, player_id, position, street, action,
                    json.dumps(frequency), expected_value, json.dumps(optimal_range),
                    json.dumps(nash_equilibrium), json.dumps(poker_matrix),
                    json.dumps(current_strategy), recommended_action
                ))
            self.conn.commit()
            logging.info("GTO solution inserted successfully.")

    def calculate_poker_matrix(self, hand_id: int, actions: List[Tuple[str, int]], mccfr_strategy: np.ndarray) -> Dict[str, float]:
        """
        Calculate the poker matrix for a list of actions, using MCCFR+ strategy and street information.
        """
        # Get the current street based on the number of board cards
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT hero_cards, board_cards
                FROM Hands
                WHERE hand_id = ?
            """, (hand_id,))
            result = cursor.fetchone()
            
        if not result:
            logging.warning(f"No hand data found for hand_id {hand_id}")
            return {'fold': 0.25, 'call': 0.25, 'raise': 0.25, 'bet': 0.25, 'check': 0.25}

        hero_cards, board_cards = result
        board_cards_list = board_cards.split() if board_cards else []

        # Determine street and log cards
        if not board_cards_list:
            street = 'preflop'
            #logging.info(f"Preflop: Hero cards = {hero_cards}, No board cards")
        else:
            if len(board_cards_list) == 3:
                street = 'flop'
                #logging.info(f"Flop: Hero cards = {hero_cards}, Board cards = {' '.join(board_cards_list[:3])}")
            elif len(board_cards_list) == 4:
                street = 'turn'
                #logging.info(f"Turn: Hero cards = {hero_cards}, Board cards = {' '.join(board_cards_list[:4])}")
            elif len(board_cards_list) == 5:
                street = 'river'
                #logging.info(f"River: Hero cards = {hero_cards}, Board cards = {' '.join(board_cards_list[:5])}")
            else:
                street = 'preflop'
                logging.warning(f"Unexpected number of board cards ({len(board_cards_list)}), defaulting to preflop")

        # Log MCCFR+ strategy
        action_map = {0: 'fold', 1: 'call', 2: 'raise', 3: 'bet', 4: 'check'}
        mccfr_matrix = {action_map[i]: float(prob) for i, prob in enumerate(mccfr_strategy)}
        #logging.info(f"MCCFR+ Strategy: {mccfr_matrix}")

        # Calculate action frequencies based on actual actions
        action_counts = defaultdict(int)
        for action, _ in actions:
            action_counts[action] += 1
        total_actions = len(actions)
        #logging.info(f"Observed Actions: {dict(action_counts)} (Total: {total_actions})")

        # Combine MCCFR+ strategy with observed frequencies using weighted average
        poker_matrix = {}
        weight_observed = 0.3  # Weight for observed actions
        weight_mccfr = 0.7     # Weight for MCCFR+ strategy

        for action in set(action_counts.keys()).union(set(mccfr_matrix.keys())):
            observed_freq = action_counts.get(action, 0) / total_actions if total_actions > 0 else 0
            mccfr_prob = mccfr_matrix.get(action, 0)
            poker_matrix[action] = round(
                (weight_observed * observed_freq) + (weight_mccfr * mccfr_prob),
                4
            )
           # logging.info(f"Combining {action}: Observed={observed_freq}, MCCFR={mccfr_prob}, Result={poker_matrix[action]}")

        # Normalize the matrix to sum to 1
        total = sum(poker_matrix.values())
        if total > 0:
            for action in poker_matrix:
                poker_matrix[action] = round(poker_matrix[action] / total, 4)
            #logging.info(f"Normalized Poker Matrix: {poker_matrix} (Total={total})")
        else:
            logging.warning("Poker matrix total is 0, using default values")
            poker_matrix = {'fold': 0.25, 'call': 0.25, 'raise': 0.25, 'bet': 0.25, 'check': 0.25}
                
        return poker_matrix

    def calculate_equity(self, hand_id: int) -> float:
        """
        Calculate equity for a given hand.
        Args:
            hand_id: The hand ID.
        Returns:
            The equity (probability of winning) as a float between 0 and 1.
        """
        try:
            with self.conn:
                cursor = self.conn.cursor()

                # Get hero's cards and board cards
                cursor.execute("""
                    SELECT hero_cards, board_cards
                    FROM Hands
                    WHERE hand_id = ?
                """, (hand_id,))
                result = cursor.fetchone()
                if not result:
                    logging.warning(f"No hand data found for hand_id {hand_id}")
                    return 0.5  # Default equity if no data is found

                hero_cards, board_cards = result

                # Check if hero_cards and board_cards are present
                if not hero_cards or not isinstance(hero_cards, str):
                    logging.warning(f"No hero cards found for hand_id {hand_id}")
                    return 0.5

                # Convert card strings to deuces format
                hero_cards_list = [Card.new(card) for card in hero_cards.split()]
                board_cards_list = [Card.new(card) for card in board_cards.split()] if board_cards else []

                # Calculate equity
                evaluator = Evaluator()
                hand_strength = evaluator.evaluate(hero_cards_list, board_cards_list)
                max_strength = 7462
                equity = round(1 - (hand_strength / max_strength), 4)

                #logging.info(f"Calculated equity for hand_id {hand_id}: {equity}")
                return equity

        except Exception as e:
            #logging.error(f"Error calculating equity for hand {hand_id}: {e}")
            return 0.5  # Default equity in case of error

    def calculate_ev(self, hand_id: int, player_id: int) -> Dict[str, float]:
        """
        Calculate EV using MCCFR+ counterfactual values, actual net results,
        and action amounts.
        """
        ev_by_street = {}
        
        # Get hero's actions and results
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT a.street, a.action_type, a.amount, r.net_result
                FROM Actions a
                LEFT JOIN Results r ON a.hand_id = r.hand_id AND a.player_id = r.player_id
                WHERE a.hand_id = ? AND a.player_id = ?
            """, (hand_id, player_id))
            actions = cursor.fetchall()

        for street, action_type, amount, net_result in actions:
            street = street.lower()
            action_type = action_type.lower()
            num_players = self.get_active_players_by_street(hand_id).get(street.lower(), 0)
            if num_players == 0:
                continue
                
            info_set = f"{hand_id}_{street.lower()}_{num_players}"
            
            # Get counterfactual values from MCCFR+
            cf_values = self.mccfr._calculate_counterfactual_value(hand_id, street.lower(), num_players, 0)
            
            # Calculate EV as weighted sum of counterfactual values
            strategy = self.mccfr.get_strategy(info_set)
            ev = sum(prob * value for prob, value in zip(strategy, cf_values))
            
            # Adjust EV based on action type and amount
            action_factor = {
                'fold': 0.1,
                'call': 0.8 * (1 + (amount / 100)),  # Scale call EV based on amount
                'raise': 1.2 * (1 + (amount / 50)),  # Scale raise EV more aggressively
                'bet': 1.0 * (1 + (amount / 75)),
                'check': 0.9
            }.get(action_type.lower(), 1.0)
            ev *= action_factor
            
            # Adjust EV with actual net result if available
            if net_result is not None:
                # Weight the adjustment based on action type and amount
                adjustment_weight = {
                    'fold': 0.2,
                    'call': 0.5 * (1 + (amount / 200)),
                    'raise': 0.7 * (1 + (amount / 100)),
                    'bet': 0.6 * (1 + (amount / 150)),
                    'check': 0.4
                }.get(action_type.lower(), 0.5)
                ev = (ev * (1 - adjustment_weight)) + (net_result * adjustment_weight)
            
            ev_by_street[street] = round(ev, 4)
            #logging.info(f"EV for {street} ({action_type}, amount={amount}): {ev_by_street[street.lower()]} (net_result: {net_result})")
        
        return ev_by_street

    def calculate_nash_equilibrium(self, hand_id: int, street: str, num_players: int) -> Dict[str, float]:
        """
        Calculate Nash Equilibrium with stronger factor weights.
        Args:
            hand_id: The hand ID
            street: The street (e.g., 'preflop', 'flop')
            num_players: Number of active players
        Returns:
            A dictionary of action probabilities rounded to 4 decimal places
        """
        # Get hero player info from database
        hero_info = self.get_hero_player_info(hand_id)
        #logging.info(f"Hero info: {hero_info}")
        if not hero_info:
            logging.warning(f"No hero player found for hand_id {hand_id}")
            return {'fold': 0.25, 'call': 0.25, 'raise': 0.25, 'bet': 0.25, 'check': 0.25}
            
        player_id, position, stack_size, bb_size = hero_info
        #logging.info(f"Player ID: {player_id}, Position: {position}, Stack size: {stack_size}, BB size: {bb_size}")

        # Verify we have valid hero info
        if not all([player_id, position, stack_size, bb_size]):
            logging.warning(f"Incomplete hero info for hand_id {hand_id}")
            return {'fold': 0.25, 'call': 0.25, 'raise': 0.25, 'bet': 0.25, 'check': 0.25}
            
        # Get hero actions for the current street
        actions = self.get_hero_actions(hand_id, player_id).get(street.lower(), [])
        #logging.info(f"Actions: {actions}")
        
        # Get equity and other relevant factors
        equity = self.calculate_equity(hand_id)
        #logging.info(f"Equity: {equity}")
        
        # Base strategy from MCCFR+
        info_set = f"{hand_id}_{street.lower()}_{num_players}"
        base_strategy = self.mccfr.get_average_strategy(info_set)
        #logging.info(f"Base strategy: {base_strategy}")
        
        # If no base strategy, return uniform distribution
        if base_strategy is None or len(base_strategy) != 5:
            logging.warning("No valid base strategy found, returning uniform distribution")
            return {'fold': 0.25, 'call': 0.25, 'raise': 0.25, 'bet': 0.25, 'check': 0.25}
        
        # Stronger position factors
        position_factors = {
            'UTG': {'fold': 1.5, 'call': 0.5, 'raise': 0.6, 'bet': 0.4, 'check': 0.5},
            'MP': {'fold': 1.3, 'call': 0.7, 'raise': 0.8, 'bet': 0.6, 'check': 0.6},
            'CO': {'fold': 0.6, 'call': 1.4, 'raise': 1.5, 'bet': 1.3, 'check': 0.8},
            'BTN': {'fold': 0.4, 'call': 1.6, 'raise': 1.7, 'bet': 1.4, 'check': 1.0},
            'SB': {'fold': 0.8, 'call': 1.2, 'raise': 1.3, 'bet': 1.1, 'check': 0.9},
            'BB': {'fold': 0.8, 'call': 1.3, 'raise': 1.1, 'bet': 1.0, 'check': 1.0}
        }.get(position, {'fold': 1.0, 'call': 1.0, 'raise': 1.0, 'bet': 1.0, 'check': 1.0})
        #logging.info(f"Position factors: {position_factors}")
        
        # Stronger equity adjustment
        equity_factor = min(max(equity * 3.0, 0.3), 1.7)
        #logging.info(f"Equity factor: {equity_factor}")
        
        # Action history with stronger weighting
        action_count = defaultdict(int)
        for action_type, _ in actions:
            action_count[action_type] += 1
        total_actions = len(action_type)
        #logging.info(f"Action count: {action_count}")

        # Stronger action history factors
        action_factors = {
            'fold': 1.0 + (1.0 * action_count.get('fold', 0) / max(total_actions, 1)),
            'call': 1.0 + (0.8 * action_count.get('call', 0) / max(total_actions, 1)),
            'raise': 1.0 + (1.2 * action_count.get('raise', 0) / max(total_actions, 1)),
            'bet': 1.0 + (0.8 * action_count.get('bet', 0) / max(total_actions, 1)),
            'check': 1.0 + (0.9 * action_count.get('check', 0) / max(total_actions, 1))
        }
        #logging.info(f"Action factors: {action_factors}")
        
        # Stack size adjustment
        stack_factor = min(max(stack_size / (bb_size * 10), 0.5), 1.5)
        #logging.info(f"Stack factor: {stack_factor}")
        
        # Combine all factors with stronger weights
        action_names = ["fold", "call", "raise", "bet", "check"]
        adjusted_strategy = np.zeros(5)
        #logging.info(f"Adjusted strategy_zeros: {adjusted_strategy}")
        # Stronger weights for factors
        position_weight = 0.7
        equity_weight = 0.5
        history_weight = 0.5
        stack_weight = 0.3
        base_weight = 0.1
        
        for i, action in enumerate(action_names):
            # Calculate weighted sum of factors
            adjusted_value = (
                (base_strategy[i] * base_weight) +
                (position_factors[action] * position_weight) +
                (equity_factor * equity_weight) +
                (action_factors[action] * history_weight) +
                (stack_factor * stack_weight)
            )
            adjusted_strategy[i] = adjusted_value
        
        # Apply minimum and maximum probabilities while preserving relative differences
        min_prob = 0.05
        max_prob = 0.9
        
        # First normalize the strategy
        total = np.sum(adjusted_strategy)
        if total > 0:
            adjusted_strategy /= total
            #logging.info(f"Adjusted strategy_total: {adjusted_strategy}")
        # Then apply clipping while preserving relative differences
        # Find the scaling factor that keeps values within bounds
        scale = 1.0
        if np.max(adjusted_strategy) > max_prob:
            scale = max_prob / np.max(adjusted_strategy)
        if np.min(adjusted_strategy) < min_prob:
            scale = min(scale, min_prob / np.min(adjusted_strategy))

        adjusted_strategy *= scale
        #logging.info(f"Adjusted strategy_scale: {adjusted_strategy}")
        # Renormalize after scaling
        total = np.sum(adjusted_strategy)
        #logging.info(f"Total: {total}") 
        if total > 0:
            adjusted_strategy /= total
        else:
            # If total is 0, use base strategy instead of uniform distribution
            adjusted_strategy = base_strategy
            #logging.info(f"Strategy_norm: {adjusted_strategy} = {base_strategy}")
            if np.sum(adjusted_strategy) > 0:
                adjusted_strategy /= np.sum(adjusted_strategy)
                #logging.info(f"Adjusted strategy_norm: {adjusted_strategy}")
            else:
                adjusted_strategy = np.ones(5) / 5
                #logging.info(f"Else_Adjusted strategy_norm: {adjusted_strategy}")

        #logging.info(f"Final adjusted strategy: {adjusted_strategy}")

        # Convert to dictionary and round to 4 decimal places
        nash_equilibrium = {action: round(float(prob), 4) for action, prob in zip(action_names, adjusted_strategy)}
        #logging.info(f"Nash Equilibrium: {nash_equilibrium}")
        
        return nash_equilibrium

    def calculate_optimal_range(self, position: str, street: str, num_players: int) -> List[str]:
        """
        Calculate optimal range based on position, street, and number of players
        """
        # Define standard preflop ranges for 6-max games
        preflop_ranges = {
            "UTG": "AA,KK,QQ,JJ,AKs",
            "MP": "AA,KK,QQ,JJ,AKs,AQs",
            "CO": "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,ATs",
            "BTN": "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,KQs,AKo",
            "SB": "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs",
            "BB": "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs"
        }
        
        if street == 'preflop':
            return preflop_ranges.get(position, []).split(',')
        else:
            # Postflop ranges are based on preflop ranges and board texture
            preflop_range = preflop_ranges.get(position, []).split(',')
            return self._filter_postflop_range(preflop_range, street)

    def _filter_postflop_range(self, preflop_range: List[str], street: str) -> List[str]:
        """
        Filter preflop range based on street and board texture
        """
        # This is a simplified version - in practice, this would need to consider
        # actual board cards and hand strength
        if street == 'flop':
            return [hand for hand in preflop_range if 
                    hand in ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs']]
        elif street == 'turn':
            return [hand for hand in preflop_range if 
                    hand in ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs']]
        elif street == 'river':
            return [hand for hand in preflop_range if 
                    hand in ['AA', 'KK', 'QQ', 'AKs']]
        return preflop_range

    def analyze_current_strategy(self, actions: List[Tuple[str, int]], position: str, street: str, pot_size: float,
                                 stack_size: float, poker_matrix: Dict[str, float], nash_equilibrium: Dict[str, float],
                                 equity: float, optimal_range: List[str], is_hero_all_in: bool) -> Dict[str, float]:
        """
        Analyze the current strategy based on actions and contextual data.
        """
        # Base strategy weights
        weights = {
            'nash': 0.4,        # Weight for Nash equilibrium
            'poker_matrix': 0.3, # Weight for poker matrix
            'equity': 0.2,      # Weight for equity
            'position': 0.1     # Weight for position
        }

        # Position factors with check
        position_factors = {
            'UTG': {'fold': 1.2, 'call': 0.8, 'raise': 0.7, 'bet': 0.6, 'check': 0.5},
            'MP': {'fold': 1.1, 'call': 0.9, 'raise': 0.8, 'bet': 0.7, 'check': 0.6},
            'CO': {'fold': 0.8, 'call': 1.1, 'raise': 1.0, 'bet': 0.9, 'check': 0.8},
            'BTN': {'fold': 0.6, 'call': 1.3, 'raise': 1.2, 'bet': 1.1, 'check': 1.0},
            'SB': {'fold': 0.7, 'call': 1.1, 'raise': 1.0, 'bet': 0.9, 'check': 0.8},
            'BB': {'fold': 0.7, 'call': 1.2, 'raise': 1.1, 'bet': 1.0, 'check': 0.9}
        }.get(position, {'fold': 1.0, 'call': 1.0, 'raise': 1.0, 'bet': 1.0, 'check': 1.0})

        # Calculate equity adjustment with check
        equity_adjustment = {
            'fold': max(0, 1 - equity * 1.5),
            'call': equity * 0.8,
            'raise': equity * 1.2,
            'bet': equity * 1.0,
            'check': equity * 0.9
        }

        # Calculate stack-to-pot ratio adjustment with check
        spr = stack_size / pot_size if pot_size > 0 else 1.0
        spr_adjustment = {
            'fold': min(1.0, 0.8 + (0.2 * spr)),
            'call': min(1.0, 0.9 + (0.1 * spr)),
            'raise': min(1.0, 1.0 + (0.2 * spr)),
            'bet': min(1.0, 0.9 + (0.1 * spr)),
            'check': min(1.0, 0.7 + (0.3 * spr))
        }

        # Combine all factors
        combined_strategy = defaultdict(float)
        for action in ['fold', 'call', 'raise', 'bet', 'check']:
            # Base components
            nash_component = nash_equilibrium.get(action, 0.2) * weights['nash']
            matrix_component = poker_matrix.get(action, 0.2) * weights['poker_matrix']
            
            # Adjusted components
            equity_component = equity_adjustment[action] * weights['equity']
            position_component = position_factors[action] * weights['position']
            
            # Combine with SPR adjustment
            combined = (nash_component + matrix_component + equity_component + position_component) * spr_adjustment[action]
            combined_strategy[action] = combined

        # Normalize to sum to 1
        total = sum(combined_strategy.values())
        if total > 0:
            for action in combined_strategy:
                combined_strategy[action] = round(combined_strategy[action] / total, 4)
        else:
            # Fallback to uniform distribution if total is 0
            combined_strategy = {'fold': 0.2, 'call': 0.2, 'raise': 0.2, 'bet': 0.2, 'check': 0.2}

        # Adjust for all-in situations
        if is_hero_all_in:
            combined_strategy['fold'] = 0.0
            combined_strategy['check'] = 0.0
            total = sum(combined_strategy.values())
            if total > 0:
                for action in combined_strategy:
                    combined_strategy[action] = round(combined_strategy[action] / total, 4)

        return dict(combined_strategy)

    def get_recommended_action(self, actions: List[Tuple[str, int]], equity: float, current_strategy: Dict[str, float]) -> str:
        """
        Get the recommended action based on equity, current strategy, and MCCFR+ analysis.
        Args:
            actions: List of (action_type, is_all_in) tuples
            equity: Current hand equity (0-1)
            current_strategy: Current strategy probabilities
        Returns:
            Recommended action as string ('fold', 'call', 'raise', 'bet', 'check')
        """
        if not actions:
            return 'check'  # Default to check if no actions available

        # Get the last action to understand the current situation
        last_action, is_all_in = actions[-1] if actions else ('check', False)

        # Calculate action strength score based on multiple factors
        action_scores = {
            'fold': self._calculate_fold_score(equity, current_strategy, last_action, is_all_in),
            'call': self._calculate_call_score(equity, current_strategy, last_action, is_all_in),
            'raise': self._calculate_raise_score(equity, current_strategy, last_action, is_all_in),
            'bet': self._calculate_bet_score(equity, current_strategy, last_action, is_all_in),
            'check': self._calculate_check_score(equity, current_strategy, last_action, is_all_in)
        }

        # Get the action with the highest score
        recommended_action = max(action_scores.keys(), key=lambda k: action_scores[k])
        
        # Apply final validation to ensure the recommendation makes sense
        if is_all_in and recommended_action in ['fold', 'check']:
            # If all-in, only consider call or raise
            recommended_action = max(['call', 'raise'], key=lambda k: action_scores[k])
            
        return recommended_action

    def _calculate_fold_score(self, equity: float, strategy: Dict[str, float], last_action: str, is_all_in: bool) -> float:
        """Calculate fold score based on multiple factors."""
        if is_all_in:
            return -1.0  # Never fold when all-in
            
        base_score = strategy.get('fold', 0.0)
        
        # Adjust based on last action
        if last_action in ['raise', 'bet']:
            base_score *= 1.2  # More likely to fold against aggression
        elif last_action == 'check':
            base_score *= 0.5  # Less likely to fold after check
            
        # Adjust based on equity
        base_score *= max(0, 1 - (equity * 1.5))
        
        return base_score

    def _calculate_call_score(self, equity: float, strategy: Dict[str, float], last_action: str, is_all_in: bool) -> float:
        """Calculate call score based on multiple factors."""
        base_score = strategy.get('call', 0.0)
        
        # Adjust based on last action
        if last_action in ['raise', 'bet']:
            base_score *= 1.1  # More likely to call against aggression
        elif last_action == 'check':
            base_score *= 0.8  # Less likely to call after check
            
        # Adjust based on equity
        base_score *= min(1.0, equity * 1.2)
        
        # Increase score if all-in
        if is_all_in:
            base_score *= 1.5
            
        return base_score

    def _calculate_raise_score(self, equity: float, strategy: Dict[str, float], last_action: str, is_all_in: bool) -> float:
        """Calculate raise score based on multiple factors."""
        if is_all_in:
            return -1.0  # Can't raise when all-in
            
        base_score = strategy.get('raise', 0.0)
        
        # Adjust based on last action
        if last_action in ['raise', 'bet']:
            base_score *= 1.3  # More likely to raise against aggression
        elif last_action == 'check':
            base_score *= 0.7  # Less likely to raise after check
            
        # Adjust based on equity
        base_score *= min(1.0, equity * 1.5)
        
        return base_score

    def _calculate_bet_score(self, equity: float, strategy: Dict[str, float], last_action: str, is_all_in: bool) -> float:
        """Calculate bet score based on multiple factors."""
        if is_all_in:
            return -1.0  # Can't bet when all-in
            
        base_score = strategy.get('bet', 0.0)
        
        # Adjust based on last action
        if last_action == 'check':
            base_score *= 1.2  # More likely to bet after check
        elif last_action in ['raise', 'bet']:
            base_score *= 0.6  # Less likely to bet after aggression
            
        # Adjust based on equity
        base_score *= min(1.0, equity * 1.3)
        
        return base_score

    def _calculate_check_score(self, equity: float, strategy: Dict[str, float], last_action: str, is_all_in: bool) -> float:
        """Calculate check score based on multiple factors."""
        if is_all_in:
            return -1.0  # Can't check when all-in
            
        base_score = strategy.get('check', 0.0)
        
        # Adjust based on last action
        if last_action == 'check':
            base_score *= 1.1  # More likely to check after check
        elif last_action in ['raise', 'bet']:
            base_score *= 0.4  # Less likely to check after aggression
            
        # Adjust based on equity
        base_score *= min(1.0, (1 - equity) * 1.1)
        
        return base_score

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logging.info("Database connection closed.")

# Usage
if __name__ == "__main__":
    gto_solver = GTOSolverData()
    gto_solver.reset_database()
    gto_solver.create_tables()
    try:
        # Get all hand_ids from the Games table
        cursor = gto_solver.conn.cursor()
        cursor.execute("SELECT hand_id FROM Games ORDER BY hand_id")
        hand_ids = [row[0] for row in cursor.fetchall()]

        if not hand_ids:
            logging.error("No hands found in the database")
            exit()

        # Process each hand
        for hand_id in hand_ids:
            logging.info(f"Processing hand_id: {hand_id}")
            gto_solver.process_hand(hand_id)

    finally:
        gto_solver.close()