# Poker Hand Analysis System

A comprehensive poker hand analysis system that processes hand histories, calculates GTO solutions, and provides strategic insights using advanced algorithms including Monte Carlo CFR and Neural Network.

## Project Structure

```

├── convert/
│   ├── converter.py      # Main conversion utility (7.8KB)
│   ├── filter.py         # Data filtering module (1.5KB)
│   ├── analizer.py       # Hand analysis module (6.5KB)
│   ├── convert.py        # Format conversion module (5.7KB)
│   └── hand_parser.py    # Hand history parser (4.5KB)
├── db/
│   ├── poker_hand_analysis.py
│   ├── GTO_Solver_Data.py
│   └── populate_exploitability.py
└── readme.md            # Project documentation
```

## Overview

This system consists of several Python modules that work together to:
- Parse and store poker hand histories
- Calculate Game Theory Optimal (GTO) solutions
- Analyze player tendencies and exploitability
- Provide strategic recommendations

## Components

### 1. Conversion Utilities (`convert/`)

#### `converter.py`
- Main conversion utility for hand histories
- Handles multiple input formats
- Standardizes data representation
- Processes batch conversions

#### `filter.py`
- Filters and validates hand history data
- Removes invalid or incomplete hands
- Applies custom filtering rules

#### `analizer.py`
- Analyzes hand patterns and statistics
- Generates preliminary reports
- Identifies key decision points

#### `convert.py`
- Handles format-specific conversions
- Supports multiple poker site formats
- Standardizes output format

#### `hand_parser.py`
- Parses raw hand history files
- Extracts key information
- Validates hand structure

### 2. Database Management (`poker_hand_analysis.py`)
- Handles database setup and management
- Parses poker hand histories
- Stores hand data, player actions, and results
- Calculates equity and pot odds

Key features:
- Hand history parsing with detailed validation
- Monte Carlo equity calculations
- Comprehensive database schema for poker hand analysis

### 3. GTO Solver (`GTO_Solver_Data.py`)
- Implements Monte Carlo Counterfactual Regret Minimization (MCCFR+)
- Calculates Nash equilibrium strategies and Poker Matrix
- Provides action recommendations based on:
  - Hand equity
  - Position
  - Stack sizes
  - Player tendencies
  - Board texture

Key features:
- Advanced MCCFR+ implementation
- Dynamic strategy adjustment
- Multi-street analysis
- Position-based strategy optimization

### 4. Exploitability Analysis (`populate_exploitability.py`)
- Analyzes player tendencies and patterns
- Calculates exploitability scores
- Uses neural networks for pattern recognition
- Provides detailed player profiling

Key features:
- Neural network-based tendency analysis
- Advanced player profiling
- Multi-dimensional exploitability scoring
- Action frequency analysis

## Database Schema

### Main Tables
1. **Games**
   - Hand ID
   - Stakes
   - Game type
   - Number of players
   - Blinds information

2. **Players**
   - Player information
   - Position
   - Stack sizes
   - Hero identification

3. **Hands**
   - Hero cards
   - Board cards
   - Pot sizes per street

4. **Actions**
   - Player actions
   - Betting amounts
   - Street information
   - Pot odds

5. **Results**
   - Hand results
   - Net winnings
   - Showdown information
   - Equity calculations

6. **GTO_Solutions**
   - Optimal strategies
   - Nash equilibrium calculations
   - Recommended actions
   - Expected values

7. **Exploitability**
   - Player tendencies
   - VPIP/PFR metrics
   - Aggression factors
   - Exploitability scores

## Usage

1. Initialize the database:
```python
from poker_hand_analysis import main as db_setup
db_setup()
```

2. Process hand histories:
```python
from GTO_Solver_Data import GTOSolverData
solver = GTOSolverData()
solver.process_hand(hand_id)
```

3. Analyze player exploitability:
```python
from populate_exploitability import main as analyze_exploitability
analyze_exploitability()
```

4. Convert hand histories:
```python
from convert.converter import convert_hands
convert_hands(input_path, output_path)
```

## Requirements

- Python 3.7+
- SQLite3
- PyTorch
- NumPy
- Treys (poker hand evaluation library)
- Deuces (poker hand evaluation library)

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install torch numpy treys deuces
```

3. Set up the database path in the configuration files

## Configuration

Key configuration variables are found at the top of each module:
- Database paths
- Folder paths for hand histories
- Simulation parameters
- Neural network parameters

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.