import os
import re

# Define the input and output directories
input_dir = r'd:\hand\4'
output_dir = r'd:\hand\5'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to convert the log content
def convert_log(content):
    # Split the content into lines
    lines = content.splitlines()
    
    # Initialize the converted lines list
    converted_lines = []
    
    # Dictionary to map player nicknames to their positions
    player_positions = {}
    
    # Step 1: Process player stack lines to map nicknames to positions
    for line in lines[1:6]:  # Player stack lines are typically lines 1-5
        if ': $' in line:  # Check if it's a player stack line
            player_match = re.match(r'(.+?) \((\w+)\): \$[\d.,]+ \(\d+\.?\d* bb\)', line)
            if player_match:
                nickname, position = player_match.groups()
                player_positions[nickname] = position
    
    # Step 2: Replace nicknames with positions in the entire content
    content = content.splitlines()
    for i, line in enumerate(content):
        for nickname, position in player_positions.items():
            if nickname != "Hero":  # Don't touch Hero lines
                line = re.sub(rf'{re.escape(nickname)} \(\w+\)', position, line)  # Replace nickname (position)
                line = re.sub(rf'{re.escape(nickname)}', position, line)  # Replace nickname in actions
        content[i] = line
    
    # Step 3: Process the modified content
    i = 0
    while i < len(content):
        line = content[i]
        if re.match(r'^[A-Za-z ]+,', line):
            # Process the first line to get the game type and stakes
            stakes_match = re.search(r'\$([\d,.]+)/\$([\d,.]+)', line)
            game_type_match = re.search(r'\$(.+?) No Limit Hold\'em Cash', line)
            if stakes_match and game_type_match:
                stakes = f"${stakes_match.group(1)}/${stakes_match.group(2)}"
                game_type = "NLH"
                players_match = re.search(r'(\d+) Players', line)
                players = players_match.group(1) if players_match else "?"
                converted_lines.append(f"{stakes}, {game_type}, {players} Players\n\n")
        elif ': $' in line:  # Player stack lines
            # Remove commas from stack amounts
            line = re.sub(r'\$([\d,]+\.?\d*)', lambda m: f"${m.group(1).replace(',', '')}", line)
            converted_lines.append(line + '\n')
        elif any(line.startswith(street) for street in ["Preflop:", "Flop:", "Turn:", "River:", "Final Board:"]):
            # Simplify headers
            header = line.split(':')[0]
            if header == "Preflop":
                hero_cards = re.search(r'Hero is \w+ with (.+)', line)
                if hero_cards:
                    converted_lines.append(f"{header}: Hero {hero_cards.group(1)}\n")
                    i += 1
                    if i < len(content):
                        actions = content[i].split(', ')
                        for action in actions:
                            converted_lines.append(action + '\n')
            elif header in ["Flop", "Turn", "River"]:
                # Remove commas from pot amounts
                line = re.sub(r'\(\$([\d,]+\.?\d*)\)', lambda m: f"(${m.group(1).replace(',', '')})", line)
                converted_lines.append(line + '\n')
                i += 1
                if (i < len(content) and 
                    not any(content[i].startswith(h) for h in ["Preflop:", "Flop:", "Turn:", "River:", "Final Board:", "Results:"])):
                    actions = content[i].split(', ')
                    for action in actions:
                        # Remove commas from action amounts
                        action = re.sub(r'\$([\d,]+\.?\d*)', lambda m: f"${m.group(1).replace(',', '')}", action)
                        converted_lines.append(action + '\n')
            elif header == "Final Board":
                # Before adding Final Board, ensure we've included all streets
                final_cards = line.split(': ')[1].strip().split()  # Get the final board cards
                if len(final_cards) == 5:  # If we have all 5 cards
                    # Get the last pot amount from Results line
                    results_line = next((l for l in content if l.startswith("Results:")), None)
                    if results_line:
                        pot_match = re.search(r'\$([\d.]+) pot', results_line)
                        pot_amount = pot_match.group(1) if pot_match else "0.00"
                        
                        # Check if we need to add missing Turn or River
                        if not any("Turn:" in l for l in converted_lines[-5:]):
                            turn_card = final_cards[3]  # Fourth card is turn
                            converted_lines.append(f"Turn: (${pot_amount}) {turn_card}\n")
                        if not any("River:" in l for l in converted_lines[-5:]):
                            river_card = final_cards[4]  # Fifth card is river
                            converted_lines.append(f"River: (${pot_amount}) {river_card}\n")
                # Now add the Final Board line
                converted_lines.append(line + '\n')
                
        elif line.startswith("Results:"):
            # Remove commas from pot and rake amounts
            results = line.split(':', 1)[1].strip()
            results = re.sub(r'\$([\d,]+\.?\d*)', lambda m: f"${m.group(1).replace(',', '')}", results)
            pot_match = re.search(r'\$([\d.]+) pot', results)
            rake_match = re.search(r'\(\$([\d.]+) rake\)', results)
            
            if pot_match and rake_match:
                converted_lines.append(f"Results: ${pot_match.group(1)} pot (${rake_match.group(1)} rake)\n")
            else:
                converted_lines.append(f"Results: {results}\n")
            
            # Process player results with comma removal
            i += 1
            while i < len(content) and ("showed" in content[i] or "lost" in content[i] or "won" in content[i]):
                player_result = content[i]
                player_result = re.sub(r'\$([\d,]+\.?\d*)', lambda m: f"${m.group(1).replace(',', '')}", player_result)
                player_result = re.sub(r'(?<=lost)(?=\w)', '\n', player_result)
                player_result = re.sub(r'(?<=won)(?=\w)', '\n', player_result)
                converted_lines.append(player_result + '\n')
                i += 1
            continue
        else:
            # Split actions and remove commas from amounts
            actions = re.split(r', (?=\w+ )', line)
            for action in actions:
                action = re.sub(r'(\w+) (raises to|bets|checks|folds|calls|raises) \$?([\d,]+\.?\d*)', 
                              lambda m: f"{m.group(1)} {m.group(2)} ${m.group(3).replace(',', '')}", 
                              action)
                action = re.sub(r'(\w+) is all-in', r'\1 all-in', action)
                converted_lines.append(action + '\n')
        i += 1
    
    # Join the converted lines into a single string
    return ''.join(converted_lines)

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.startswith('hand_') and filename.endswith('.txt'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read the content of the input file
        with open(input_path, 'r') as file:
            content = file.read()
        
        # Convert the content
        converted_content = convert_log(content)
        
        # Write the converted content to the output file
        with open(output_path, 'w') as file:
            file.write(converted_content)

print("Conversion complete.")
