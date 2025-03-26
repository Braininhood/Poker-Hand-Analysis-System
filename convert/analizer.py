import glob
import re
from collections import defaultdict

def parse_hand_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines

    hand_data = {
        'site': None,
        'stakes': None,
        'hero_position': None,
        'players': [],
        'hero_hole_cards': [],
        'actions': defaultdict(list),
        'board': [],
        'results': {'hero_net': 0.0, 'pot': 0.0, 'rake': 0.0}
    }

    # Skip irrelevant lines
    lines = [line for line in lines if not line.startswith("Poker Tools")]

    if not lines:
        return None  # Ignore empty or non-game files

    # Parse site and stakes
    if lines:
        first_line_parts = lines[0].split(',')
        if len(first_line_parts) >= 2:
            hand_data['site'] = first_line_parts[0].strip()
            stakes_part = first_line_parts[1].strip().split(' ', 1)[0]
            hand_data['stakes'] = stakes_part

    # Parse players and hero position
    for line in lines[1:]:
        if ':' in line and not any(street in line for street in ['Preflop:', 'Flop:', 'Turn:', 'River:']):
            position_match = re.search(r'\((.*?)\)', line)
            stack_match = re.search(r'\$([\d.]+)', line)

            if position_match and stack_match:
                position = position_match.group(1)
                stack = float(stack_match.group(1))

                if 'Hero' in line:
                    hand_data['hero_position'] = position
                    hand_data['players'].append({'position': position, 'stack': stack, 'is_hero': True})
                else:
                    position = line.split(':')[0].strip()
                    hand_data['players'].append({'position': position, 'stack': stack, 'is_hero': False})
        else:
            break

    # Parse hole cards
    for line in lines:
        if 'Preflop:' in line and 'with' in line:
            cards = line.split('with ')[1].split()
            hand_data['hero_hole_cards'] = cards
            break

    # Parse actions and board
    current_street = None
    for line in lines:
        if line.startswith('Preflop:'):
            current_street = 'preflop'
        elif line.startswith('Flop:'):
            current_street = 'flop'
            board_match = re.search(r'\((.*?)\)', line)
            if board_match:
                hand_data['board'].extend(board_match.group(1).split())
        elif line.startswith('Turn:'):
            current_street = 'turn'
            board_match = re.search(r'\((.*?)\)', line)
            if board_match:
                hand_data['board'].append(board_match.group(1).strip())
        elif line.startswith('River:'):
            current_street = 'river'
            board_match = re.search(r'\((.*?)\)', line)
            if board_match:
                hand_data['board'].append(board_match.group(1).strip())

        if current_street and ':' in line:
            action_part = line.split(':', 1)[1].strip()
            if action_part:
                hand_data['actions'][current_street].extend([a.strip() for a in action_part.split(', ')])

    # Parse results
    for line in lines:
        if 'Results:' in line:
            pot_match = re.search(r'\$([\d.]+)', line)
            if pot_match:
                hand_data['results']['pot'] = float(pot_match.group(1))

        if 'rake' in line:
            rake_match = re.search(r'\$([\d.]+)', line)
            if rake_match:
                hand_data['results']['rake'] = float(rake_match.group(1))

        if 'Hero' in line and 'net' in line:
            net_match = re.search(r'\(([+-]?\$?\d+\.\d+)\s+net\)', line)
            if net_match:
                net_str = net_match.group(1).replace('$', '')
                hand_data['results']['hero_net'] = float(net_str)

    return hand_data

def analyze_hands(hands):
    analysis = {
        'total_hands': len(hands),
        'vpip': 0,
        'pfr': 0,
        'total_profit': 0.0,
        'showdowns': 0,
        'preflop_actions': defaultdict(int),
        'positions': defaultdict(list)
    }

    for hand in hands:
        analysis['total_profit'] += hand['results']['hero_net']
        
        # Track VPIP (Voluntarily Put Money in Pot)
        if any(action.startswith('Hero') and ('calls' in action or 'raises' in action) 
               for action in hand['actions']['preflop']):
            analysis['vpip'] += 1
            
        # Track PFR (Preflop Raise)
        if any(action.startswith('Hero raises') for action in hand['actions']['preflop']):
            analysis['pfr'] += 1
            
        # Track showdowns
        if 'river' in hand['actions'] and len(hand['actions']['river']) > 0:
            analysis['showdowns'] += 1
            
        # Track position stats
        analysis['positions'][hand['hero_position']].append(hand['results']['hero_net'])

    if analysis['total_hands'] > 0:
        analysis['vpip_percent'] = (analysis['vpip'] / analysis['total_hands']) * 100
        analysis['pfr_percent'] = (analysis['pfr'] / analysis['total_hands']) * 100
        analysis['showdown_percent'] = (analysis['showdowns'] / analysis['total_hands']) * 100
        
    return analysis

def main():
    hand_files = glob.glob(r'D:\hand\hand_*.txt')  # Using raw string (r'...') for Windows paths
    all_hands = []
    
    for file in hand_files:
        try:
            hand = parse_hand_file(file)
            if hand:  # Ignore None results (invalid files)
                all_hands.append(hand)
        except Exception as e:
            print(f"Error parsing {file}: {str(e)}")
    
    if not all_hands:
        print("No valid hands found.")
        return

    analysis = analyze_hands(all_hands)
    
    # Print analysis results
    print(f"Total Hands Analyzed: {analysis['total_hands']}")
    print(f"Total Profit: ${analysis['total_profit']:.2f}")
    print(f"VPIP: {analysis.get('vpip_percent', 0):.1f}%")
    print(f"PFR: {analysis.get('pfr_percent', 0):.1f}%")
    print(f"Showdown Rate: {analysis.get('showdown_percent', 0):.1f}%")
    
    # Print position statistics
    print("\nPositional Performance:")
    for position, profits in analysis['positions'].items():
        total = sum(profits)
        count = len(profits)
        avg = total / count if count > 0 else 0
        print(f"{position}: {count} hands, Avg Profit: ${avg:.2f}")

if __name__ == "__main__":
    main()
