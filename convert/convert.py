import re
import glob
import os

def replace_suits(html_line):
    """Convert HTML suit images to letter notation and remove all HTML tags"""
    replaced = re.sub(
        r'<img\s+[^>]*alt=" of (\w+)"[^>]*>',
        lambda m: m.group(1)[0].lower(),
        html_line
    )
    replaced = re.sub(r'<[^>]+>', '', replaced)
    return replaced.strip()

def convert_and_save_hand_history(file_path):
    """Process and overwrite a single hand history file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.split('<br />')
        
        processed_lines = []
        flop_cards = None
        turn_card = None
        river_card = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            processed_line = replace_suits(line)

            # Handle special cases
            if processed_line.startswith('Poker Stars'):
                parts = processed_line.split(', ')
                if len(parts) >= 5:
                    corrected = f"{parts[0]}, {parts[3]}, {parts[4]}"
                    processed_lines.append(corrected)
                else:
                    processed_lines.append(processed_line)
                continue

            elif 'Poker Tools' in processed_line:
                links = re.findall(r'<a\s+href="([^"]*)"[^>]*>([^<]*)</a>', line)
                if len(links) >= 2:
                    tools_line = f"{links[0][1]} by {links[1][1]} - {links[1][0]}"
                    processed_lines.append(tools_line)
                else:
                    processed_lines.append(processed_line)
                continue

            elif processed_line.startswith('Preflop:'):
                parts = processed_line.split(':')
                if len(parts) >= 3:
                    corrected_preflop = f"{parts[0]}:{parts[-1].strip()}"
                    processed_lines.append(corrected_preflop)
                else:
                    processed_lines.append(processed_line)
                continue

            elif processed_line.startswith('Flop:'):
                parts = processed_line.split(')', 1)
                if len(parts) > 1:
                    cards_part = parts[1].split('(')[0].strip()
                    flop_cards = cards_part
                processed_lines.append(processed_line)

            elif processed_line.startswith('Turn:'):
                parts = processed_line.split(')', 1)
                if len(parts) > 1:
                    cards_part = parts[1].split('(')[0].strip()
                    turn_card = cards_part
                processed_lines.append(processed_line)

            elif processed_line.startswith('River:'):
                parts = processed_line.split(')', 1)
                if len(parts) > 1:
                    cards_part = parts[1].split('(')[0].strip()
                    river_card = cards_part
                processed_lines.append(processed_line)

            elif processed_line.startswith('Final Board:'):
                continue

            elif processed_line.startswith('Results:'):
                parts = processed_line.split(':')
                if len(parts) >= 3:
                    corrected_results = f"{parts[0]}:{parts[2].strip()}"
                    processed_lines.append(corrected_results)
                else:
                    processed_lines.append(processed_line)
                continue

            else:
                processed_lines.append(processed_line)

        # Build final board line
        if flop_cards and turn_card and river_card:
            final_board_line = f"Final Board: {flop_cards} {turn_card} {river_card}"
            for i, pline in enumerate(processed_lines):
                if pline.startswith('Results:'):
                    processed_lines.insert(i, final_board_line)
                    break

        # Create final output
        output = '\n'.join(line for line in processed_lines if line.strip())
        
        # Overwrite original file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output)
        return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def batch_convert_hands(directory):
    """Process all hand history files in a directory"""
    file_pattern = os.path.join(directory, "hand_*.txt")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    success_count = 0
    failure_count = 0
    
    print(f"Starting conversion of {len(files)} files in {directory}")
    for file_path in files:
        print(f"Processing {os.path.basename(file_path)}...", end=' ', flush=True)
        if convert_and_save_hand_history(file_path):
            success_count += 1
            print("✓")
        else:
            failure_count += 1
            print("✗")
    
    print("\nConversion Summary:")
    print(f"Successfully converted: {success_count} files")
    print(f"Failed conversions:    {failure_count} files")
    print(f"Total processed:       {success_count + failure_count} files")

if __name__ == "__main__":
    hand_directory = r"D:\hand"
    
    # Safety check
    if not os.path.exists(hand_directory):
        print(f"Directory not found: {hand_directory}")
        exit(1)
        
    if not os.path.isdir(hand_directory):
        print(f"Path is not a directory: {hand_directory}")
        exit(1)
    
    print("=== Poker Hand History Converter ===")
    batch_convert_hands(hand_directory)
    print("\nOperation completed. Always keep backups of your original files!")