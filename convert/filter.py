import os
import glob

# Set your input directory where the hand_*.txt files are located
input_dir = 'D:\hand'  # Change this to your input directory if needed
output_dir = r'D:\hand\2'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# The specific line to remove
poker_tools_line = 'Poker Tools by CardRunners - http://www.cardrunners.com/'

# Process each hand history file
for input_path in glob.glob(os.path.join(input_dir, 'hand_*.txt')):
    has_hero_result = False
    filtered_lines = []
    
    with open(input_path, 'r') as file:
        for line in file:
            # Check if the stripped line matches the Poker Tools line
            if line.strip() == poker_tools_line:
                continue  # Skip this line
            # Add the line to the filtered list
            filtered_lines.append(line)
            # Check if this line contains Hero won or lost
            if 'Hero' in line and (' won' in line or ' lost' in line):
                has_hero_result = True
    
    # If the file contains Hero result, save it to the output directory
    if has_hero_result:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as file:
            file.writelines(filtered_lines)
        print(f"Processed and copied: {filename}")
    else:
        print(f"Skipped {os.path.basename(input_path)} (no Hero result found)")

print("Processing complete.")