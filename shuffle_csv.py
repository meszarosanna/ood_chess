#!/usr/bin/env python3
"""
Script to shuffle the lines of ood_puzzles_chess960.csv while preserving the header.
"""

import csv
import random
import sys
from pathlib import Path

def shuffle_csv_lines(input_file, output_file=None, seed=None):
    """
    Shuffle the lines of a CSV file while preserving the header.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output CSV file. If None, overwrites input file.
        seed (int, optional): Random seed for reproducible shuffling
    """
    if seed is not None:
        random.seed(seed)
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        return False
    
    # Read all lines
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print("Error: File must have at least a header and one data line.")
        return False
    
    # Separate header from data lines
    header = lines[0]
    data_lines = lines[1:]
    
    # Shuffle the data lines
    shuffled_data = data_lines.copy()
    random.shuffle(shuffled_data)
    
    # Combine header with shuffled data
    shuffled_lines = [header] + shuffled_data
    
    # Determine output file
    if output_file is None:
        output_path = input_path
    else:
        output_path = Path(output_file)
    
    # Write shuffled content
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        f.writelines(shuffled_lines)
    
    print(f"Successfully shuffled {len(data_lines)} data lines.")
    print(f"Output written to: {output_path}")
    return True

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python shuffle_csv.py <input_file> [output_file] [seed]")
        print("Example: python shuffle_csv.py ood_puzzles_chess960.csv")
        print("Example: python shuffle_csv.py ood_puzzles_chess960.csv shuffled_output.csv")
        print("Example: python shuffle_csv.py ood_puzzles_chess960.csv shuffled_output.csv 42")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    success = shuffle_csv_lines(input_file, output_file, seed)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
