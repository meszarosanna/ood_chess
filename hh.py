import pandas as pd
import chess

def main():
    
    count_all_ood = 0
    count_more_pieces = 0
    count_more_b = 0
    count_same_color = 0
    
    for i in range(102):
        with open(f"details_{i*5200000}.csv", "r") as f:
            numbers = [int(line.strip()) for line in f if line.strip()]
            count_all_ood += numbers[0]
            count_more_pieces += numbers[1]
            count_more_b += numbers[2]
            count_same_color += numbers[3]
    print(count_all_ood)
    print(count_more_pieces)
    print(count_more_b)
    print(count_same_color)





if __name__ == '__main__':
    main()