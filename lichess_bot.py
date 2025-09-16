import os
import chess
import chess.engine
import berserk
import sys
sys.path.append('searchless_chess/src')  # So imports work if running from project root

from searchless_chess.src.engines import constants

# 1. === Lichess Authentication ===
# Put your personal BOT token here (create one in lichess.org under Account -> API Access Tokens)
API_TOKEN = "lip_N3hVrHtVdtwb16pOuw11"
session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)

# 2. === Connect to Stockfish ===
# Make sure stockfish is installed and available in PATH, or give full path
MODEL_NAME = 'BC_270M' 
engine = constants.ENGINE_BUILDERS[MODEL_NAME]()

count_move = 0
count_illegal = 0

# 3. === Function: Play a move ===
def play_game(game_id):
    board = chess.Board(fen=event["game"]["initialFen"])
    print(f"Started game {game_id}")

    # Stream moves from lichess
    for event in client.bots.stream_game_state(game_id):
        if event["type"] == "gameState":
            moves = event["moves"].split()
            board = chess.Board()
            for move in moves:
                board.push_uci(move)

            # If it's our turn
            if board.turn == chess.WHITE and event["white"]["id"] == client.account.get()["id"] \
            or board.turn == chess.BLACK and event["black"]["id"] == client.account.get()["id"]:

                # Use stockfish to choose best move
                best_move = engine.play(board, legal=False) 
                count_move += 1
                if not board.is_legal(best_move):
                    best_move = engine.play(board, legal=True)
                    count_illegal += 1

                print(f"My move: {best_move}")

                # Send move to lichess
                client.bots.make_move(game_id, best_move)

# 4. === Main loop: Accept challenges and start games ===
print("Listening for events...")
for event in client.bots.stream_incoming_events():
    if event["type"] == "challenge":
        challenge_id = event["challenge"]["id"]
        variant = event["challenge"]["variant"]["key"]  # "standard", "chess960", etc.
        time_limit = event["challenge"]["timeControl"]["limit"]       # in seconds
        increment = event["challenge"]["timeControl"]["increment"]   # seconds per move

        total_time = time_limit + increment * 40  # approx. total time for 40 moves

        # Only accept Chess960 Blitz (~<=10 min total)
        if variant in ["chess960", "standard"] and total_time <= 600:  # 600s = 10 min
            print(f"Accepting challenge {challenge_id}: {variant}, {time_limit}+{increment}")
            client.bots.accept_challenge(challenge_id)
        else:
            print(f"Skipping challenge {challenge_id}: {variant}, {time_limit}+{increment}")

    elif event["type"] == "gameStart":
        game_id = event["game"]["id"]
        play_game(game_id)
