#Copyright 2025 Anna Meszaros

import chess
import berserk
import chess.variant

from searchless_chess.src.engines import constants

# Lichess Authentication
API_TOKEN = "Your_Lichess_API_Token_Here"
session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)


MODEL_NAME = 'BC_270M' 
engine = constants.ENGINE_BUILDERS[MODEL_NAME]()



def play_game(game_id, board, color,variant):
    print(f"Started game {game_id}")
    
    if variant == "standard":
        board = chess.Board()
    elif variant == "chess960":
        board = chess.Board(fen, chess960=True)
    elif variant == "horde":
        board = chess.variant.HordeBoard()


    # Stream moves from lichess
    for event in client.bots.stream_game_state(game_id):
        print(event)
        if event["type"] == "gameFull":
            if board.turn == color:
                best_move = engine.play(board, legal=True) 

                print(f"My move: {best_move}")
                
                # Send move to lichess
                client.bots.make_move(game_id, best_move)
                board.push(best_move)

        elif event["type"] == "gameState":
            if "status" in event and event["status"] != "started":
                print(f"Result: {event["status"]}")
                break
            moves = event["moves"].split()
            if variant == "standard":
                board = chess.Board()
            elif variant == "chess960":
                board = chess.Board(fen, chess960=True)
            elif variant == "horde":
                board = chess.variant.HordeBoard()

            for move in moves:
                board.push_uci(move)
            if board.can_claim_fifty_moves() or board.is_repetition():
                print("Game is over. ")
                break

            
            if board.turn == color:
                
                best_move = engine.play(board, legal=True) 

                print(f"My move: {best_move}")
                

                # Send move to lichess
                client.bots.make_move(game_id, best_move)
                board.push(best_move)

# 4. === Main loop: Accept challenges and start games ===
print("Listening for events...")
for event in client.bots.stream_incoming_events():
    if event["type"] == "challenge":
        challenge_id = event["challenge"]["id"]
        variant = event["challenge"]["variant"]["key"] 

        tc = event["challenge"].get("timeControl", {})
        time_limit = tc.get("limit", 0)
        increment = tc.get("increment", 0) 

        total_time = time_limit + increment * 40 

        # Only accept Chess960 Blitz (~<=10 min total)
        if variant in ["chess960", "standard", "horde"] and total_time <= 600: 
            print(f"Accepting challenge {challenge_id}: {variant}, {time_limit}+{increment}")
            client.bots.accept_challenge(challenge_id)
        else:
            print(f"Skipping challenge {challenge_id}: {variant}, {time_limit}+{increment}")
            client.bots.decline_challenge(challenge_id)

    elif event["type"] == "gameStart":
        
        game = event["game"]
        game_id = game["id"]
        game_info = client.games.export(game_id)
        fen = game_info.get("initialFen", "startpos")
        variant = game_info.get("variant")
        try:
            my_color = game_info["players"]["white"]["user"]["id"].lower()
        except KeyError:
            my_color = game_info["players"]["white"]
        if my_color == "chessllmodel":
            color = chess.WHITE
        else:
            color = chess.BLACK

        print(f'My color: {color}')
        play_game(game_id, fen, color, variant)
