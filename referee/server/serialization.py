# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from dataclasses import asdict
from typing import Literal

from ..game import *
from ..game.board import CellState


def serialize_game_board(board: Board) -> list[list[int]]:
    """
    Serialize a game board to a dictionary.
    """
    sz_board = [BOARD_N * [0] for _ in range(BOARD_N)]
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            sz_board[r][c] = serialize_game_board_cell(board[Coord(r, c)])

    return sz_board


def serialize_game_board_cell(cell: CellState) -> int:
    """
    Serialize a game board cell to a dictionary.
    """
    match cell.state:
        case PlayerColor.RED:
            return 1
        case PlayerColor.BLUE:
            return -1
        case 'LilyPad':
            return 2
        case None:
            return 0
        case _:
            raise ValueError(f"Invalid cell state: {cell}")


def serialize_game_player(player: Player | PlayerColor | None) -> int:
    """
    Serialize a game player to a dictionary.
    """
    if isinstance(player, Player):
        player = player.color

    return int(player) if player != None else 0


def serialize_game_action(action: Action) -> dict:
    """
    Serialize a game action to a dictionary.
    """
    match action:
        case MoveAction(coord, directions):
            if type(directions) != tuple:
                directions = (directions,)
            return {
                "type": "MoveAction",
                "coord": [coord.r, coord.c],
                "directions": [[*d] for d in directions],
            }
        
        case GrowAction():
            return {
                "type": "GrowAction",
            }


def serialize_game_update(update: GameUpdate) -> dict:
    """
    Serialize a game update to a dictionary.
    """
    update_cls_name = update.__class__.__name__
    update_payload = {}

    match update:
        case PlayerInitialising(player):
            update_payload = {
                "player": serialize_game_player(player),
            }

        case GameBegin(board):
            update_payload = {
                "board": serialize_game_board(board),
            }

        case TurnBegin(turn_id, player):
            update_payload = {
                "turnId": turn_id,
                "player": serialize_game_player(player),
            }

        case TurnEnd(turn_id, player, action):
            update_payload = {
                "turnId": turn_id,
                "player": serialize_game_player(player),
                "action": serialize_game_action(action),
            }

        case BoardUpdate(board):
            update_payload = {
                "board": serialize_game_board(board),
            }

        case GameEnd(winner):
            update_payload = {
                "winner": serialize_game_player(winner),
            }

    return {
        "type": f"GameUpdate:{update_cls_name}",
        **update_payload,
    }
