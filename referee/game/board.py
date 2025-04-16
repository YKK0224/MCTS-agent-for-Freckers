# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from dataclasses import dataclass
from typing import Literal

from .coord import Coord, Direction
from .player import PlayerColor
from .actions import Action, MoveAction, GrowAction
from .exceptions import IllegalActionException
from .constants import *


ILLEGAL_RED_DIRECTIONS = set([
    Direction.Up,
    Direction.UpRight,
    Direction.UpLeft,
])

ILLEGAL_BLUE_DIRECTIONS = set([
    Direction.Down,
    Direction.DownRight,
    Direction.DownLeft,
])


@dataclass(frozen=True, slots=True)
class CellState:
    """
    A structure representing the state of a cell on the game board. A cell can
    be empty, have a lily pad, or be occupied by a frog of a given player colour
    (it is assumed the cell already has a lily pad if it is occupied).
    """
    state: PlayerColor | Literal["LilyPad"] | None = None

    def __post_init__(self):
        if self.state is None:
            object.__setattr__(self, "state", None)

    def __str__(self):
        return f"CellState({self.state})"
    
    def __iter__(self):
        yield self.state


@dataclass(frozen=True, slots=True)
class CellMutation:
    """
    A structure representing a change in the state of a single cell on the game
    board after an action has been played.
    """
    cell: Coord
    prev: CellState
    next: CellState

    def __str__(self):
        return f"CellMutation({self.cell}, {self.prev}, {self.next})"


@dataclass(frozen=True, slots=True)
class BoardMutation:
    """
    A structure representing a change in the state of the game board after an
    action has been played. Each mutation consists of a set of cell mutations.
    """
    action: Action
    cell_mutations: set[CellMutation]

    def __str__(self):
        return f"BoardMutation({self.cell_mutations})"


class Board:
    """
    A class representing the game board for internal use in the referee. 

    NOTE: Don't assume this class is an "ideal" board representation for your
    own agent; you should think carefully about how to design data structures
    for representing the state of a game with respect to your chosen strategy.
    This class has not been optimised beyond what is necessary for the referee.
    """
    def __init__(
        self, 
        initial_state: dict[Coord, CellState] = {},
        initial_player: PlayerColor = PlayerColor.RED
    ):
        """
        Create a new board. It is optionally possible to specify an initial
        board state (in practice this is only used for testing).
        """
        self._state: dict[Coord, CellState] = {
            Coord(r, c): CellState() 
            for r in range(BOARD_N) 
            for c in range(BOARD_N)
        }
        self._state.update(initial_state)

        if not initial_state:
            for r in [0, BOARD_N - 1]:
                for c in [0, BOARD_N - 1]:
                    self._state[Coord(r, c)] = CellState("LilyPad")

            for r in [1, BOARD_N - 2]:
                for c in range(1, BOARD_N - 1):
                    self._state[Coord(r, c)] = CellState("LilyPad")
                
            for c in range(1, BOARD_N - 1):
                self._state[Coord(0, c)] = CellState(PlayerColor.RED)
                self._state[Coord(BOARD_N - 1, c)] = CellState(PlayerColor.BLUE)

        self._turn_color: PlayerColor = initial_player
        self._history: list[BoardMutation] = []

    def __getitem__(self, cell: Coord) -> CellState:
        """
        Return the state of a cell on the board.
        """
        if not self._within_bounds(cell):
            raise IndexError(f"Cell position '{cell}' is invalid.")
        return self._state[cell]

    def apply_action(self, action: Action) -> BoardMutation:
        """
        Apply an action to a board, mutating the board state. Throws an
        IllegalActionException if the action is invalid.
        """
        match action:
            case MoveAction(coord, direction):
                mutation = self._resolve_move_action(action)
            case GrowAction():
                mutation = self._resolve_grow_action(action)
            case _:
                raise IllegalActionException(
                    f"Unknown action {action}", self._turn_color)

        for cell_mutation in mutation.cell_mutations:
            self._state[cell_mutation.cell] = cell_mutation.next
        
        self._history.append(mutation)
        self._turn_color = self._turn_color.opponent

        return mutation

    def undo_action(self) -> BoardMutation:
        """
        Undo the last action played, mutating the board state. Throws an
        IndexError if no actions have been played.
        """
        if len(self._history) == 0:
            raise IndexError("No actions to undo.")

        mutation: BoardMutation = self._history.pop()

        self._turn_color = self._turn_color.opponent

        for cell_mutation in mutation.cell_mutations:
            self._state[cell_mutation.cell] = cell_mutation.prev

        return mutation

    def render(self, use_color: bool=False, use_unicode: bool=False) -> str:
        """
        Returns a visualisation of the game board as a multiline string, with
        optional ANSI color codes and Unicode characters (if applicable).
        """
        def apply_ansi(str, bold=True, color=None):
            bold_code = "\033[1m" if bold else ""
            color_code = ""
            if color == "RED":
                color_code = "\033[31m"
            if color == "BLUE":
                color_code = "\033[34m"
            if color == "LilyPad":
                color_code = "\033[32m"
            return f"{bold_code}{color_code}{str}\033[0m"

        output = ""
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if self._cell_occupied(Coord(r, c)):
                    state = self._state[Coord(r, c)].state
                    if state == "LilyPad":
                        text = "*"
                    elif state == PlayerColor.RED or state == PlayerColor.BLUE:
                        text = "R" if state == PlayerColor.RED else "B"
                    else:
                        text = " "

                    if use_color:
                        output += apply_ansi(text, color=str(state), bold=False)
                    else:
                        output += text
                else:
                    output += "."
                output += " "
            output += "\n"
        return output
    
    @property
    def turn_count(self) -> int:
        """
        The number of actions that have been played so far.
        """
        return len(self._history)
    
    @property
    def turn_limit_reached(self) -> bool:
        """
        True iff the maximum number of turns has been reached.
        """
        return self.turn_count >= MAX_TURNS

    @property
    def turn_color(self) -> PlayerColor:
        """
        The player whose turn it is (represented as a colour).
        """
        return self._turn_color
    
    @property
    def game_over(self) -> bool:
        """
        True iff the game is over. 
        """
        if self.turn_limit_reached:
            return True

        # If a player's tokens are all in the final row, the game is over.
        if self._player_score(PlayerColor.RED) == BOARD_N - 2 or \
           self._player_score(PlayerColor.BLUE) == BOARD_N - 2:
            return True

        return False
    
    @property
    def winner_color(self) -> PlayerColor | None:
        """
        The player (color) who won the game, or None if no player has won.
        """
        if not self.game_over:
            return None
        
        red_score = self._player_score(PlayerColor.RED)
        blue_score = self._player_score(PlayerColor.BLUE)
        if red_score > blue_score:
            return PlayerColor.RED
        elif blue_score > red_score:
            return PlayerColor.BLUE

    def _within_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N
    
    def _cell_occupied(self, coord: Coord) -> bool:
        return self._state[coord].state != None
    
    def _cell_empty(self, coord: Coord) -> bool:
        return self._state[coord].state == None
    
    def _row_count(self, color: PlayerColor, row: int) -> int:
        return sum(
            (1 if self._state[Coord(row, c)].state == color else 0)
            for c in range(BOARD_N)
        )
    
    def _player_score(self, color: PlayerColor) -> int:
        return {
            PlayerColor.RED: self._row_count(PlayerColor.RED, BOARD_N - 1),
            PlayerColor.BLUE: self._row_count(PlayerColor.BLUE, 0)
        }[color]
    
    def _cell_occupied_by_player(self, coord: Coord) -> bool:
        return self._state[coord].state in [PlayerColor.RED, PlayerColor.BLUE]
    
    def _occupied_coords(self) -> set[Coord]:
        return set(filter(self._cell_occupied, self._state.keys()))
    
    def _assert_coord_valid(self, coord: Coord):
        if type(coord) != Coord or not self._within_bounds(coord):
            raise IllegalActionException(
                f"'{coord}' is not a valid coordinate.", self._turn_color)
        
    def _assert_coord_occ_by(self, coord: Coord, color: PlayerColor):
        if self._cell_empty(coord) or self._state[coord].state != color:
            raise IllegalActionException(
                f"Coord {coord} is not occupied by player {color}.", 
                    self._turn_color)
        
    def _assert_coord_empty(self, coord: Coord):
        if self._cell_occupied(coord):
            raise IllegalActionException(
                f"Coord {coord} is already occupied.", self._turn_color)
        
    def _assert_direction_valid(self, direction: Direction):
        if type(direction) != Direction:
            raise IllegalActionException(
                f"'{direction}' is not a valid direction.", self._turn_color)

    def _assert_has_attr(self, action: Action, attr: str):
        if not hasattr(action, attr):
            raise IllegalActionException(
                f"Action '{action}' is missing '{attr}' attribute.", 
                    self._turn_color)
        
    def _assert_direction_legal(self, direction: Direction, color: PlayerColor):
        if (color == PlayerColor.RED and (direction in ILLEGAL_RED_DIRECTIONS)) or \
           (color == PlayerColor.BLUE and (direction in ILLEGAL_BLUE_DIRECTIONS)):
                raise IllegalActionException(
                    f"Player {self._turn_color} cannot move in direction {direction}.",
                    self._turn_color
                )
        
    def _has_neighbour(self, coord: Coord, color: PlayerColor) -> bool:
        for direction in Direction:
            try:
                neighbour = coord + direction
                if self._state[neighbour].state == color:
                    return True
            except ValueError:
                pass
        return False
        
    def _resolve_move_destination(self, move_action: MoveAction) -> Coord:
        curr_coord = move_action.coord

        # Regular move to directly adjacent cell
        try:
            if len(move_action.directions) == 1 and \
                not self._cell_occupied_by_player(
                    curr_coord + move_action.directions[0]
                ):
                curr_coord += move_action.directions[0]
                return curr_coord
        except ValueError:
            raise IllegalActionException(
                f"Move action {move_action.coord} {move_action.directions} "
                "is prohibited.", self._turn_color)

        # If we reach this point, we expect one or more jumps
        for direction in move_action.directions:
            try:
                curr_coord += direction
                if not self._cell_occupied_by_player(curr_coord):
                    raise IllegalActionException(
                        f"Jump {move_action.coord} {move_action.directions} "
                        "over unoccupied cell is prohibited.", 
                        self._turn_color)
                
                curr_coord += direction
                if self._cell_occupied_by_player(curr_coord):
                    raise IllegalActionException(
                        f"Jump {move_action.coord} {move_action.directions} "
                        "is blocked.", self._turn_color)
                
            except ValueError:
                raise IllegalActionException(
                    f"Move {move_action.coord} {move_action.directions} "
                    "is prohibited.", self._turn_color)
                
        return curr_coord

    def _validate_move_action(self, action: MoveAction):
        if type(action) != MoveAction:
            raise IllegalActionException(
                f"Action '{action}' is not a MOVE action object.", 
                    self._turn_color)
        
        self._assert_has_attr(action, "coord")
        self._assert_has_attr(action, "directions")

        self._assert_coord_valid(action.coord)
        self._assert_coord_occ_by(action.coord, self._turn_color)

        if type(action.directions) != tuple and \
              type(action.directions) != list:
            raise IllegalActionException(
                f"Action '{action}' has invalid directions object.", 
                    self._turn_color)

        if len(action.directions) == 0:
            raise IllegalActionException(
                f"Action '{action}' has no direction(s).", 
                    self._turn_color)

        for direction in action.directions:
            self._assert_direction_valid(direction)
            self._assert_direction_legal(direction, self._turn_color)

        dest_coord = self._resolve_move_destination(action)
        
        if self._state[dest_coord].state != "LilyPad":
            raise IllegalActionException(
                f"Move {action.coord} {action.directions} "
                "is prohibited.", self._turn_color)

    def _resolve_move_action(self, action: MoveAction) -> BoardMutation:
        self._validate_move_action(action)

        # Ensure action directions object is immutable
        action = MoveAction(action.coord, tuple(action.directions))

        from_coord = action.coord
        dest_coord = self._resolve_move_destination(action)

        cell_mutations = {
            from_coord: CellMutation(
                from_coord,
                self._state[from_coord],
                CellState(None)
            ),
            dest_coord: CellMutation(
                dest_coord,
                self._state[dest_coord],
                self._state[from_coord]
            )
        }

        return BoardMutation(
            action,
            cell_mutations=set(cell_mutations.values())
        )
    
    def _resolve_grow_action(self, action: GrowAction) -> BoardMutation:
        cell_mutations = {}

        player_cells = set(
            coord for coord, cell in self._state.items()
            if cell.state == self._turn_color
        )

        neighbour_cells = set()
        for cell in player_cells:
            for direction in Direction:
                try:
                    neighbour = cell + direction
                    neighbour_cells.add(neighbour)
                except ValueError:
                    pass

        for cell in neighbour_cells:
            if self._cell_empty(cell):
                cell_mutations[cell] = CellMutation(
                    cell,
                    self._state[cell],
                    CellState('LilyPad')
                )

        return BoardMutation(
            action,
            cell_mutations=set(cell_mutations.values())
        )
    
    def set_cell_state(self, cell: Coord, state: CellState):
        self._state[cell] = state

    def set_turn_color(self, color: PlayerColor):
        self._turn_color = color
