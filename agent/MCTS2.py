
# agent/MCTS.py
import math
import random
import time
from copy import deepcopy
from typing import List, Optional, Tuple

from referee.game.player import PlayerColor
from referee.game.board import Board
from referee.game.coord import Coord, Direction
from referee.game.actions import MoveAction, GrowAction, Action
from referee.game.exceptions import IllegalActionException
from referee.game.constants import BOARD_N

# Simulation parameters
SIM_DEPTH = 15
EXPLORATION_CONSTANT = math.sqrt(2)

class GameState:
    """
    Wraps a Board for MCTS: provides move generation, successor, terminal check, and utility.
    """
    def __init__(self, last_move: Optional[Action], board: Board):
        self.last_move = last_move
        self.board = board

    def get_moves(self) -> List[Action]:
        """
        Generate all legal MoveAction and GrowAction for current player.
        Returns a list sorted by heuristic weight (longer forward moves first).
        """
        moves: List[Action] = []
        weights: List[Tuple[int, Action]] = []
        player = self.board.turn_color
        goal_row = 0 if player == PlayerColor.BLUE else BOARD_N - 1

        # Directions for movement and jump
        if player == PlayerColor.RED:
            dirs = [Direction.Down, Direction.DownLeft, Direction.DownRight, Direction.Left, Direction.Right]
        else:
            dirs = [Direction.Up, Direction.UpLeft, Direction.UpRight, Direction.Left, Direction.Right]

        # All neighbors for grow
        grow_dirs = list(Direction)

        # Simple moves and jumps
        for coord, cell in self.board._state.items():
            if cell.state == player and coord.r != goal_row:
                # Simple one-step moves
                for d in dirs:
                    try:
                        nxt = coord + d
                        if self.board[nxt].state == "LilyPad":
                            move = MoveAction(coord, (d,))
                            weight = abs(nxt.r - coord.r)
                            weights.append((weight, move))
                    except (ValueError, KeyError):
                        continue
                # Chain jumps via DFS
                def dfs(c: Coord, path: Tuple[Direction, ...], visited: set):
                    for d in dirs:
                        try:
                            mid = c + d
                            dest = mid + d
                            if (self.board[mid].state in (PlayerColor.RED, PlayerColor.BLUE)) and self.board[dest].state == "LilyPad" and dest not in visited:
                                new_path = path + (d,)
                                move = MoveAction(coord, new_path)
                                weight = len(new_path)
                                weights.append((weight, move))
                                visited.add(dest)
                                dfs(dest, new_path, visited)
                                visited.remove(dest)
                        except (ValueError, KeyError):
                            continue
                dfs(coord, (), set())

        # Grow action
        can_grow = False
        for coord, cell in self.board._state.items():
            if cell.state == player:
                for d in grow_dirs:
                    try:
                        tgt = coord + d
                        if self.board[tgt].state is None:
                            can_grow = True
                            break
                    except (ValueError, KeyError):
                        continue
                if can_grow:
                    break
        if can_grow:
            weights.append((1, GrowAction()))

        # Sort by descending weight
        weights.sort(key=lambda x: x[0], reverse=True)
        moves = [m for _, m in weights]
        return moves

    def move(self, action: Action) -> 'GameState':
        """
        Returns a new GameState after applying the action (deep copy board).
        """
        new_board = deepcopy(self.board)
        try:
            new_board.apply_action(action)
        except IllegalActionException:
            pass
        return GameState(action, new_board)

    def is_terminal(self) -> bool:
        return self.board.game_over

    def get_utility(self) -> float:
        """
        +1 for win, -1 for loss, or refined heuristic combining progress, piece count, mobility, and center control.
        """
        # Terminal outcome
        if self.is_terminal():
            red_score = self.board._player_score(PlayerColor.RED)
            blue_score = self.board._player_score(PlayerColor.BLUE)
            if red_score > blue_score:
                return 1.0
            if blue_score > red_score:
                return -1.0
            return 0.0

        # 1. Progress towards goal
        red_prog = self.board._row_count(PlayerColor.RED, BOARD_N - 1) / (BOARD_N - 2)
        blue_prog = self.board._row_count(PlayerColor.BLUE, 0) / (BOARD_N - 2)
        prog_diff = red_prog - blue_prog

        # 2. Piece count difference
        red_pcs = sum(1 for cell in self.board._state.values() if cell.state == PlayerColor.RED)
        blue_pcs = sum(1 for cell in self.board._state.values() if cell.state == PlayerColor.BLUE)
        piece_diff = (red_pcs - blue_pcs) / (BOARD_N * BOARD_N)

        # 3. Mobility difference
        my_moves = len(self.get_moves())
        # Opponent moves: simulate turn change
        opp_color = PlayerColor.BLUE if self.board.turn_color == PlayerColor.RED else PlayerColor.RED
        # Temporarily switch turn for mobility count
        original = self.board._turn_color
        self.board._turn_color = opp_color
        opp_moves = len(self.get_moves())
        self.board._turn_color = original
        mobility_diff = (my_moves - opp_moves) / max(my_moves + opp_moves, 1)

        # 4. Center control: count frogs on central region
        mid = BOARD_N // 2
        center_coords = [Coord(r, c) for r in range(mid-1, mid+2) for c in range(mid-1, mid+2)]
        center_red = sum(1 for coord in center_coords if self.board._state.get(coord, None) and self.board._state[coord].state == PlayerColor.RED)
        center_blue = sum(1 for coord in center_coords if self.board._state.get(coord, None) and self.board._state[coord].state == PlayerColor.BLUE)
        center_diff = (center_red - center_blue) / 9

        # Combine with weights
        score = (0.4 * prog_diff
                 + 0.3 * piece_diff
                 + 0.2 * mobility_diff
                 + 0.1 * center_diff)

        # Return from current player's perspective
        return score if self.board.turn_color == PlayerColor.RED else -score

class Node:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(self, state: GameState, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: List['Node'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = state.get_moves()

    def ucb_score(self) -> float:
        if self.visits == 0:
            return float('inf')
        # Exploitation + Exploration (UCB1)
        return (self.value / self.visits) + EXPLORATION_CONSTANT * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def select(self) -> 'Node':
        # Select child with highest UCB score
        return max(self.children, key=lambda c: c.ucb_score())

    def expand(self) -> 'Node':
        # Expand this node by creating a new child from an untried move
        if not self.untried_moves:
            return self
        move = self.untried_moves.pop(0)
        next_state = self.state.move(move)
        child = Node(next_state, parent=self)
        self.children.append(child)
        return child

    def update(self, reward: float):
        # Update visit count and total value
        self.visits += 1
        self.value += reward

class MCTS:
    """
    Monte Carlo Tree Search algorithm for Freckers.
    """
    def __init__(self, root_state: GameState):
        self.root = Node(root_state)

    def search(self, iteration: int = None, iterations: int = 500, time_limit: float = None) -> Optional[Action]:
        """
        Run MCTS: either `iteration` fixed playouts or up to `iterations`,
        optionally bounded by `time_limit` seconds.
        """
        start = time.time()
        total = iteration if iteration is not None else iterations
        for _ in range(total):
            if time_limit and (time.time() - start) > time_limit:
                break
            node = self._select()
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        if not self.root.children:
            return None
        best = max(self.root.children, key=lambda c: c.visits)
        return best.state.last_move

    def _select(self) -> Node:
        node = self.root
        while not node.state.is_terminal():
            if node.untried_moves:
                return node.expand()
            node = node.select()
        return node

    def _simulate(self, node: Node) -> float:
        board = node.state.board
        depth = SIM_DEPTH
        history = []
        state = node.state
        while not board.game_over and depth > 0:
            moves = state.get_moves()
            if not moves:
                break
            move = random.choice(moves)
            mutation = board.apply_action(move)
            history.append(mutation)
            state = GameState(move, board)
            depth -= 1
        reward = state.get_utility()
        for _ in history:
            board.undo_action()
        return reward

    def _backpropagate(self, node: Node, reward: float):
        while node:
            node.update(reward)
            node = node.parent

    def reroot(self, new_state: GameState):
        for child in self.root.children:
            if child.state.board == new_state.board:
                child.parent = None
                self.root = child
                return
        self.root = Node(new_state)

