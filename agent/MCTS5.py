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

# MCTS simulation parameters
SIM_DEPTH = 15
EXPLORATION_CONST = math.sqrt(2)

# Map delta to Direction
DELTA_TO_DIR = {
    (1, 0): Direction.Down,
    (1, -1): Direction.DownLeft,
    (1, 1): Direction.DownRight,
    (-1, 0): Direction.Up,
    (-1, -1): Direction.UpLeft,
    (-1, 1): Direction.UpRight,
    (0, 1): Direction.Right,
    (0, -1): Direction.Left,
}

class GameState:
    """
    Wraps a Board: provides legal moves, successor, terminal check, utility.
    """
    def __init__(self, last_move: Optional[Action], board: Board):
        self.last_move = last_move
        self.board = board

    def get_moves(self) -> List[Action]:
        player = self.board.turn_color
        sign = 1 if player == PlayerColor.RED else -1
        forward_offsets = [(1, 0), (1, -1), (1, 1)]
        forward_dirs    = [DELTA_TO_DIR[(dr*sign, dc*sign)] for dr,dc in forward_offsets]
        lateral_offsets = [(0, 1), (0, -1)]
        lateral_dirs    = [DELTA_TO_DIR[off] for off in lateral_offsets]

        weights: List[Tuple[int, Action]] = []
        goal_row = BOARD_N - 1 if player == PlayerColor.RED else 0
        has_empty_global = False
        
        for coord, cell in self.board._state.items():
            if cell.state != player or coord.r == goal_row:
                continue
           # One-step forward moves
            local_forward = False
            for d in forward_dirs:
                try:
                    tgt = coord + d
                    s = self.board[tgt].state
                    if s == 'LilyPad':
                        local_forward = True
                        weight = 100 if tgt.r == goal_row else abs(tgt.r - coord.r)
                        weights.append((weight, MoveAction(coord, (d,))))
                    elif s is None:
                        has_empty_global = True
                except ValueError:
                    pass

            # Chain jumps
            local_jump = False
            def dfs(src: Coord, path: Tuple[Direction, ...], visited: set):
                nonlocal local_jump
                for d in forward_dirs:
                    try:
                        mid = src + d
                        tgt = mid + d
                        if (self.board[mid].state in (PlayerColor.RED, PlayerColor.BLUE)
                            and self.board[tgt].state == 'LilyPad' and tgt not in visited):
                            new_path = path + (d,)
                            w = 100 if tgt.r == goal_row else len(new_path)
                            weights.append((w, MoveAction(coord, new_path)))
                            visited.add(tgt)
                            local_jump = True
                            dfs(tgt, new_path, visited)
                            visited.remove(tgt)
                    except ValueError:
                        pass
            dfs(coord, (), set())

            # Only add Grow if no move or jump available
            if not local_forward and not local_jump:
                allow_lateral = (
                    (player == PlayerColor.BLUE and coord.r <= 2)
                    or (player == PlayerColor.RED and coord.r >= BOARD_N - 3))
                if allow_lateral:
                    for d in lateral_dirs:
                        try:
                            tgt = coord + d
                            if self.board[tgt].state == "LilyPad":
                                w = abs(tgt.c - coord.c)  # lateral weight = 1
                                weights.append((w, MoveAction(coord, (d,))))
                            elif self.board[tgt].state is None:
                                has_empty_global = True
                        except ValueError:
                            pass
        if has_empty_global:
            weights.append((0.5, GrowAction()))

        weights.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in weights]

    def move(self, action: Action) -> 'GameState':
        new_board = deepcopy(self.board)
        try:
            new_board.apply_action(action)
        except IllegalActionException:
            pass
        return GameState(action, new_board)

    def is_terminal(self) -> bool:
        return self.board.game_over

    def get_utility(self) -> float:       
        if self.is_terminal:
            #Find end game utility
            red_score = self.board._player_score(PlayerColor.RED)
            blue_score = self.board._player_score(PlayerColor.BLUE)
            
            if self.board.turn_color == PlayerColor.RED:
                if red_score > blue_score:
                    return 1.0
                elif red_score < blue_score:
                    return -1.0
                else:
                    return 0
            elif self.board.turn_color == PlayerColor.BLUE:
                if blue_score > red_score:
                    return 1.0
                elif blue_score < red_score:
                    return -1.0
                else:
                    return 0   
        else:
            #If the game is not terminated, calculate utility for non-terminal position
            #Count the number of frog on goal state, and divided by the number of frog to normalise the value
            red_progress = self.board._row_count(PlayerColor.RED, BOARD_N - 1) / (BOARD_N - 2)
            blue_progress = self.board._row_count(PlayerColor.BLUE, 0) / (BOARD_N - 2)
            
            utility = red_progress - blue_progress
            
            #Invert the sign when blue, so higher is always better.
            if self.board.turn_color == PlayerColor.BLUE:
                utility *= -1
            return utility

class Node:
    """
    Node in MCTS tree.
    """
    def __init__(self,state:GameState,parent:Optional['Node']=None):
        self.state=state; self.parent=parent
        self.children:List['Node']=[]
        self.visits=0 
        self.value=[0.0, 0.0]
        self.untried_moves=state.get_moves()

    def ucb1(self)->float:
        if self.state.board.turn_color == PlayerColor.RED:
            curr_player = 0
        else:
            curr_player = 1
        if self.visits==0: return float('inf')
        return (self.value[curr_player]/self.visits) + EXPLORATION_CONST*math.sqrt(math.log(self.parent.visits)/self.visits)

    def select(self)->'Node':
        return max(self.children, key=lambda c:c.ucb1())

    def expand(self)->'Node':
        if not self.untried_moves: return self
        mv=self.untried_moves.pop(0)
        child=Node(self.state.move(mv), parent=self)
        self.children.append(child)
        return child

    def update(self,reward:float):
        self.visits+=1
        #self.value+=reward
        if self.state.board.turn_color == PlayerColor.RED:
            self.value[0] += reward
            self.value[1] -= reward
        else:
            self.value[1] += reward
            self.value[0] -= reward

class MCTS:
    """
    Monte Carlo Tree Search.
    """
    def __init__(self,root_state:GameState):
        self.root=Node(root_state)

    def search(self, iterations, time_limit: Optional[float] = None) -> Optional[Action]:
        start=time.time()
        total=iterations
        for _ in range(total):
            if time_limit and (time.time()-start)>time_limit: break
            node=self._select() 
            reward=self._simulate(node) 
            self._backpropagate(node,reward)
        if not self.root.children:
            # Fallback if no simulations generated any children
            moves = self.root.state.get_moves()
            if moves:
                return random.choice(moves)
            else:
                return GrowAction()
        best=max(self.root.children, key=lambda c:c.visits)
        return best.state.last_move

    def _select(self)->Node:
        node=self.root
        while True:
            if node.untried_moves:
                return node.expand()
            if not node.children or node.state.is_terminal():
                return node
            node=node.select()

    def _simulate(self,node:Node)->float:
        board=node.state.board; state=node.state; depth=SIM_DEPTH; history=[]
        while not board.game_over and depth>0:
            moves=state.get_moves()
            if not moves: break
            mv=random.choice(moves)
            history.append(board.apply_action(mv)); state=GameState(mv, board); depth-=1
        reward=state.get_utility()
        for _ in history: board.undo_action()
        return reward

    def _backpropagate(self,node:Node,reward:float):
        while node:
            node.update(reward)
            node=node.parent

    def reroot(self,new_state:GameState):
        for child in self.root.children:
            if child.state.board==new_state.board:
                child.parent=None; self.root=child; return
        self.root=Node(new_state)
