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
        offsets = [(1, 0), (1, -1), (1, 1)]
        forward_dirs = [DELTA_TO_DIR[(dr*sign, dc*sign)] for dr,dc in offsets]

        weights: List[Tuple[int, Action]] = []
        goal_row = BOARD_N-1 if player==PlayerColor.RED else 0

        # if no forward pads but empty forward => grow
        has_pad = has_empty = False
        for coord, cell in self.board._state.items():
            if cell.state==player:
                for d in forward_dirs:
                    try:
                        s = self.board[coord+d].state
                        if s=='LilyPad': has_pad=True
                        elif s is None: has_empty=True
                    except ValueError:
                        pass
        if not has_pad and has_empty:
            return [GrowAction()]

        # generate moves
        for coord, cell in self.board._state.items():
            if cell.state!=player or coord.r==goal_row:
                continue
            # one-step
            for d in forward_dirs:
                try:
                    tgt = coord + d
                    if self.board[tgt].state=='LilyPad':
                        w = abs(tgt.r - coord.r)
                        weights.append((w, MoveAction(coord, (d,))))
                except ValueError:
                    pass
            # jumps
            def dfs(src:Coord, path:Tuple[Direction,...], visited:set):
                for d in forward_dirs:
                    try:
                        mid = src + d; tgt = mid + d
                        if self.board[mid].state in (PlayerColor.RED,PlayerColor.BLUE) and self.board[tgt].state=='LilyPad' and tgt not in visited:
                            new_path = path+(d,)
                            weights.append((len(new_path), MoveAction(coord,new_path)))
                            visited.add(tgt)
                            dfs(tgt,new_path,visited)
                            visited.remove(tgt)
                    except ValueError:
                        pass
            dfs(coord, (), set())

        # optional grow
        if has_empty:
            weights.append((1, GrowAction()))

        weights.sort(key=lambda x: x[0], reverse=True)
        return [a for _,a in weights]

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
        if self.is_terminal():
            r=self.board._player_score(PlayerColor.RED)
            b=self.board._player_score(PlayerColor.BLUE)
            if r>b: return 1.0
            if b>r: return -1.0
            return 0.0
        red_prog=self.board._row_count(PlayerColor.RED,BOARD_N-1)/(BOARD_N-2)
        blue_prog=self.board._row_count(PlayerColor.BLUE,0)/(BOARD_N-2)
        diff = red_prog-blue_prog
        return diff if self.board.turn_color==PlayerColor.RED else -diff

class Node:
    """
    Node in MCTS tree.
    """
    def __init__(self,state:GameState,parent:Optional['Node']=None):
        self.state=state 
        self.parent=parent
        self.children:List['Node']=[]
        self.visits=0; 
        self.value=[0.0, 0.0]
        self.untried_moves=state.get_moves()

    def ucb1(self)->float:
        if self.visits==0: 
            return float('inf')
        curr_play = 0
        return (self.value/self.visits) + EXPLORATION_CONST*math.sqrt(math.log(self.parent.visits)/self.visits)

    def select(self)->'Node':
        if not self.children:
            return self
        return max(self.children, key=lambda c:c.ucb1())

    def expand(self)->'Node':
        if not self.untried_moves: 
            return self
        mv=self.untried_moves.pop(0)
        child=Node(self.state.move(mv), parent=self)
        self.children.append(child)
        return child

    def update(self,reward:float):
        self.visits+=1; 
        if self.state.board.turn_color == PlayerColor.RED:
            self.total_utility[0] += reward
            self.total_utility[1] -= reward
        else:
            self.total_utility[1] += reward
            self.total_utility[0] -= reward

class MCTS:
    """
    Monte Carlo Tree Search.
    """
    def __init__(self,root_state:GameState):
        self.root=Node(root_state)

    def search(self, iteration: Optional[int] = None, iterations: int = 200, time_limit: Optional[float] = None) -> Optional[Action]:  # 固定使用50次模拟，使红蓝对称且加快速度
        start=time.time(); total=iterations
        for _ in range(total):
            if time_limit and (time.time()-start)>time_limit: break
            node=self._select(); reward=self._simulate(node); self._backpropagate(node,reward)
        if not self.root.children: return None
        best=max(self.root.children, key=lambda c:c.visits)
        return best.state.last_move

    def _select(self)->Node:
        node=self.root
        while not node.state.is_terminal():
            if node.untried_moves: return node.expand()
            node=node.select()
        return node

    def _simulate(self,node:Node)->float:
        board=node.state.board; state=node.state; depth=SIM_DEPTH; history=[]
        while not board.game_over and depth>0:
            moves=state.get_moves()
            if not moves: break
            mv=random.choice(moves)
            history.append(board.apply_action(mv)); state=GameState(mv, board);
            depth-=1
        reward=state.get_utility()
        for _ in history: board.undo_action()
        return reward

    def _backpropagate(self,node:Node,reward:float):
        while node:
            node.update(reward); node=node.parent

    def reroot(self,new_state:GameState):
        for child in self.root.children:
            if child.state.board==new_state.board:
                child.parent=None; self.root=child; return
        self.root=Node(new_state)
