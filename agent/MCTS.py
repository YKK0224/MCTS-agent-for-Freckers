import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import math
from referee.game.player import PlayerColor
from referee.game import board
from referee.game.board import Board
from referee.game.coord import Coord, Direction
from referee.game.actions import Action, MoveAction, GrowAction
from referee.game.exceptions import IllegalActionException
from referee.game.constants import *
import random
from collections import defaultdict
from copy import deepcopy

class Node:
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent                    #parent node
        self.children = []                      #children nodes
        self.total_utility = [0.0, 0.0]         #Utility, left fot red and right for blue
        self.num_visit = 0                      #Number of visits of a node
        self.possible_moves = state.get_moves() #possible moves that have not tried
    
    def select(self):
        C = math.sqrt(2)
        if self.state.board.turn_color == PlayerColor.RED:
            current_player = 0                  #current player is red frog
        else:
            current_player = 1                  #current player is blue frog
            
        # Function that calculates the upper confidence bound
        def UCB(child) -> float:
            #Prioritise the node that has not been visited yet
            if child.num_visit == 0:
                return float('inf')
            
            exploit = child.total_utility[current_player] / child.num_visit
            explore = C * math.sqrt(math.log(self.num_visit) / child.num_visit)
            return exploit + explore
        #Select the child that has the highest UCB
        return max(self.children, key=UCB)  #可以修改，不使用max
    
    #Expand a new node
    def expand(self):
        #Find a leagal move and move to that node
        move = self.possible_moves.pop()
        next_state = self.state.move(move)
        child = Node(next_state, parent=self)
        self.children.append(child)
        return child
    
    #Update the utility of nodes in MCTS
    def update_utility(self, utility):
        self.num_visit+=1
        if self.state.board.turn_color == PlayerColor.RED:
            self.total_utility[0] += utility
            self.total_utility[1] -= utility
        else:
            self.total_utility[1] += utility
            self.total_utility[0] -= utility
        
class MCTS:
    def __init__(self, state):
        self.root = Node(state)
        self.depth = 30
        
    def search(self, iteration = 3000):
        for i in range(iteration):
            node = self.select()
            utility = self.simulate(node)
            self.back_prop(node, utility)

            #print(i + 1, "round complete")
        #可修改不使用max和lambda
        return max(self.root.children, key=lambda c: c.num_visit).state.last_move
    
    def select(self):
        curr = self.root
        while not curr.state.is_terminal():
            if curr.possible_moves:
                return curr.expand()
            else:
                curr = curr.select()
        return curr
    
    
    def simulate(self, node):
        #可修改
        curr_state = deepcopy(node.state)
        
        while not curr_state.is_terminal() and self.depth > 0:
            possible_moves = curr_state.get_moves()
            if not possible_moves:
                break
            move = random.choice(list(possible_moves))
            curr_state = curr_state.move(move)
            self.depth -= 1
        return curr_state.get_reward()
    
    def back_prop(self, node, utility):
        while node:
            node.update_utility(utility)
            node = node.parent
        
    
class GameState:
    def __init__(self, last_move, board):
        self.last_move = last_move
        self.board = board
        
    def get_moves(self):
        
        red_dir = [
            Direction.Down,
            Direction.DownLeft,
            Direction.DownRight,
            Direction.Left,
            Direction.Right,
        ]
        
        blue_dir = [
            Direction.Up,
            Direction.UpLeft,
            Direction.UpRight,
            Direction.Left,
            Direction.Right,
        ]
        #可能还有左和右
        all_dir = [
            Direction.Down,
            Direction.DownLeft,
            Direction.DownRight,
            Direction.Up,
            Direction.UpLeft,
            Direction.UpRight,
        ]
        
        visited = set()
        grow_pads = set()
        moves = defaultdict(set)
        def evaluate(move):
            score = 1
            if isinstance(move, MoveAction):
                start_coord = move.coord
                end_coord = move.coord
                dirs = move.directions
                
                for d in dirs:
                    end_coord += d
                    
                score = abs(end_coord.r - start_coord.r)    #垂直距离评估，可以考虑别的
            moves[score].add(move)
                
                
        #Implement an single move
        def simple_move(coord, dirs):
            for d in dirs:
                #Ingnore the case that move to out of bound
                try:
                    next_coord = coord + d
                    if self.board[next_coord].state == "LilyPad":
                        move = MoveAction(coord, (d,))
                        evaluate(move)
                except ValueError:
                    continue
                
        #Implement jump moves
        def chain_jump(curr_coord, dirs, path: list[Direction]):
            if self.board[curr_coord].state != self.board.turn_color:
                return
            if path:
                move = MoveAction(curr_coord, tuple(path.copy()))
                evaluate(move)
            
            for d in dirs:
                try:
                    
                    mid_coord = curr_coord + d
                    final_coord = mid_coord + d
                    #可改
                    if self.board[mid_coord].state in (PlayerColor.BLUE, PlayerColor.RED) and self.board[final_coord].state == "LilyPad":
                        new_path = path + [d]
                        if final_coord not in visited:
                            visited.add(final_coord)
                            chain_jump(final_coord, dirs, new_path)
                            visited.remove(final_coord)
                            
                except ValueError:
                    continue
                
        def grow(coord):
            for d in all_dir:
                try:
                    next_coord = coord + d
                    if self.board[next_coord].state == None:
                        grow_pads.add(next_coord)
                    
                except ValueError:
                    continue
                
            if grow_pads:
                move = GrowAction()
                evaluate(move)
                
        if self.board.turn_color == PlayerColor.RED:
            red_coords = []
            #简化
            for Coord, CellState in self.board._state.items():
                if CellState.state == PlayerColor.RED:
                    red_coords.append(Coord)
                    
            for coord in red_coords:
                simple_move(coord, red_dir)
                chain_jump(coord, red_dir, [])
                grow(coord)
        elif self.board.turn_color == PlayerColor.BLUE:
            blue_coords = []
            #简化
            for Coord, CellState in self.board._state.items():
                if CellState.state == PlayerColor.BLUE:
                    blue_coords.append(Coord)
            for coord in blue_coords:
                simple_move(coord, blue_dir)
                chain_jump(coord, blue_dir, [])
                grow(coord)
                
        if moves:
            best_move = moves[max(moves)]
        else:
            best_move = None
            
        return best_move
    def move(self, action):
        new_board = deepcopy(self.board)
        #可以修改
        try:
            mutation = new_board.apply_action(action)
        except IllegalActionException as e:
            raise ValueError(f"Illegal Action: {e}")
        return GameState(last_move=action, board=new_board)
    
    def is_terminal(self):
        return self.board.game_over
    
    #可修改
    def get_reward(self):
        #需要补充comment和改名字
        def heuristic():
            red_progress = self.board._row_count(PlayerColor.RED, BOARD_N - 1) / (BOARD_N - 2)
            blue_progress = self.board._row_count(PlayerColor.BLUE, 0) / (BOARD_N - 2)
            score = red_progress - blue_progress
            
            if self.board.turn_color == PlayerColor.BLUE:
                score *= -1
            return score
            
        if self.is_terminal():
            red_score = self.board._player_score(PlayerColor.RED)
            blue_score = self.board._player_score(PlayerColor.BLUE)
            
            if self.board.turn_color == PlayerColor.RED:
                if red_score > blue_score:
                    return 1.0
                elif blue_score > red_score:
                    return -1.0
                else:
                    return 0
            else:
                if blue_score > red_score:
                    return 1.0
                elif red_score > blue_score:
                    return -1.0
                else:
                    return 0
                
        else:
            return heuristic()
        
if __name__ == "__main__":
    board = Board()
    initial_state = GameState(None, board)
    
    mcts = MCTS(initial_state)
    best_action = mcts.search()
    print("The best action: ", best_action)