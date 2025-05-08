import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import math
from typing import List
from referee.game.player import PlayerColor
from referee.game import board
from referee.game.board import Board
from referee.game.coord import Direction
from referee.game.actions import MoveAction, GrowAction, Action
from referee.game.exceptions import IllegalActionException
from referee.game.constants import *
import random
from collections import defaultdict
from copy import deepcopy

class TreeNode:
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent                    #parent node
        self.children = []                      #children nodes
        self.total_utility = [0.0, 0.0]         #Utility, left fot red and right for blue
        self.num_visit = 0                      #Number of visits of a node
        self.moves = state.get_moves()          #possible moves that have not tried
    
    def select_child(self):
        #Select the child that has the highest UCB
        C = math.sqrt(2)     
        best = None
        best_UCB = -float('inf')
        for child in self.children:
            #Prioritise the node that has not been visited yet
            if child.num_visit == 0:
                UCB = float('inf')
            else:
                if self.state.board.turn_color == PlayerColor.RED:
                    curr_player = 0                  #current player is red frog
                else:
                    curr_player = 1                  #current player is blue frog
                exploit = child.total_utility[curr_player] / child.num_visit
                explore = C * math.sqrt(math.log(self.num_visit) / child.num_visit)
                UCB = exploit + explore
                
            if UCB > best_UCB:
                best_UCB = UCB
                best = child
        return best 
    
    #Expand a new node
    def expand(self):
        #Find a leagal move and move to that node
        #sorted_moves = sorted(self.moves, key=lambda m: (len(m.directions)if isinstance(m, MoveAction) else 0), reverse=True)
        move = self.moves.pop()
        #move = sorted_moves[0]
        next_state = self.state.move(move)
        child = TreeNode(next_state, parent=self)
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
   
class GameState:
    def __init__(self, last_move, board):
        self.last_move = last_move
        self.board = board

    #Find all the possible moves of a current frog
    def get_moves(self):
        #Directions allowed for red frog
        red_dir = [Direction.Down, Direction.DownLeft, Direction.DownRight, 
                        Direction.Left, Direction.Right]
        #Directions allowed for blue frog
        blue_dir = [Direction.Up,   Direction.UpLeft,   Direction.UpRight,
                        Direction.Left, Direction.Right]
        #All the possible directions, can be used to generate lilipads
        all_dir = [Direction.Up, Direction.UpLeft, Direction.UpRight,
                        Direction.Down, Direction.DownLeft, Direction.DownRight,
                        Direction.Left, Direction.Right]
        
        grow_pads = set()               #Collect the coordinates for growing a lilypad
        moves = defaultdict(set)        #A dictionary that has weight of moves as key, and corresponding moves as value
        
        #Function that evaluates the weight of a move, and add to a dictionary
        #Dictionary is used to find the move with highest weight, and thus pruning the move that has low weight
        def evaluate(move):
            #要修改逻辑，改成direction最长的优先？
            #The weight for grow action is 1
            weight = 1
            if isinstance(move, MoveAction):
                start_coord = move.coord
                end_coord = move.coord
                #dirs = move.directions
                for d in move.directions:
                    end_coord += d
                    #If current move is a jump move, jump to desired coordinate
                    '''
                    if(self.board[end_coord].state == PlayerColor.BLUE or 
                            self.board[end_coord].state == PlayerColor.RED):
                        end_coord += d
                    '''
                #Determine the weight of move action by calculating vertical distance
                weight = abs(end_coord.r - start_coord.r)
                
            moves[weight].add(move)
                 
        #Implement an single move
        def simple_move(coord, dirs):
            for d in dirs:
                #Ingnore the case that move to out of bound
                try:
                    nxt = coord + d
                    if self.board[nxt].state == "LilyPad":
                        move = MoveAction(coord, (d,))
                        evaluate(move)
                except ValueError:
                    continue
        #Track all the visited coordinate for dfs
        visited = set()
        
        #Implement jump moves
        def chain_jump(curr_coord, dirs, path):
            #Ignore the case that current coordinate does not have the desired frog
            if self.board[curr_coord].state != self.board.turn_color:
                return
            #Record and evaluate each move
            if path:
                move = MoveAction(curr_coord, tuple(path.copy()))
                evaluate(move)
            
            for d in dirs:
                try:
                    #Find the middle coordinate and final coordinate
                    mid = curr_coord + d
                    #Ignore the case that jump comdition is not satisfied
                    if self.board[mid].state != PlayerColor.BLUE and self.board[mid].state != PlayerColor.RED:
                        continue
                    fin = mid + d          
                    if self.board[fin].state == "LilyPad":
                        new_path = path + (d,)
                        #Using dfs to find chain jumps
                        if fin not in visited:
                            visited.add(fin)
                            chain_jump(fin, dirs, new_path)
                            visited.remove(fin)
                            
                except ValueError:
                    continue 
            
        '''
        def jump(start, curr, dirs, path):
            if self.board[curr].state != self.board.turn_color:
                return
            if path:
                move = MoveAction(curr, tuple(path.copy()))
                evaluate(move)
            for d in dirs:
                try:
                    mid = curr + d
                    fin = mid + d
                    if (
                        self.board[mid].state in (PlayerColor.RED, PlayerColor.BLUE)
                        and self.board[fin].state == "LilyPad"
                        and fin not in visited
                    ):
                        visited.add(fin)
                        jump(start, fin, dirs, path + (d,))
                        visited.remove(fin)
                except ValueError:
                    pass
        '''
        
        
        def grow(coord):
            #Checking surrounding positions to make sure that lily pad can be grown
            for d in all_dir:
                try:
                    next_coord = coord + d
                    if self.board[next_coord].state == None:
                        grow_pads.add(next_coord)
                    
                except ValueError:
                    continue
            #If there are valid positions to grow, add the grow action to dict   
            if grow_pads:
                evaluate(GrowAction())
                
        #Red's turn 
        if self.board.turn_color == PlayerColor.RED:
            #List for recording red frog coordinates
            red_coords = []
            
            for Coord, CellState in self.board._state.items():
                if CellState.state == PlayerColor.RED:
                    red_coords.append(Coord)
            #For each red frog, evaluate three moves, and add the weight to dict     
            for coord in red_coords:
                simple_move(coord, red_dir)
                chain_jump(coord, red_dir, ())
                #jump(coord, coord, red_dir, ())
                grow(coord)
        #Blue's turn
        elif self.board.turn_color == PlayerColor.BLUE:
            blue_coords = []
 
            for Coord, CellState in self.board._state.items():
                if CellState.state == PlayerColor.BLUE:
                    blue_coords.append(Coord)
                    
            for coord in blue_coords:
                simple_move(coord, blue_dir)
                chain_jump(coord, blue_dir, ())
                #jump(coord, coord, blue_dir, ())
                grow(coord)
        if moves:
            best_move = moves[max(moves)]
        else:
            best_move = None
  
        return best_move
    
    #Apply moves to update the board
    def move(self, action):
        new_board = deepcopy(self.board)
        #可以修改
        try:
            mutation = new_board.apply_action(action)
        except IllegalActionException:
            pass
        return GameState(last_move=action, board=new_board)
    
    def is_terminal(self):
        return self.board.game_over
    
    #Find the utility of a current state
    def get_utility(self):       
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
        
        
class MCTS:
    def __init__(self, state):
        self.root = TreeNode(state)
        self.depth = 15             #depth of simulation
        
    #A function that do the MCTS search
    def search(self, cycles):
        #cycles represent the number of selection, simulation, back propogation performed
        for i in range(cycles):
            #Select a child node to expand
            child = self.select()
            #Simulate and back propagate the utility to the tree
            utility = self.simulate(child)
            self.back_prop(child, utility)
            
        best_child = None
        best_visit = -1
        #Loop through all the child nodes, find the one with highest number of visits
        #and return its associated move
        for child in self.root.children:
            if child.num_visit > best_visit:
                best_visit = child.num_visit
                best_child = child
        return best_child.state.last_move
    
    # A function that selects a child node to expand
    def select(self):
        curr = self.root
        while not curr.state.is_terminal():
            #If there are still moves can go, expand current node
            if curr.moves:
                return curr.expand()
            #If no more moves can go, seelct the child with highest UCB
            else:
                curr = curr.select_child()
        return curr
    
    #A function that do a random move and finds the utility
    def simulate(self, node):
        #可修改
        curr_state = node.state
        depth_limit = self.depth            #Number of moves to go
        while not curr_state.is_terminal() and depth_limit > 0:
            moves = curr_state.get_moves()
            if not moves:
                break
            move = random.choice(list(moves))
            curr_state = curr_state.move(move)
            depth_limit -= 1
        return curr_state.get_utility()
    
    #Back propagate the number of visits and utility
    def back_prop(self, node, utility):
        while node:
            node.update_utility(utility)
            node = node.parent
        
 
        
if __name__ == "__main__":
    board = Board()
    initial_state = GameState(None, board)
    
    mcts = MCTS(initial_state)
    best_action = mcts.search()
    print("The best action: ", best_action)