# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.exceptions import IllegalActionException
from referee.game.board import Board
from .MCTS5 import GameState, MCTS

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        
        self._color = color
        self._board: Board | None = None
        self.current_state: GameState | None = None
        self.mcts: MCTS | None = None
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        if self._board is None:
            #from referee.game.board import Board
            self._board = Board()

        if self.current_state is None:
            self.current_state = GameState(last_move=None, board=self._board)
       
       # Initialize MCTS on first turn, otherwise reroot into the existing tree
        if self.mcts is None:
            self.mcts = MCTS(self.current_state)
        else:
            self.mcts.reroot(self.current_state)

        # Run MCTS with a fixed iteration budget (200 rollouts)
        best_move = self.mcts.search(200, time_limit=180)
        return best_move
    
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        if self._board is None:
            self._board = Board()

        try:
            self._board.apply_action(action)
        except IllegalActionException as e:
            print(f"IllegalMove: {e}")

         # Advance our GameState and reroot the MCTS tree
        self.current_state = GameState(last_move=action, board=self._board)
        if self.mcts:
            self.mcts.reroot(self.current_state)

        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
            case GrowAction():
                print(f"Testing: {color} played GROW action")
            case _:
                raise ValueError(f"Unknown action type: {action}")