# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from dataclasses import dataclass

from .coord import Coord, Direction


@dataclass(frozen=True, slots=True)
class MoveAction():
    """
    A dataclass representing a "move action", which consists of a coordinate 
    and one or more directions (multiple directions used for multiple hops).
    """
    coord: Coord
    _directions: Direction | tuple[Direction]

    @property
    def directions(self) -> tuple[Direction]:
        if isinstance(self._directions, Direction):
            return (self._directions,)
        return self._directions
    
    def __str__(self) -> str:
        try:
            dirs_text = ", ".join(str(d) for d in self.directions)
            return f"MOVE({self.coord}, [{dirs_text}])"
        except:
            return f"MOVE(<invalid coord/directions>)"
        

@dataclass(frozen=True, slots=True)
class GrowAction():
    """
    A dataclass representing a "grow action".
    """
    def __str__(self) -> str:
        return "GROW"


Action = MoveAction | GrowAction
