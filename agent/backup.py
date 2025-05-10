def get_moves(self) -> List[Action]:
        player = self.board.turn_color
        sign = 1 if player == PlayerColor.RED else -1
        offsets = [(1, 0), (1, -1), (1, 1)]
        forward_dirs = [DELTA_TO_DIR[(dr*sign, dc*sign)] for dr,dc in offsets]

        weights: List[Tuple[int, Action]] = []
        goal_row = BOARD_N-1 if player==PlayerColor.RED else 0

        # if no forward pads but empty forward => grow
        
        has_move = False 
        has_empty = False
        '''
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
        '''
        
        # generate moves
        for coord, cell in self.board._state.items():
            if cell.state!=player or coord.r==goal_row:
                continue
            # one-step
            for d in forward_dirs:
                try:
                    tgt = coord + d
                    if self.board[tgt].state=='LilyPad':
                        has_move = True
                        w = 100 if tgt.r == goal_row else abs(tgt.r - coord.r)
                        weights.append((w, MoveAction(coord, (d,))))
                    elif self.board[tgt].state is None:
                        has_empty = True
                except ValueError:
                    pass
            # jumps via DFS
            def dfs(src:Coord, path:Tuple[Direction,...], visited:set):
                for d in forward_dirs:
                    try:
                        mid = src + d; tgt = mid + d
                        if (self.board[mid].state in (PlayerColor.RED,PlayerColor.BLUE)) and self.board[tgt].state=='LilyPad' and tgt not in visited:
                            new_path = path+(d,)
                            w = 100 if tgt.r == goal_row else len(new_path)
                            weights.append((w, MoveAction(coord,new_path)))
                            visited.add(tgt)
                            dfs(tgt,new_path,visited)
                            visited.remove(tgt)
                    except ValueError:
                        pass
            dfs(coord, (), set())
        '''
        # optional grow
        if has_empty:
            weights.append((1, GrowAction()))
        '''
        if not has_move and has_empty:
            weights.append((1, GrowAction()))

        weights.sort(key=lambda x: x[0], reverse=True)
        return [a for _,a in weights]