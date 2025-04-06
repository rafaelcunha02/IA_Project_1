class Simulation:
    def __init__(self, reds, score, aligned_reds, aligned_blues, game=None, special=True, valid_moves=None):
        self.reds = reds
        self.score = score
        self.aligned_reds = aligned_reds
        self.aligned_blues = aligned_blues
        self.game = game
        self.valid_moves = valid_moves
        self.special = special

        self._hash = self._compute_hash()

    def _compute_hash(self):
        if self.special:
            blocks_hash = hash(tuple(tuple(tuple(row) for row in block.shape) for block in self.game.blocks)) if self.game and hasattr(self.game, 'blocks') else 0
            return hash((self.reds, self.score, blocks_hash))
        else:
            if self.game and hasattr(self.game, 'grid'):
                grid_hash = hash(tuple(tuple(row) for row in self.game.grid))
                blocks_hash = hash(tuple(tuple(tuple(row) for row in block.shape) for block in self.game.blocks)) if self.game and hasattr(self.game, 'blocks') else 0
                return hash((grid_hash, self.reds, self.score, blocks_hash))
            return 0  
    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self.special:
            if (self.reds, self.score, self.aligned_reds, self.aligned_blues) != \
               (other.reds, other.score, other.aligned_reds, other.aligned_blues):
                return False

            if not (self.game and other.game and hasattr(self.game, 'blocks') and hasattr(other.game, 'blocks')):
                return False

            self_blocks = {tuple(map(tuple, block.shape)) for block in self.game.blocks}
            other_blocks = {tuple(map(tuple, block.shape)) for block in other.game.blocks}

            return self_blocks == other_blocks
        else:
            return hash(self) == hash(other)

    def __str__(self):
        return f"Simulation(reds={self.reds}, score={self.score}, aligned_reds={self.aligned_reds}, aligned_blues={self.aligned_blues}, cost={self.cost})"

    def __lt__(self, other):
        return self.reds < other.reds