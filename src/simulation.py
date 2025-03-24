class Simulation:
    def __init__(self, reds, score, aligned_reds, aligned_blues, game=None, cost=0, valid_moves=None):
        self.reds = reds
        self.score = score
        self.aligned_reds = aligned_reds
        self.aligned_blues = aligned_blues
        self.game = game
        self.cost = cost
        self.valid_moves = valid_moves

    def __hash__(self):
        # Create a hash that includes which blocks are available
        if self.game and hasattr(self.game, 'blocks'):
            block_shapes = frozenset(tuple(map(tuple, block.shape)) for block in self.game.blocks)
            return hash((self.reds, self.score, self.aligned_reds, self.aligned_blues, block_shapes))
        return hash((self.reds, self.score, self.aligned_reds, self.aligned_blues))

    def __eq__(self, other):
        # First compare basic metrics
        if (self.reds, self.score, self.aligned_reds, self.aligned_blues) != \
           (other.reds, other.score, other.aligned_reds, other.aligned_blues):
            return False
        
        # Then ensure both have game states with blocks
        if not (self.game and other.game and hasattr(self.game, 'blocks') and hasattr(other.game, 'blocks')):
            return False
            
        # Compare the available blocks as sets (order doesn't matter)
        self_blocks = {tuple(map(tuple, block.shape)) for block in self.game.blocks}
        other_blocks = {tuple(map(tuple, block.shape)) for block in other.game.blocks}
        
        # Two states are only equal if they have the same blocks available
        return self_blocks == other_blocks
    
    def __str__(self):
        return f"Simulation(reds={self.reds}, score={self.score}, aligned_reds={self.aligned_reds}, aligned_blues={self.aligned_blues}, cost={self.cost})"
    
    def __lt__(self, other):
        return self.cost < other.cost