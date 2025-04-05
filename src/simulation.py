class Simulation:
    def __init__(self, reds, score, aligned_reds, aligned_blues, game=None, special=False, cost=0, valid_moves=None):
        self.reds = reds
        self.score = score
        self.aligned_reds = aligned_reds
        self.aligned_blues = aligned_blues
        self.game = game
        self.cost = cost
        self.valid_moves = valid_moves
        self.special = special

        # Precompute the hash for the grid
        self._hash = self._compute_hash()

    def _compute_hash(self):
        # Create a hash based only on the grid
        if self.game and hasattr(self.game, 'grid'):
            # Convert the grid (2D list) into a tuple of tuples for hashing
            return hash(tuple(tuple(row) for row in self.game.grid))
        return 0  # Default hash if the grid is not available

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self.special:
            print("special")
            # Use the custom logic when `special` is True
            if (self.reds, self.score, self.aligned_reds, self.aligned_blues) != \
               (other.reds, other.score, other.aligned_reds, other.aligned_blues):
                return False

            # Ensure both have game states with blocks
            if not (self.game and other.game and hasattr(self.game, 'blocks') and hasattr(other.game, 'blocks')):
                return False

            # Compare the available blocks as sets (order doesn't matter)
            self_blocks = {tuple(map(tuple, block.shape)) for block in self.game.blocks}
            other_blocks = {tuple(map(tuple, block.shape)) for block in other.game.blocks}

            # Two states are only equal if they have the same blocks available
            return self_blocks == other_blocks
        else:
            # Compare hashes only when `special` is False
            return hash(self) == hash(other)

    def __str__(self):
        return f"Simulation(reds={self.reds}, score={self.score}, aligned_reds={self.aligned_reds}, aligned_blues={self.aligned_blues}, cost={self.cost})"

    def __lt__(self, other):
        return self.cost < other.cost