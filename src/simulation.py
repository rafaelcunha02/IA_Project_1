class Simulation:
    def __init__(self, reds, score, aligned_reds, aligned_blues, game = None, cost = 0):
        self.reds = reds
        self.score = score
        self.aligned_reds = aligned_reds
        self.aligned_blues = aligned_blues
        self.game = game
        self.cost = cost

    def __hash__(self):
        return hash((self.reds, self.score, self.aligned_reds, self.aligned_blues, self.cost))

    def __eq__(self, other):
        return (self.reds, self.score, self.aligned_reds, self.aligned_blues) == \
               (other.reds, other.score, other.aligned_reds, other.aligned_blues)
    
    def __str__(self):
        return f"Simulation(reds={self.reds}, score={self.score}, aligned_reds={self.aligned_reds}, aligned_blues={self.aligned_blues}, cost={self.cost})"
    
    def __lt__(self, other):
        return self.cost < other.cost