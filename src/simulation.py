class Simulation:
    def __init__(self, reds, score, aligned_reds, aligned_blues, cost = 0):
        self.reds = reds
        self.score = score
        self.aligned_reds = aligned_reds
        self.aligned_blues = aligned_blues
        self.cost = cost

    def __hash__(self):
        return hash((self.reds, self.score, self.aligned_reds, self.aligned_blues, self.cost))

    def __eq__(self, other):
        return (self.reds, self.score, self.aligned_reds, self.aligned_blues) == \
               (other.reds, other.score, other.aligned_reds, other.aligned_blues)