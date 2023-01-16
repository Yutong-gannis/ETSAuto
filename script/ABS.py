class TTC:
    def __init__(self, distance):
        self.distance = distance
        self.history_distance = distance
        self.relative_speed = None
        self.TTC = None
    def update(self, distance, delta_t):
        self.history_distance = self.distance
        self.distance = distance
        self.relative_speed = (self.history_distance - self.distance)/delta_t
        self.TTC = self.distance / self.relative_speed
        return self.TTC

