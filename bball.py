import numpy as np


class Shot():
    def __init__(self):
        # constants
        self.bball_radius = 9.5 / 12 / 2  # ft
        self.height_basket = 10  # ft
        self.g = -32.174  # ft/s^2

    def setup(self):
        # setup cannon parameters: distance, height, speed
        min_params = [5.0, 0.0, 34.3]
        max_params = [25.0, 10.0, 68.6]
        dist, height, speed = np.random.uniform(min_params, max_params)
        self.dist = dist  # ft, distance to hoop
        self.height = height  # ft, height of the cannon
        self.speed = speed  # ft/s

        self.delta_h = self.height_basket - self.height
        self.rim = dist + self.ball_radius * np.array([-1, 1])  # m

        self.path = []
        self.angle = 0
        self.result = 'no shot'

        return dist, height, speed
