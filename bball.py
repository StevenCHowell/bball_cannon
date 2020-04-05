import itertools
import numpy as np
from bokeh import plotting
from bokeh.resources import INLINE
from bokeh.palettes import Dark2_8 as palette

colors = itertools.cycle(palette)
# plotting.output_notebook(resources=INLINE)
# plotting.output_html(resources=INLINE)

class Shot():
    def __init__(self):
        # constants
        self.ball_radius = 9.5 / 12 / 2  # ft
        self.height_basket = 10  # ft
        self.g = -32.174  # ft/s^2
        self.path = []

    def setup(self):
        # randomly select cannon parameters: distance, height, speed
        min_params = [5.0, 0.0, 34.3]
        max_params = [25.0, 10.0, 68.6]
        dist, height, speed = np.random.uniform(min_params, max_params)
        self.dist = dist  # ft, distance to hoop
        self.height = height  # ft, height of the cannon
        self.speed = speed  # ft/s

        self.delta_h = self.height_basket - self.height
        self.rim = dist + self.ball_radius * np.array([-1, 1])  # m

        self.angle = 0

        return np.array([dist, height, speed])

    def shoot_reward(self, angle):
        shoot_delta_x(angle)
        if self.rim[0] < self.x_shot < self.rim[1]:
            # Big reward for success.  Not sure how big to make this.
            reward = 500.0
        else:
            # Penalty for missing equal to distance under/over shot.
            reward = -abs(self.delta_x)

        return reward

    def shoot_delta_x(self, angle):
        self.angle = np.deg2rad(angle)
        self.delta_x = self.dist
        if 0 < angle or angle < 90:
            cos = np.cos(self.angle)
            sin = np.sin(self.angle)

            # calculate the determinant
            D = (self.speed * sin) ** 2 + 2 * self.g * self.delta_h
            if D > 0:  # ball passes through correct height twice
                # negative root is the downward part of arc
                self.t_m = (-self.speed * sin - np.sqrt(D)) / self.g
                self.x_shot = self.speed * cos * self.t_m
                self.delta_x = self.x_shot - self.dist  # +/-: over/under-shot
            else:
                # ball not shot high enough
                self.t_m = 1.3  # s
        return self.delta_x

    def calc_path(self):
        if self.angle:
            cos = np.cos(self.angle)
            sin = np.sin(self.angle)
            t_final = self.t_m * 1.1
            t = np.linspace(0, t_final, 100)
            x = self.speed * cos * t
            y = self.height + self.speed * sin * t + self.g * t**2 / 2
            self.path.append(np.c_[x, y][y>=0])
        else:
            print('Error: no attempts made!')

    def show_path(self):
        if self.path:
            p = plotting.figure(
                width=800, height=400,
                x_axis_label='(ft)', y_axis_label='(ft)',
                x_range=(-1, self.dist*1.3), y_range=(-0.5, 20),
            )

            # plot the hoop
            p.line(np.array([1, 1.015, 1.015]) * self.rim[1],
                   [self.height_basket, self.height_basket, 0], color='gray')
            p.circle(self.rim, self.height_basket,
                     radius=0.02, color='orange')
            p.line(self.rim, self.height_basket, color='orange')

            # plot the attempts
            for i, (path, color) in enumerate(zip(self.path, colors)):
                name = f'Attempt {i+1}'
                p.line(path[:, 0], path[:, 1], color=color,
                       legend_label=name, muted_alpha=0.2)
                t = np.linspace(0, 1, 100)

            # plot the last cannon
            p.line(0, [0, self.height], color='black')
            p.line(
                [-np.cos(self.angle)/2, 0],
                [self.height - np.sin(self.angle)/2, self.height],
                line_width=5, color=color,
            )

            p.legend.location = 'top_right'
            p.legend.click_policy = 'mute'
            p.toolbar.autohide = True
            plotting.show(p)
        else:
            print('Error: no attempts made!')


class AgentPoly():
    def __init__(self, shot):
        self.shot = shot

    def learn(self, max_iters, *, angle_0=89, step_size=10):
        p_or_m = np.sign(45 - angle_0)
        angles = [angle_0, angle_0 + (p_or_m * step_size)]
        delta_x = []

        # fire first two shots
        for angle in angles:
            delta_x.append(self.shot.shoot_delta_x(angle))

        # first first order polynomial
        coef = np.polyfit(angles, delta_x, 1)
        guess = -coef[1] / coef[0]

        # fire estimated best guess as third shot
        angles.append(guess)
        delta_x.append(self.shot.shoot_delta_x(guess))

        success_angle = np.nan
        for i in range(max_iters-3):
            if np.abs(delta_x[-1]) < self.shot.ball_radius:
                success_angle = angles[-1]
                break
            else:
                # fit second order polynomial
                coef = np.polyfit(angles, delta_x, 2)
                guess = np.roots(coef).max()

                # fire estimated best guess
                angles.append(guess)
                delta_x.append(self.shot.shoot_delta_x(guess))

        self.results = np.c_[angles, delta_x]
        return success_angle