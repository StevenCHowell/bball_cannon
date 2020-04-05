import numpy as np
import bball


def generate_samples(n_samples=20000):
    table = np.empty((n_samples, 4))
    for i in range(n_samples):
        shot = bball.Shot()
        table[i, 1:] = shot.setup()
        table[i, 0] = bball.AgentPoly(shot).learn(10)
    return table


if __name__ == '__main__':
    n_samples = 50_000_000
    table = generate_samples(n_samples)
    np.savetxt(f'./{n_samples}_shot_angles.csv', table, delimiter=',')
