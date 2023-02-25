from pyexpat import model
from statistics import mode
import numpy as np
import csv

# GLOBAL VARIABLES
NUM_ITERATIONS = 1

NUM_ACTIONS = 7
NUM_POSSIBLE_POSITIONS = 500
NUM_POSSIBLE_VELOCITIES = 100

class QLearning:
    def __init__(self, n_actions, discount, learning_rate):
        self.n_actions = n_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros((NUM_POSSIBLE_POSITIONS, NUM_POSSIBLE_VELOCITIES, NUM_ACTIONS))


    def update(self, s, a, r, s_prime):
        # print(s, a)
        Q_s_prime_max = np.max(self.Q[s_prime, :])
        Q_s_a = self.Q[s, a]
        update = self.learning_rate * (r + self.discount * Q_s_prime_max - Q_s_a)
        # if update > 0: print(update)
        self.Q[s, a] +=  update



class linear_index():
    def __init__(self) -> None:
        positions = np.arange(NUM_POSSIBLE_POSITIONS)
        velocities = np.arange(NUM_POSSIBLE_VELOCITIES)
        pos,vel = np.meshgrid(positions,velocities)
        self.positions = pos.flatten()
        self.velocities = vel.flatten()
        self.linear_index = 1+pos+500*vel  

    def get_pos(self, li):
        idx = np.where(self.linear_index==li)[0]
        return self.positions[idx][0]

    def get_vel(self, li):
        idx = np.where(self.linear_index==li)[0]
        return self.velocities[idx][0]

    def get_states(self):
        positions, velocities, = np.arange(NUM_POSSIBLE_POSITIONS), np.arange(NUM_POSSIBLE_VELOCITIES)
        aa, bb = np.meshgrid(positions, velocities)
        combinations = np.vstack((aa.ravel(), bb.ravel())).T
        return combinations





def data_batch(model, ln):
    with open('data/medium.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        i = 0
        for row in reader:
            # if i == 4: break
            # i += 1
            s_indx, a, r, s_prime_indx = [int(num) for num in row]
            s = (ln.get_pos(s_indx), ln.get_vel(s_indx))
            s_prime = (ln.get_pos(s_prime_indx), ln.get_vel(s_prime_indx))
            model.update(s, a, r, s_prime)






def main():

    model = QLearning(n_actions=7, discount=1, learning_rate=0.001)
    ln = linear_index()
    for i in range(NUM_ITERATIONS):
        data_batch(model, ln)

    # print(model.Q)

    with open('policies/medium.policy', 'w') as f:
        for state, row in enumerate(model.Q):
            action = np.argmax(row) + 1
            f.write(str(action) + '\n')

if __name__ == "__main__":
    main()
