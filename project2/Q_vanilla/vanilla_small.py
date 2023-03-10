from pyexpat import model
from statistics import mode
import numpy as np
import csv

# GLOBAL VARIABLES
N_ACTIONS = 4
N_STATES = 10 * 10
DISCOUNT = 0.95
MAX_ITERATIONS = 100

class QLearning:
    def __init__(self, n_states, n_actions, discount, learning_rate):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros((n_states, n_actions))


    def update(self, s, a, r, s_prime):
        Q_s_prime_max = np.max(self.Q[s_prime, :])
        Q_s_a = self.Q[s, a]
        update = self.learning_rate * (r + self.discount * Q_s_prime_max - Q_s_a)
        self.Q[s, a] +=  update

def data_batch(model):
    with open('data/small.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        i = 0
        for row in reader:
            # if i == 4: break
            # i += 1
            s, a, r, s_prime = [int(num) for num in row]
            model.update(s-1, a-1, r, s_prime-1)

def main():
    model = QLearning(n_states=500*100, n_actions=7, discount=1, learning_rate=0.001)

    for i in range(MAX_ITERATIONS):
        data_batch(model)

    with open('policies/small.policy', 'w') as f:
        for state, row in enumerate(model.Q):
            action = np.argmax(row) + 1
            f.write(str(action) + '\n')

if __name__ == "__main__":
    main()
