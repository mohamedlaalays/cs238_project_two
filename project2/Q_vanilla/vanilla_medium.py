from pyexpat import model
from statistics import mode
import numpy as np
import csv

MAX_ITERATIONS = 1

class QLearning:
    def __init__(self, n_states, n_actions, discount, learning_rate):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros((n_states, n_actions))


    def update(self, s, a, r, s_prime):
        # print(s, a)
        Q_s_prime_max = np.max(self.Q[s_prime, :])
        Q_s_a = self.Q[s, a]
        update = self.learning_rate * (r + self.discount * Q_s_prime_max - Q_s_a)
        # if update > 0: print(update)
        self.Q[s, a] +=  update

def data_batch(model):
    with open('../data/medium.csv', newline='') as csvfile:
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

    with open('/policies/medium.policy', 'w') as f:
        for row in model.Q:
            # print(row)
            action = np.argmax(row) + 1
            # print("action", action)
            # if action != 1: print(action)
            f.write(str(action) + '\n')

if __name__ == "__main__":
    main()
