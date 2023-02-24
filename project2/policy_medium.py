import numpy as np
import csv


def scale_gradient(gradient, L2_max):
    norm_gradient = np.linalg.norm(gradient)
    scale_factor = min(L2_max / norm_gradient, 1)
    return scale_factor * gradient


class GradientQLearning:
    def __init__(self, action_space, discount, Q, gradient_Q, theta, learning_rate):
        self.action_space = action_space # action space (assumes 1:nactions)
        self.discount = discount # discount
        self.Q = Q # parameterized action value function Q(theta, s, a)
        self.gradient_Q = gradient_Q
        self.theta = theta # action value function parameter
        self.learning_rate = learning_rate # learning rate


    def update(self, s, a, r, s_prime):
        u = max([self.Q(self.theta, s_prime, a_prime) for a_prime in self.action_space])
        delta = (r + self.discount * u - self.Q(self.theta, s, a)) * self.gradient_Q(self.theta, s, a)
        self.theta += self.learning_rate * delta
        return self




# @dataclass
# class GradientQLearning:
#     action_space: list[int]
#     discount: float
#     Q: callable
#     grad_Q: callable
#     theta: np.ndarray
#     learning_rate: float

def beta(s, a):
    return np.array([s, s**2, a, a**2, 1])

def Q(theta, s, a):
    return np.dot(theta, beta(s, a))

def grad_Q(theta, s, a):
    return beta(s, a)


def data_batch(model):
    with open('data/medium.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        i = 0
        for row in reader:
            # if i == 4: break
            # i += 1
            s, a, r, s_prime = [int(num) for num in row]
            model.update(s-1, a-1, r, s_prime-1)


def main():
    theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    learning_rate = 0.5
    action_space = 50,000
    discount = 1
    model = GradientQLearning(action_space, discount, Q, grad_Q, theta, learning_rate)


    for i in range(100):
        data_batch(model)

    with open('policies/medium.policy', 'w') as f:
        # for state, row in enumerate(model.Q):
        #     action = np.argmax(row) + 1
        #     f.write(str(action) + '\n')




if __name__ == "__main__":
    main()
