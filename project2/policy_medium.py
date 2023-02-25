import numpy as np
from tqdm import tqdm
import csv

# GLOBAL VARIABLES
NUM_ITERATIONS = 100
NUM_POSSIBLE_POSITIONS = 500
NUM_POSSIBLE_VELOCITIES = 100


def scale_gradient(gradient, L2_max):
    norm_gradient = np.linalg.norm(gradient)
    scale_factor = min(L2_max / norm_gradient, 1)
    return scale_factor * gradient


class GradientQLearning:
    def __init__(self, action_space, discount, Q, gradient_Q, theta, learning_rate):
        self.action_space = action_space
        self.discount = discount
        self.Q = Q
        self.gradient_Q = gradient_Q
        self.theta = theta
        self.learning_rate = learning_rate


    def update(self, s, a, r, s_prime):
        u = max([self.Q(self.theta, s_prime, a_prime) for a_prime in self.action_space])
        delta = (r + self.discount * u - self.Q(self.theta, s, a)) * self.gradient_Q(self.theta, s, a)
        # print("scale: ", scale_gradient(delta, 1))
        self.theta += (self.learning_rate * scale_gradient(delta, 1))



def beta(s, a):
    # print(np.array([s[0], s[1], s[0]+s[1], a, a**2, 1]))
    p, v = s
    return np.array([1, \
                    p, v, a,\
                    p^2, p*v, v^2, a*p, a*v, \
                    p^3, p^2*v, p*v^2, v^3, a^2*p, a^2*v,\
                    p^4, p^3*v, p^2*v^2, p*v^3, v^4, a^2*p^2, a^2*v^2,\
                    p^5, p^4*v, p^3*v^2, p^2*v^3, p*v^4, v^5, a^3*p^3, a^3*v^3,\
                    p^6, p^5*v, p^4*v^2, p^3*v^3, p^2*v^4, p*v^5, v^6, a^4*p^4, a^4*v^4])

def Q(theta, s, a):
    return np.dot(theta, beta(s, a))


#
# IS THE GRADIENT ALWAYS beta regardless of the basis function we use
#
def grad_Q(theta, s, a):
    return beta(s, a)


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
            # print("s and s_prime: ", s, s_prime)
            model.update(s, a, r, s_prime)


def main():
    theta = np.random.rand(39)
    learning_rate = 0.001
    action_space = [i for i in range(1, 8)]
    discount = 1
    # print(action_space)
    ln = linear_index()
    model = GradientQLearning(action_space, discount, Q, grad_Q, theta, learning_rate)


    for i in tqdm(range(NUM_ITERATIONS)):
        data_batch(model, ln)

    actions = [i for i in range(1, 8)]
    states = ln.get_states() # STATES ARE OFF BY ONE

    # print(states.shape)


    with open('policies/medium.policy', 'w') as f:
        i = 0
        for state in states:
            
            # DEBUGGING
            # if i < 10:
            #     print([model.Q(theta, state, action) for action in actions])
            #     i += 1


            
            action = np.argmax(np.array([model.Q(theta, tuple(state), action) for action in actions])) + 1 # TO CORRECT THE OFF BY ONE
            f.write(str(action) + '\n')




if __name__ == "__main__":
    main()
