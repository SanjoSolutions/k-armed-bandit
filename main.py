import random

class Bandit:
    def __init__(self, k):
        self.levers = [None] * k
        for i in range(k):
            minimum = random.random()
            maximum = minimum + abs(random.random() - minimum)
            lever = (minimum, maximum)
            self.levers[i] = lever

    def pull_lever(self, lever_index):
        lever = self.levers[lever_index]
        minimum, maximum = lever
        return minimum + random.random() * (maximum - minimum)


def create_k_bandit(k):
    return Bandit(k)


class Model:
    def __init__(self, bandit):
        k = len(bandit.levers)
        self.q = [0] * k
        self.n = [0] * k


def learn(bandit, number_of_learning_steps):
    model = Model(bandit)

    Q = lambda action: model.q[action]

    for learning_step_number in range(number_of_learning_steps):
        actions = tuple(range(len(bandit.levers)))
        epsilon = 0.1
        if random.random() <= epsilon:
            action = random.choice(actions)
        else:
            action = argmax(actions, Q)
        reward = bandit.pull_lever(action)
        model.n[action] += 1
        model.q[action] += 1.0 / model.n[action] * (reward - model.q[action])

    return model


def argmax(values, predicate):
    return max(values, key=predicate)


if __name__ == '__main__':
    bandit = create_k_bandit(10)
    model = learn(bandit, 1000)
    print('bandit:', bandit)
    print('model:', model)
    print('')