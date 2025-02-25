import numpy as np
import gym
from sklearn.preprocessing import PolynomialFeatures

class QLearningLinearApprox:
    def __init__(self, state_dim, action_dim, alpha=0.01, gamma=0.99, epsilon=0.1, degree=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Feature transformation (Polynomial Features for non-linearity)
        self.poly = PolynomialFeatures(degree)
        self.feature_dim = self.poly.fit_transform(np.zeros((1, state_dim))).shape[1]

        # Initialize weights randomly
        self.weights = np.random.randn(self.feature_dim, action_dim) * 0.01

    def featurize(self, state):
        """Convert state into feature vector"""
        state = np.array(state).reshape(1, -1)
        return self.poly.fit_transform(state)[0]

    def predict(self, state):
        """Compute Q-values for all actions using linear function approximation"""
        features = self.featurize(state)
        return np.dot(features, self.weights)

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)  # Explore
        q_values = self.predict(state)
        return np.argmax(q_values)  # Exploit

    def update(self, state, action, reward, next_state, done):
        """Q-learning weight update using SGD"""
        features = self.featurize(state)
        q_values = self.predict(state)

        # Compute TD target
        if done:
            target = reward
        else:
            next_q_values = self.predict(next_state)
            target = reward + self.gamma * np.max(next_q_values)

        # Compute TD error
        td_error = target - q_values[action]

        # SGD update
        self.weights[:, action] += self.alpha * td_error * features
if __name__ == '__main__':

    # ---- Train on OpenAI Gym Environment ----
    env = gym.make("CartPole-v1")  # Continuous state space, discrete action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = QLearningLinearApprox(state_dim, action_dim)

    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()[0]  # Gym API returns tuple (state, info)
        total_reward = 0

        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()
