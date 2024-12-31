import traci
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import logging

class QLearningTrafficControl:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3, epsilon_decay=0.99, min_epsilon=0.1, num_states=6, num_actions=3):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration vs exploitation
        self.epsilon_decay = epsilon_decay  # Decay for epsilon
        self.min_epsilon = min_epsilon  # Minimum epsilon for exploration
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros((num_states, num_actions))  # Q-table for state-action values
        self.maxAutoroute = 0
        self.maxBretelle = 0

    def categorize_state(self, density_bretelle, density_autoroute, thresholds):
        """Categorizes state dynamically based on thresholds."""
        if density_bretelle <= thresholds['bretelle_low'] and density_autoroute <= thresholds['autoroute_low']:
            return 0
        elif density_bretelle <= thresholds['bretelle_low'] and density_autoroute > thresholds['autoroute_low']:
            return 1
        elif thresholds['bretelle_low'] < density_bretelle <= thresholds['bretelle_medium'] and density_autoroute <= thresholds['autoroute_low']:
            return 2
        elif thresholds['bretelle_low'] < density_bretelle <= thresholds['bretelle_medium'] and density_autoroute > thresholds['autoroute_low']:
            return 3
        elif density_bretelle > thresholds['bretelle_medium'] and density_autoroute <= thresholds['autoroute_low']:
            return 4
        else:
            return 5

    def calculate_thresholds(self, time_of_day):
        """Calculates thresholds dynamically based on the time of day."""
        if time_of_day == 'morning':  # Morning - peak hours
            return {'bretelle_low': 3, 'bretelle_medium': 6, 'autoroute_low': 25}
        elif time_of_day == 'midday':  # Midday - calm period
            return {'bretelle_low': 2, 'bretelle_medium': 4, 'autoroute_low': 15}
        elif time_of_day == 'evening':  # Evening - rush hour
            return {'bretelle_low': 3, 'bretelle_medium': 5, 'autoroute_low': 20}
        else:  # Night period
            return {'bretelle_low': 1, 'bretelle_medium': 2, 'autoroute_low': 5}

    def get_state(self):
        """Gets the current state based on traffic density and time of day."""
        density_autoroute = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
        self.maxAutoroute = max(density_autoroute, self.maxAutoroute)

        density_bretelle = traci.lane.getLastStepVehicleNumber("E2_0")
        self.maxBretelle = max(density_bretelle, self.maxBretelle)

        current_time = traci.simulation.getTime()
        if 0 <= current_time < 1800:
            time_of_day = 'morning'
        elif 1800 <= current_time < 3600:
            time_of_day = 'midday'
        else:
            time_of_day = 'evening'

        thresholds = self.calculate_thresholds(time_of_day)
        return self.categorize_state(density_bretelle, density_autoroute, thresholds)

    def choose_action(self, state, evaluate=False):
        """Selects an action using an epsilon-greedy policy. If evaluate=True, use pure exploitation."""
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Exploration
        return np.argmax(self.Q[state, :])  # Exploitation

    def get_reward(self, density_autoroute, queue_bretelle, ramp_metering_time):
        """Calculates reward based on traffic conditions, including ramp metering time."""
        throughput = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))  # Highway flow
        collisions = len(traci.simulation.getCollisions())
        emergency_brakes = sum(1 for vehicle_id in traci.vehicle.getIDList() if traci.vehicle.getEmergencyDecel(vehicle_id) > 0)

        return -(density_autoroute + queue_bretelle) + 0.1 * throughput - 10 * collisions - 5 * emergency_brakes - 0.05 * ramp_metering_time

    def train(self, epoch):
        traci.start([
            "sumo-gui",
            "-c",
            "../Simulation-with-Trafic-Light/sumo.sumocfg",
            "--start",
            "true",
            "--xml-validation",
            "never",
            "--log",
            "log",
            "--quit-on-end"
        ])
        print(f"Sumo started for epoch num: {epoch}")

        for _ in range(10):
            traci.simulation.step()

        state = self.get_state()
        total_reward = 0
        done = False
        while not done:
            action = self.choose_action(state)

            signal_states = ["GGGr", "GGGy", "GGGG"]
            traci.trafficlight.setRedYellowGreenState("feux", signal_states[action])

            traci.simulation.step()
            ramp_metering_time = traci.simulation.getTime()
            density_autoroute = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
            queue_bretelle = traci.lane.getLastStepVehicleNumber("E2_0")
            reward = self.get_reward(density_autoroute, queue_bretelle, ramp_metering_time)
            total_reward += reward

            next_state = self.get_state()
            self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])
            state = next_state
            done = (traci.simulation.getMinExpectedNumber() == 0)

        traci.close()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return total_reward
   
    def convergance(self, previous_Q, threshold):
        """
        Checks if the Q-table has converged by comparing the maximum absolute difference
        between the current and previous Q-tables against a given threshold.

        Args:
            previous_Q (np.array): The Q-table from the previous iteration.
            threshold (float): The convergence threshold for Q-value changes.

        Returns:
            bool: True if the Q-values have converged, False otherwise.
        """
        delta = np.abs(self.Q - previous_Q).max()
        print(f"Convergence Check: Max Q-value change = {delta}")
        return delta < threshold



    def test_performance(self):
        """Tests the learned policy and evaluates performance."""
        traci.start([
            "sumo-gui",
            "-c",
            "../Simulation-with-Trafic-Light/sumo.sumocfg",
            "--start",
            "true",
            "--xml-validation",
            "never",
            "--log",
            "log",
            "--quit-on-end"
        ])

        print("Testing the policy...")

        for _ in range(10):
            traci.simulation.step()

        state = self.get_state()
        total_reward = 0
        ramp_metering_time = 0
        done = False

        while not done:
            action = self.choose_action(state, evaluate=True)

            signal_states = ["GGGr", "GGGy", "GGGG"]
            traci.trafficlight.setRedYellowGreenState("feux", signal_states[action])

            traci.simulation.step()
            ramp_metering_time += 1
            density_autoroute = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
            queue_bretelle = traci.lane.getLastStepVehicleNumber("E2_0")
            reward = self.get_reward(density_autoroute, queue_bretelle, ramp_metering_time)
            total_reward += reward

            next_state = self.get_state()
            state = next_state
            done = (traci.simulation.getMinExpectedNumber() == 0)

        traci.close()
        print(f"Final performance: Total reward = {total_reward}, Total ramp metering time = {ramp_metering_time}")
        return total_reward, ramp_metering_time

# Simulation and Analysis
if __name__ == '__main__':
    QL_traffic_control = QLearningTrafficControl()
    num_epochs = 4
    rewards = []

    previous_Q = np.copy(QL_traffic_control.Q)
    threshold = 1e-3  # Convergence threshold

    for epoch in range(num_epochs):
        total_reward = QL_traffic_control.train(epoch)
        rewards.append(total_reward)

        if QL_traffic_control.convergance(previous_Q, threshold):
            print("Converged!")
            break
        previous_Q = np.copy(QL_traffic_control.Q)

        # Plot Rewards
        plt.plot(rewards)
        plt.xlabel("Epoch")
        plt.ylabel("Total Reward")
        plt.title("Reward Trend Over Time")
        plt.show()

    # Test Policy
    print("Testing learned policy...")
    QL_traffic_control.test_performance()
