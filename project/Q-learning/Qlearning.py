#-------------------------------------------poo   
import traci
import numpy as np
import random as rd
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
        """Calcule dynamiquement les seuils de densité en fonction de l'heure de la journée."""
        if time_of_day == 'morning':  # Matin - heures de pointe
            return {
                'bretelle_low': 3,   
                'bretelle_medium': 6,  
                'autoroute_low': 25,  
            }
        elif time_of_day == 'midday':  # Midi - période calme
            return {
                'bretelle_low': 2,   
                'bretelle_medium': 4,  
                'autoroute_low': 15,  
            }
        elif time_of_day == 'evening':  # Soirée - fin de journée
            return {
                'bretelle_low': 3,   
                'bretelle_medium': 5,  
                'autoroute_low': 20,  
            }
        else:  # Période de nuit
            return {
                'bretelle_low': 1,   
                'bretelle_medium': 2,  
                'autoroute_low': 5,   
            }

    def get_state(self):
        """Récupère l'état actuel basé sur la densité de trafic et la période de la journée."""
        density_autoroute = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
        self.maxAutoroute = max(density_autoroute, self.maxAutoroute)

        density_bretelle = traci.lane.getLastStepVehicleNumber("E2_0")
        self.maxBretelle = max(density_bretelle, self.maxBretelle)

        # Détermine l'heure de la journée (matin, midi, soir)
        current_time = traci.simulation.getTime()
        if 0 <= current_time < 1800:
            time_of_day = 'morning'
        elif 1800 <= current_time < 3600:
            time_of_day = 'midday'
        else:
            time_of_day = 'evening'

        thresholds = self.calculate_thresholds(time_of_day)
        return self.categorize_state(density_bretelle, density_autoroute, thresholds)

    def choose_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)  # Exploration
        return np.argmax(self.Q[state, :])  # Exploitation

    def get_reward(self, density_autoroute, queue_bretelle):
        """Calculates reward based on traffic conditions, including collisions."""
        throughput = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))  # Highway flow
        
        # Récupérer le nombre de collisions
        collisions = traci.simulation.getCollisions()
        num_collisions = len(collisions)   
        emergency_brakes = 0
        for vehicle_id in traci.vehicle.getIDList():
            if traci.vehicle.getEmergencyDecel(vehicle_id) > 0:  
                emergency_brakes += 1
        # Récompense qui prend en compte la densité de trafic, les files d'attente, et les collisions
        reward = -(density_autoroute + queue_bretelle) + 0.1 * throughput - 10 * num_collisions - 5 * emergency_brakes

        return reward

    def train(self, epoch):
        traci.start([
            "sumo-gui",
            "-c",
            "../simulation/sumo.sumocfg",
            "--start",
            "true",
            "--xml-validation",
            "never",
            "--log",
            "log",
            "--quit-on-end"
        ])
        print(f"Sumo started for epoch num: {epoch}")
        logging.info(f"Starting epoch {epoch}")

        for _ in range(10):
            traci.simulation.step()

        state = self.get_state()
        done = False
        while not done:
            action = self.choose_action(state)

            # Set traffic light state dynamically
            try:
                signal_states = ["GGGr", "GGGy", "GGGG"]
                traci.trafficlight.setRedYellowGreenState("feux", signal_states[action])
            except Exception as e:
                logging.error(f"Error setting traffic light state: {e}")

            traci.simulation.step()

            density_autoroute = sum(traci.lane.getLastStepVehicleNumber(f"E0_{i}") for i in range(3))
            queue_bretelle = traci.lane.getLastStepVehicleNumber("E2_0")
            reward = self.get_reward(density_autoroute, queue_bretelle)

            next_state = self.get_state()
            self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])

            state = next_state
            done = (traci.simulation.getMinExpectedNumber() == 0)

        print(f'epsilon: {self.epsilon}')
        print(f"Q-table: {self.Q}")
        traci.close()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# Simulation

if __name__ == '__main__':
    QL_traffic_control = QLearningTrafficControl()
    for epoch in range(1):
        QL_traffic_control.train(epoch)
