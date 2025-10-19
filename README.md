# Deep-Reinforcement-Learning-based-autonomous-driving-simulation-using-Highway-Env-and-DQN-algorithm


## Features

- Highway environment simulation with multiple vehicles
- Rule-based autonomous driving agent
- Deep Q-Network (DQN) reinforcement learning agent
- Performance comparison and visualization
- Video recording of autonomous driving sessions


## Classes

### AutonomousVehicleSimulator
Main simulator class with methods for training, testing, and visualization.

### RuleBasedDriver
Simple rule-based autonomous driving agent using predefined logic.

### EnvironmentConfig
Configuration parameters for highway environment settings.

## Action Space

- 0: LANE_LEFT
- 1: IDLE
- 2: LANE_RIGHT
- 3: FASTER
- 4: SLOWER

## Observation Space

5x5 matrix: [presence, x_position, y_position, x_velocity, y_velocity]



