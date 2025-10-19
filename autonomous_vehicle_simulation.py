import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import imageio
import warnings
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings('ignore')


class EnvironmentConfig:
    """Configuration parameters for highway environment"""
    
    OBSERVATION_CONFIG = {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
    }
    
    ACTION_CONFIG = {
        "type": "DiscreteMetaAction",
    }
    
    ENV_CONFIG = {
        "lanes_count": 4,
        "vehicles_count": 50,
        "duration": 40,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }


class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress"""
    
    def __init__(self, check_freq, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_count = 0
    
    def _on_step(self):
        return True


class RuleBasedDriver:
    """Rule-based autonomous driving agent"""
    
    def __init__(self):
        self.name = "Rule-Based Driver"
    
    def decide_action(self, observation):
        ego_x = observation[0, 1]
        ego_y = observation[0, 2]
        ego_vx = observation[0, 3]
        
        vehicle_ahead = False
        vehicle_ahead_distance = float('inf')
        
        for i in range(1, len(observation)):
            if observation[i, 0] == 1:
                other_x = observation[i, 1]
                other_y = observation[i, 2]
                
                if other_x > ego_x and abs(other_y - ego_y) < 0.1:
                    vehicle_ahead = True
                    distance = other_x - ego_x
                    if distance < vehicle_ahead_distance:
                        vehicle_ahead_distance = distance
        
        if vehicle_ahead and vehicle_ahead_distance < 0.3:
            if ego_y > 0:
                return 0
            else:
                return 2
        elif vehicle_ahead and vehicle_ahead_distance < 0.5:
            return 1
        elif not vehicle_ahead or vehicle_ahead_distance > 0.5:
            if ego_vx < 0.8:
                return 3
            else:
                return 1
        
        return 1


class AutonomousVehicleSimulator:
    """Main simulator class for autonomous vehicle"""
    
    def __init__(self, env_name="highway-v0", render_mode="rgb_array"):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = None
        self.model = None
        self.config = EnvironmentConfig()
        
    def create_environment(self):
        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        
        config = {
            "observation": self.config.OBSERVATION_CONFIG,
            "action": self.config.ACTION_CONFIG,
        }
        config.update(self.config.ENV_CONFIG)
        
        self.env.unwrapped.config.update(config)
        return self.env
    
    def visualize_environment(self, save_path=None):
        if self.env is None:
            self.create_environment()
        
        observation, info = self.env.reset()
        img = self.env.render()
        
        plt.figure(figsize=(12, 3))
        plt.imshow(img)
        plt.title("Highway Environment")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return observation, info
    
    def test_manual_control(self, actions, save_path=None):
        if self.env is None:
            self.create_environment()
        
        self.env.reset()
        frames = []
        rewards = []
        
        for action in actions:
            observation, reward, terminated, truncated, info = self.env.step(action)
            frames.append(self.env.render())
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        return frames, rewards
    
    def test_rule_based_driver(self, num_steps=30):
        if self.env is None:
            self.create_environment()
        
        driver = RuleBasedDriver()
        observation, info = self.env.reset()
        
        frames = []
        rewards = []
        actions = []
        
        for step in range(num_steps):
            action = driver.decide_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            frames.append(self.env.render())
            rewards.append(reward)
            actions.append(action)
            
            if terminated or truncated:
                break
        
        return frames, rewards, actions
    
    def create_training_environment(self):
        train_env = gym.make(self.env_name, render_mode=self.render_mode)
        
        train_config = {
            "observation": self.config.OBSERVATION_CONFIG,
            "action": self.config.ACTION_CONFIG,
            "lanes_count": 4,
            "vehicles_count": 30,
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 1,
        }
        
        train_env.unwrapped.config.update(train_config)
        train_env = DummyVecEnv([lambda: train_env])
        
        return train_env
    
    def train_dqn_model(self, total_timesteps=20000, model_path=None):
        train_env = self.create_training_environment()
        
        self.model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=0,
        )
        
        callback = TrainingCallback(check_freq=2000)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        if model_path:
            self.model.save(model_path)
        
        return self.model
    
    def load_model(self, model_path):
        self.model = DQN.load(model_path)
        return self.model
    
    def test_trained_model(self, num_episodes=3, max_steps=40):
        if self.env is None:
            self.create_environment()
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        all_frames = []
        all_rewards = []
        all_actions = []
        episode_stats = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            frames = []
            actions = []
            
            for step in range(max_steps):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                frames.append(self.env.render())
                episode_reward += reward
                actions.append(action)
                
                if terminated or truncated:
                    break
            
            all_frames.append(frames)
            all_rewards.append(episode_reward)
            all_actions.append(actions)
            episode_stats.append({
                'reward': episode_reward,
                'steps': len(frames),
                'crashed': info.get('crashed', False)
            })
        
        return all_frames, all_rewards, episode_stats
    
    def create_video(self, num_steps=50, output_path="autonomous_vehicle_demo.mp4", fps=5):
        if self.env is None:
            self.create_environment()
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        obs, info = self.env.reset()
        video_frames = []
        total_reward = 0
        
        for step in range(num_steps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            video_frames.append(self.env.render())
            
            if terminated or truncated:
                break
        
        imageio.mimsave(output_path, video_frames, fps=fps)
        
        return output_path, total_reward, len(video_frames)
    
    def evaluate_driver(self, driver_type, num_episodes=5):
        env = gym.make(self.env_name, render_mode=self.render_mode)
        config = {
            "observation": self.config.OBSERVATION_CONFIG,
            "action": self.config.ACTION_CONFIG,
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40,
        }
        env.unwrapped.config.update(config)
        
        episode_rewards = []
        episode_lengths = []
        crashes = 0
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(40):
                if driver_type == "AI":
                    if self.model is None:
                        raise ValueError("Model not trained or loaded")
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    rule_driver = RuleBasedDriver()
                    action = rule_driver.decide_action(obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if info.get('crashed', False):
                        crashes += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'crash_rate': crashes / num_episodes,
            'all_rewards': episode_rewards
        }
    
    def compare_drivers(self, num_episodes=5):
        ai_stats = self.evaluate_driver("AI", num_episodes)
        rule_stats = self.evaluate_driver("Rule-Based", num_episodes)
        
        return ai_stats, rule_stats
    
    def visualize_comparison(self, ai_stats, rule_stats, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        categories = ['AI Model', 'Rule-Based']
        means = [ai_stats['mean_reward'], rule_stats['mean_reward']]
        stds = [ai_stats['std_reward'], rule_stats['std_reward']]
        
        axes[0].bar(categories, means, yerr=stds, capsize=10, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        axes[0].set_title('Average Performance', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        lengths = [ai_stats['mean_length'], rule_stats['mean_length']]
        axes[1].bar(categories, lengths, color=['#3498db', '#f39c12'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Steps', fontsize=12, fontweight='bold')
        axes[1].set_title('Mean Episode Length', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        crash_rates = [ai_stats['crash_rate']*100, rule_stats['crash_rate']*100]
        axes[2].bar(categories, crash_rates, color=['#9b59b6', '#e67e22'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
        axes[2].set_ylabel('Crash Rate (%)', fontsize=12, fontweight='bold')
        axes[2].set_title('Safety Comparison', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def close(self):
        if self.env is not None:
            self.env.close()


def main():
    simulator = AutonomousVehicleSimulator()
    
    simulator.create_environment()
    simulator.visualize_environment(save_path="environment_initial.png")
    
    test_actions = [3, 3, 1, 1, 2, 1, 0, 1, 3]
    frames, rewards = simulator.test_manual_control(test_actions)
    
    frames, rewards, actions = simulator.test_rule_based_driver(num_steps=30)
    
    model_path = "autonomous_vehicle_model"
    simulator.train_dqn_model(total_timesteps=20000, model_path=model_path)
    
    all_frames, all_rewards, episode_stats = simulator.test_trained_model(num_episodes=3)
    
    video_path, total_reward, num_frames = simulator.create_video(
        num_steps=50, 
        output_path="autonomous_vehicle_demo.mp4"
    )
    
    ai_stats, rule_stats = simulator.compare_drivers(num_episodes=5)
    simulator.visualize_comparison(ai_stats, rule_stats, save_path="performance_comparison.png")
    
    simulator.close()


if __name__ == "__main__":
    main()
