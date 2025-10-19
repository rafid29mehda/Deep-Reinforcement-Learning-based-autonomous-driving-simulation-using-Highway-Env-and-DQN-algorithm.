from autonomous_vehicle_simulation import AutonomousVehicleSimulator


def simple_demo():
    simulator = AutonomousVehicleSimulator()
    simulator.create_environment()
    simulator.visualize_environment(save_path="environment_initial.png")
    simulator.train_dqn_model(total_timesteps=10000, model_path="demo_model")
    frames, rewards, stats = simulator.test_trained_model(num_episodes=2)
    print(f"Test completed. Episode stats: {stats}")
    simulator.close()


def full_pipeline():
    simulator = AutonomousVehicleSimulator()
    simulator.create_environment()
    simulator.train_dqn_model(total_timesteps=20000, model_path="full_model")
    ai_stats, rule_stats = simulator.compare_drivers(num_episodes=5)
    
    print(f"AI Mean Reward: {ai_stats['mean_reward']:.2f}")
    print(f"AI Crash Rate: {ai_stats['crash_rate']*100:.1f}%")
    print(f"Rule-Based Mean Reward: {rule_stats['mean_reward']:.2f}")
    print(f"Rule-Based Crash Rate: {rule_stats['crash_rate']*100:.1f}%")
    
    simulator.visualize_comparison(ai_stats, rule_stats, save_path="comparison.png")
    simulator.create_video(output_path="autonomous_demo.mp4")
    simulator.close()


if __name__ == "__main__":
    simple_demo()
