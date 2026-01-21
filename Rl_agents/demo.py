"""
Demo script - Démonstration rapide de l'entraînement RL sur Dash Game
Lance un entraînement court et affiche les résultats
"""

import sys
import os

# Ajouter les chemins
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Dash_Game'))

from src.game_env import DashGameEnv, DashTrainingManager
from src.replay_viewer import ReplayViewer
from agents import DQNAgent, RandomAgent
from visualizer import TrainingPlotter


def quick_demo():
    """Démonstration rapide d'entraînement"""
    
    print("=" * 60)
    print("  DASH GAME - RL Training Demo")
    print("=" * 60)
    
    # Changer vers le dossier Dash_Game pour charger les assets
    original_dir = os.getcwd()
    dash_game_dir = os.path.join(os.path.dirname(__file__), '..', 'Dash_Game')
    os.chdir(dash_game_dir)
    
    try:
        # Configuration
        level = "layouts/level1.csv"
        num_episodes = 200
        
        # Créer l'agent DQN
        agent = DQNAgent(
            state_size=65,
            action_size=2,
            hidden_sizes=[64, 32],
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.99
        )
        
        # Plotter pour visualisation
        plotter = TrainingPlotter(save_dir=os.path.join(original_dir, "demo_plots"))
        
        # Gestionnaire d'entraînement
        manager = DashTrainingManager(level, replay_frequency=50, save_replays=True)
        
        print(f"\nEntraînement de l'agent DQN sur {num_episodes} épisodes...")
        print("-" * 60)
        
        for episode in range(1, num_episodes + 1):
            # Créer l'environnement
            should_record = (episode % 50 == 0)
            env = DashGameEnv(level, record_replay=should_record, max_steps=3000)
            
            # Exécuter l'épisode
            obs = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(obs, training=True)
                next_obs, reward, done, info = env.step(action)
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward
            
            # Enregistrer les stats
            plotter.add_episode(
                total_reward,
                info['won'],
                info['progress'],
                agent.epsilon,
                info['steps']
            )
            
            # Sauvegarder le replay si enregistré
            if should_record and env.get_replay():
                manager.saved_replays.append(env.get_replay())
            
            # Affichage
            if episode % 20 == 0:
                wins = sum(plotter.wins_history[-20:])
                avg_progress = sum(plotter.progress_history[-20:]) / 20 * 100
                print(f"Episode {episode:4d} | "
                      f"ε={agent.epsilon:.3f} | "
                      f"Reward: {total_reward:7.1f} | "
                      f"Progress: {info['progress']*100:5.1f}% | "
                      f"Wins(20): {wins} | "
                      f"Avg Progress: {avg_progress:.1f}%")
        
        print("-" * 60)
        print("\nEntraînement terminé!")
        
        # Générer les graphiques
        print("\nGénération des graphiques...")
        plotter.plot_all(save=True, show=False)
        print(f"Graphiques sauvegardés dans {plotter.save_dir}")
        
        # Évaluation finale
        print("\nÉvaluation de l'agent (20 parties sans exploration)...")
        wins = 0
        best_progress = 0
        best_replay = None
        
        for i in range(20):
            env = DashGameEnv(level, record_replay=True, max_steps=3000)
            obs = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(obs, training=False)
                obs, _, done, info = env.step(action)
            
            if info['won']:
                wins += 1
            if info['progress'] > best_progress:
                best_progress = info['progress']
                best_replay = env.get_replay()
        
        print(f"  Taux de victoire: {wins}/20 ({wins*5}%)")
        print(f"  Meilleure progression: {best_progress*100:.1f}%")
        
        # Visualiser la meilleure partie
        if best_replay:
            print("\nVisualization de la meilleure partie...")
            print("Contrôles: ESPACE=Pause, ←/→=Naviguer, ESC=Quitter")
            
            viewer = ReplayViewer(level)
            viewer.play_replay(best_replay, speed=1.0)
            viewer.close()
        
    finally:
        os.chdir(original_dir)
    
    print("\n" + "=" * 60)
    print("  Démo terminée!")
    print("=" * 60)


def test_environment():
    """Test de l'environnement avec un agent aléatoire"""
    
    print("Test de l'environnement avec agent aléatoire...")
    
    original_dir = os.getcwd()
    dash_game_dir = os.path.join(os.path.dirname(__file__), '..', 'Dash_Game')
    os.chdir(dash_game_dir)
    
    try:
        env = DashGameEnv("layouts/level1.csv", record_replay=True)
        
        obs = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Observation sample: {obs[:10]}")
        
        agent = RandomAgent(jump_probability=0.25)
        
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        print(f"\nÉpisode terminé:")
        print(f"  Steps: {steps}")
        print(f"  Won: {info['won']}")
        print(f"  Died: {info['died']}")
        print(f"  Progress: {info['progress']*100:.1f}%")
        
        # Afficher les infos sur l'état
        state = env.get_state_info()
        print(f"\nÉtat final du jeu:")
        print(f"  Position joueur: ({state.player_x:.2f}, {state.player_y:.2f})")
        print(f"  Vélocité: {state.player_velocity:.3f}")
        print(f"  Gravité inversée: {state.reversed_gravity}")
        print(f"  Objets proches: {len(state.nearby_objects)}")
        
        if state.nearby_objects:
            print("\n  Premiers objets proches:")
            for i, obj in enumerate(state.nearby_objects[:5]):
                print(f"    {i+1}. Type={obj.object_type}, RelX={obj.relative_x:.2f}, RelY={obj.relative_y:.2f}")
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test environment only')
    args = parser.parse_args()
    
    if args.test:
        test_environment()
    else:
        quick_demo()
