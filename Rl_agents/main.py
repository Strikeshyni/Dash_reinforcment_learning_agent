"""
Main - Script principal pour l'entraînement des agents RL sur Dash Game
Permet d'entraîner des agents, visualiser des replays et évaluer les performances
"""

import sys
import os
import time
import argparse
import copy
from typing import List, Optional, Tuple
import numpy as np

# Ajouter le dossier Dash_Game au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Dash_Game'))

from src.game_env import DashGameEnv, DashTrainingManager, ReplayFrame
from src.replay_viewer import ReplayViewer, save_replay_to_file
from agents import DQNAgent, EpsilonGreedyAgent, RandomAgent, DuelingDQNAgent, StackedFramesDQNAgent
from visualizer import TrainingPlotter, JumpHeatmap


class BestModelTracker:
    """
    Tracker qui garde le meilleur modèle dans chaque intervalle de sauvegarde.
    Ex: si save_frequency=200, garde le meilleur modèle entre 0-200, 200-400, etc.
    """
    
    def __init__(self, save_frequency: int = 200):
        self.save_frequency = save_frequency
        self.best_progress = 0.0
        self.best_reward = float('-inf')
        self.best_episode = 0
        self.best_weights = None
        self.best_replay = None
        self.interval_start = 0
    
    def update(self, episode: int, progress: float, reward: float, 
               agent, replay: List = None) -> bool:
        """
        Met à jour le tracker avec les résultats d'un épisode.
        
        Returns:
            True si c'est le nouveau meilleur de l'intervalle
        """
        # Score combiné: priorité à la progression, puis reward
        current_score = progress * 1000 + reward
        best_score = self.best_progress * 1000 + self.best_reward
        
        is_new_best = current_score > best_score
        
        if is_new_best:
            self.best_progress = progress
            self.best_reward = reward
            self.best_episode = episode
            # Copier les poids de l'agent
            if hasattr(agent, 'weights'):
                self.best_weights = copy.deepcopy(agent.weights)
                if hasattr(agent, 'target_weights'):
                    self.best_target_weights = copy.deepcopy(agent.target_weights)
            if replay:
                self.best_replay = replay.copy()
        
        return is_new_best
    
    def should_save(self, episode: int) -> bool:
        """Vérifie si on doit sauvegarder (fin d'intervalle)"""
        return episode > 0 and episode % self.save_frequency == 0
    
    def get_best_and_reset(self) -> Tuple[dict, int, float]:
        """
        Retourne les infos du meilleur modèle et reset pour le prochain intervalle.
        
        Returns:
            (best_weights_dict, best_episode, best_progress)
        """
        result = {
            'weights': self.best_weights,
            'target_weights': getattr(self, 'best_target_weights', None),
            'episode': self.best_episode,
            'progress': self.best_progress,
            'reward': self.best_reward,
            'replay': self.best_replay
        }
        
        # Reset pour le prochain intervalle (mais garder les meilleurs poids en mémoire)
        best_episode = self.best_episode
        best_progress = self.best_progress
        
        self.best_progress = 0.0
        self.best_reward = float('-inf')
        self.best_episode = 0
        self.best_replay = None
        self.interval_start = best_episode
        
        return result, best_episode, best_progress


class TrainingVisualizer:
    """Affiche les statistiques d'entraînement en temps réel"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards = []
        self.wins = []
        self.progress = []
        self.steps = []
    
    def add_episode(self, reward: float, won: bool, progress: float, steps: int):
        """Ajoute les stats d'un épisode"""
        self.rewards.append(reward)
        self.wins.append(1 if won else 0)
        self.progress.append(progress)
        self.steps.append(steps)
    
    def get_stats(self) -> dict:
        """Retourne les statistiques moyennes sur la fenêtre"""
        window = slice(-self.window_size, None)
        return {
            'avg_reward': np.mean(self.rewards[window]) if self.rewards else 0,
            'win_rate': np.mean(self.wins[window]) if self.wins else 0,
            'avg_progress': np.mean(self.progress[window]) if self.progress else 0,
            'avg_steps': np.mean(self.steps[window]) if self.steps else 0,
            'total_episodes': len(self.rewards),
            'total_wins': sum(self.wins)
        }
    
    def print_stats(self, episode: int, epsilon: float = None):
        """Affiche les statistiques"""
        stats = self.get_stats()
        eps_str = f"ε={epsilon:.4f}" if epsilon is not None else ""
        print(f"Episode {episode:5d} | "
              f"Reward: {stats['avg_reward']:7.2f} | "
              f"Win Rate: {stats['win_rate']*100:5.1f}% | "
              f"Progress: {stats['avg_progress']*100:5.1f}% | "
              f"Steps: {stats['avg_steps']:6.0f} | "
              f"Total Wins: {stats['total_wins']} | "
              f"{eps_str}")


def train_agent(agent, 
                level_path: str = "layouts/level1.csv",
                num_episodes: int = 1000,
                max_steps: int = 5000,
                replay_frequency: int = 500,
                print_frequency: int = 100,
                save_frequency: int = 500,
                save_path: str = "checkpoints"):
    """
    Entraîne un agent sur le jeu Dash avec sauvegarde intelligente.
    Sauvegarde le MEILLEUR modèle de chaque intervalle (pas le dernier).
    
    Args:
        agent: Agent à entraîner
        level_path: Chemin vers le niveau (relatif à Dash_Game)
        num_episodes: Nombre d'épisodes d'entraînement
        max_steps: Maximum de steps par épisode
        replay_frequency: Fréquence d'enregistrement des replays (1/N)
        print_frequency: Fréquence d'affichage des stats
        save_frequency: Fréquence de sauvegarde de l'agent (garde le meilleur de l'intervalle)
        save_path: Dossier de sauvegarde
    """
    # Créer le dossier de checkpoints
    os.makedirs(save_path, exist_ok=True)
    
    # Changer de répertoire pour charger les assets
    original_dir = os.getcwd()
    dash_game_dir = os.path.join(os.path.dirname(__file__), '..', 'Dash_Game')
    os.chdir(dash_game_dir)
    
    try:
        # Gestionnaire d'entraînement
        manager = DashTrainingManager(
            level_path=level_path,
            replay_frequency=replay_frequency,
            save_replays=True
        )
        
        # Utiliser TrainingPlotter pour générer des graphiques
        plotter = TrainingPlotter(save_dir=os.path.join(original_dir, save_path, "plots"))
        best_tracker = BestModelTracker(save_frequency=save_frequency)
        
        print("=" * 80)
        print(f"Training agent on {level_path}")
        print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
        print(f"Saving BEST model every {save_frequency} episodes")
        print(f"Plots will be saved to: {os.path.join(save_path, 'plots')}")
        print("=" * 80)
        
        start_time = time.time()
        global_best_progress = 0.0
        global_best_episode = 0
        
        for episode in range(1, num_episodes + 1):
            # Créer l'environnement - enregistrer plus souvent pour le tracker
            # should_record = (episode % max(1, replay_frequency // 4) == 0)
            env = DashGameEnv(level_path, record_replay=False, max_steps=max_steps)
            
            # Réinitialiser le frame buffer si agent avec mémoire temporelle
            if hasattr(agent, 'reset_frame_buffer'):
                agent.reset_frame_buffer()
            
            # Exécuter l'épisode
            result = manager.run_episode(env, agent, training=True)
            
            # Enregistrer les stats dans le plotter
            epsilon = getattr(agent, 'epsilon', None)
            plotter.add_episode(
                result['total_reward'],
                result['won'],
                result['progress'],
                epsilon,
                result['steps']
            )
            
            # Tracker le meilleur modèle de l'intervalle
            is_new_best = best_tracker.update(
                episode, 
                result['progress'], 
                result['total_reward'],
                agent,
                None # Pas de replay ici
            )
            
            # Tracker le meilleur global
            if result['progress'] > global_best_progress:
                global_best_progress = result['progress']
                global_best_episode = episode
            
            # Afficher les stats avec indication du meilleur
            if episode % print_frequency == 0:
                epsilon = getattr(agent, 'epsilon', None)
                # Calculer les stats moyennes
                window = 100
                recent_rewards = plotter.rewards_history[-window:] if len(plotter.rewards_history) >= window else plotter.rewards_history
                recent_wins = plotter.wins_history[-window:] if len(plotter.wins_history) >= window else plotter.wins_history
                recent_progress = plotter.progress_history[-window:] if len(plotter.progress_history) >= window else plotter.progress_history
                recent_steps = plotter.steps_history[-window:] if len(plotter.steps_history) >= window else plotter.steps_history
                
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                win_rate = np.mean(recent_wins) if recent_wins else 0
                avg_progress = np.mean(recent_progress) if recent_progress else 0
                avg_steps = np.mean(recent_steps) if recent_steps else 0
                
                eps_str = f"ε={epsilon:.4f}" if epsilon is not None else ""
                print(f"Episode {episode:5d} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Win Rate: {win_rate*100:5.1f}% | "
                      f"Progress: {avg_progress*100:5.1f}% | "
                      f"Steps: {avg_steps:6.0f} | "
                      f"Total Wins: {sum(plotter.wins_history)} | "
                      f"{eps_str}")
                
                # Afficher le meilleur de l'intervalle courant
                if best_tracker.best_episode > 0:
                    interval_start = ((episode - 1) // save_frequency) * save_frequency + 1
                    print(f"  └── Best in [{interval_start}-{episode}]: "
                          f"ep{best_tracker.best_episode} ({best_tracker.best_progress*100:.1f}%)")
            
            # Sauvegarder le MEILLEUR de l'intervalle
            if best_tracker.should_save(episode):
                best_data, best_ep, best_prog = best_tracker.get_best_and_reset()
                
                interval_start = episode - save_frequency + 1
                print(f"\n{'='*60}")
                print(f"Saving BEST model from interval [{interval_start}-{episode}]")
                print(f"   Best episode: {best_ep} with {best_prog*100:.1f}% progress")
                
                # Restaurer les meilleurs poids dans l'agent pour sauvegarder
                if best_data['weights'] is not None and hasattr(agent, 'weights'):
                    # Sauvegarder les poids actuels
                    current_weights = copy.deepcopy(agent.weights)
                    current_target = copy.deepcopy(getattr(agent, 'target_weights', None))
                    
                    # Mettre les meilleurs poids pour sauvegarder
                    agent.weights = best_data['weights']
                    if best_data['target_weights'] is not None:
                        agent.target_weights = best_data['target_weights']
                    
                    # Sauvegarder
                    agent_path = os.path.join(original_dir, save_path, f"agent_best_ep{best_ep}.pkl")
                    agent.save(agent_path)
                    
                    # Sauvegarder le replay correspondant
                    # if best_data['replay']:
                    #     replay_path = os.path.join(original_dir, save_path, f"replay_best_ep{best_ep}.vis")
                    #     save_replay_to_file(best_data['replay'], replay_path)
                    
                    # Restaurer les poids actuels pour continuer l'entraînement
                    agent.weights = current_weights
                    if current_target is not None:
                        agent.target_weights = current_target
                    
                    print(f"   Saved: agent_best_ep{best_ep}.pkl")
                else:
                    # Agent sans weights (ex: RandomAgent)
                    if hasattr(agent, 'save'):
                        agent_path = os.path.join(original_dir, save_path, f"agent_ep{episode}.pkl")
                        agent.save(agent_path)
                
                print(f"{'='*60}\n")
        
        elapsed_time = time.time() - start_time
        
        print("=" * 80)
        print(f"Training completed in {elapsed_time:.1f}s")
        print(f"Global best: Episode {global_best_episode} with {global_best_progress*100:.1f}% progress")
        print(f"Final stats:")
        
        # Calculer et afficher les stats finales
        window = 100
        recent_rewards = plotter.rewards_history[-window:]
        recent_wins = plotter.wins_history[-window:]
        recent_progress = plotter.progress_history[-window:]
        recent_steps = plotter.steps_history[-window:]
        
        print(f"  Avg Reward (last {window}): {np.mean(recent_rewards):.2f}")
        print(f"  Win Rate (last {window}): {np.mean(recent_wins)*100:.1f}%")
        print(f"  Avg Progress (last {window}): {np.mean(recent_progress)*100:.1f}%")
        print(f"  Total Wins: {sum(plotter.wins_history)}")
        print("=" * 80)
        
        # Sauvegarder l'agent final
        if hasattr(agent, 'save'):
            final_path = os.path.join(original_dir, save_path, f"{agent.__class__.__name__}_final.pkl")
            agent.save(final_path)
        
        # Générer et sauvegarder tous les graphiques
        print("\nGenerating training plots...")
        plotter.plot_all(window_size=100, save=True, show=False)
        plotter.save_data()
        print(f"Plots saved to: {os.path.join(save_path, 'plots')}")
        print(f"   - training_summary.png (all metrics)")
        print(f"   - rewards.png")
        print(f"   - win_rate.png")
        print(f"   - progress.png")
        print(f"   - epsilon.png")
        print(f"   - training_data.pkl (raw data)")
        
        # Retourner le gestionnaire pour accéder aux replays
        return manager, plotter
        
    finally:
        os.chdir(original_dir)


def evaluate_agent(agent,
                   level_path: str = "layouts/level1.csv",
                   num_episodes: int = 100,
                   max_steps: int = 5000,
                   record_best: bool = True):
    """
    Évalue un agent sans entraînement
    
    Args:
        agent: Agent à évaluer
        level_path: Chemin vers le niveau
        num_episodes: Nombre d'épisodes d'évaluation
        max_steps: Maximum de steps par épisode
        record_best: Si True, enregistre la meilleure partie
        
    Returns:
        dict: Statistiques d'évaluation
    """
    original_dir = os.getcwd()
    dash_game_dir = os.path.join(os.path.dirname(__file__), '..', 'Dash_Game')
    os.chdir(dash_game_dir)
    
    try:
        wins = 0
        total_reward = 0
        total_progress = 0
        best_progress = 0
        best_replay = None
        
        for episode in range(num_episodes):
            env = DashGameEnv(level_path, record_replay=True, max_steps=max_steps)
            
            # Réinitialiser le frame buffer si agent avec mémoire temporelle
            if hasattr(agent, 'reset_frame_buffer'):
                agent.reset_frame_buffer()
            
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.select_action(obs, training=False)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            total_progress += info['progress']
            
            if info['won']:
                wins += 1
            
            if info['progress'] > best_progress:
                best_progress = info['progress']
                if record_best:
                    best_replay = env.get_replay()
            
            if (episode + 1) % 10 == 0:
                print(f"Evaluation: {episode + 1}/{num_episodes} episodes completed")
        
        stats = {
            'win_rate': wins / num_episodes,
            'avg_reward': total_reward / num_episodes,
            'avg_progress': total_progress / num_episodes,
            'best_progress': best_progress,
            'total_wins': wins
        }
        
        print("\nEvaluation Results:")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Avg Reward: {stats['avg_reward']:.2f}")
        print(f"  Avg Progress: {stats['avg_progress']*100:.1f}%")
        print(f"  Best Progress: {stats['best_progress']*100:.1f}%")
        
        if record_best and best_replay:
            replay_path = os.path.join(original_dir, f"{agent.__class__.__name__}_best_replay.vis")
            save_replay_to_file(best_replay, replay_path)
            print(f"Meilleur replay sauvegardé dans: {replay_path}")
        
        return stats, best_replay
        
    finally:
        os.chdir(original_dir)


def watch_replay(replay_path: str = None, 
                 replay_data: List = None,
                 level_path: str = "layouts/level1.csv",
                 speed: float = 1.0):
    """
    Visualise un replay
    
    Args:
        replay_path: Chemin vers le fichier de replay
        replay_data: Données de replay directement
        level_path: Chemin vers le niveau
        speed: Vitesse de lecture
    """
    original_dir = os.getcwd()
    
    # Résoudre le chemin absolu du replay avant de changer de dossier
    if replay_path and not os.path.isabs(replay_path):
        replay_path = os.path.abspath(replay_path)

    dash_game_dir = os.path.join(os.path.dirname(__file__), '..', 'Dash_Game')
    os.chdir(dash_game_dir)
    
    try:
        if replay_path:
            import pickle
            with open(replay_path, 'rb') as f:
                replay_data = pickle.load(f)
            
            # Vérification du type de fichier
            if isinstance(replay_data, dict) and ('weights' in replay_data or 'q_values' in replay_data):
                print(f"\nERREUR: Le fichier '{os.path.basename(replay_path)}' semble être un checkpoint d'agent (.pkl), pas un replay (.vis).")
                print("Les checkpoints contiennent l'état de l'IA (poids), pas l'enregistrement d'une partie.")
                print("Pour regarder jouer cet agent, utilisez '--mode eval --load <votre_checkpoint.pkl>'")
                return
        
        if replay_data:
            viewer = ReplayViewer(level_path)
            viewer.play_replay(replay_data, speed)
            viewer.close()
        else:
            print("No replay data provided")
            
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description="Dash RL Training")
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'watch', 'compare'],
                       help='Mode: train, eval, watch, or compare agents')
    parser.add_argument('--agent', type=str, default='dqn',
                       choices=['dqn', 'dueling', 'stacked', 'epsilon', 'random'],
                       help='Type of agent: dqn, dueling (Dueling DQN), stacked (Stacked Frames), epsilon, random')
    parser.add_argument('--level', type=str, default='layouts/level1.csv',
                       help='Path to level file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum steps per episode')
    parser.add_argument('--replay-freq', type=int, default=500,
                       help='Frequency of replay recording')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load agent from')
    parser.add_argument('--replay', type=str, default=None,
                       help='Path to replay file for watch mode')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Replay speed multiplier')
    parser.add_argument('--grid-search', action='store_true',
                       help='Run hyperparameter grid search')
    parser.add_argument('--save-path', type=str, default="checkpoints",
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Mode Grid Search
    if args.grid_search:
        print("Starting Grid Search...")
        
        # Définition de la grille de paramètres
        param_grid = {
            'learning_rate': [0.001, 0.0005],
            'hidden_sizes': [[128, 64]],
            'gamma': [0.99, 0.95],
            'batch_size': [64, 128]
        }
        
        # Générer toutes les combinaisons
        import itertools
        keys = param_grid.keys()
        combinations = list(itertools.product(*param_grid.values()))
        
        best_score = -float('inf')
        best_params = None
        
        print(f"Testing {len(combinations)} combinations...")
        
        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            print(f"\nConfiguration {i+1}/{len(combinations)}: {params}")
            
            # Créer l'agent avec ces paramètres
            agent = DQNAgent(
                state_size=95,  # 5 + 30 * 3 (updated observation space)
                action_size=2,
                learning_rate=params['learning_rate'],
                hidden_sizes=params['hidden_sizes'],
                gamma=params['gamma'],
                batch_size=params['batch_size'],
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.99
            )
            
            # Entraînement court (pour test)
            manager, _ = train_agent(
                agent, 
                level_path=args.level, 
                num_episodes=200,  # Nombre d'épisodes réduit pour le grid search
                max_steps=3000, 
                print_frequency=50,
                save_frequency=1000, # Ne pas sauvegarder les intermédiaires
                save_path="grid_search_tmp"
            )
            
            # Évaluation
            stats, _ = evaluate_agent(
                agent, 
                level_path=args.level, 
                num_episodes=20, 
                record_best=False
            )
            
            score = stats['avg_progress'] + stats['avg_reward'] / 1000.0
            print(f"Score: {score:.4f} (Win rate: {stats['win_rate']:.2f}, Progress: {stats['avg_progress']:.2f})")
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"*** New Best Configuration! ***")
        
        print("\n" + "="*60)
        print("GRID SEARCH RESULT")
        print("="*60)
        print(f"Best Score: {best_score}")
        print(f"Best Params: {best_params}")
        return

    # Créer l'agent
    if args.agent == 'dqn':
        agent = DQNAgent()
        if args.load:
            agent.load(args.load)
    elif args.agent == 'dueling':
        agent = DuelingDQNAgent(
            state_size=95,
            action_size=2,
            hidden_sizes=[256, 128],
            learning_rate=0.005,
            gamma=0.99,
            epsilon_decay=0.995,
            use_prioritized_replay=True
        )
        if args.load:
            agent.load(args.load)
    elif args.agent == 'stacked':
        agent = StackedFramesDQNAgent(
            base_state_size=95,
            num_frames=4,
            action_size=2,
            hidden_sizes=[256, 128, 64],
            learning_rate=0.005,
            gamma=0.99,
            epsilon_decay=0.995
        )
        if args.load:
            agent.load(args.load)
    elif args.agent == 'epsilon':
        agent = EpsilonGreedyAgent(epsilon=0.1)
    else:
        agent = RandomAgent(jump_probability=0.3)
    
    if args.mode == 'train':
        manager, viz = train_agent(
            agent,
            level_path=args.level,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            replay_frequency=args.replay_freq,
            save_path=args.save_path
        )
        
        # Afficher le meilleur replay si disponible
        best_replay = manager.get_replay(-1)
        if best_replay:
            print("\nWatching last recorded replay...")
            watch_replay(replay_data=best_replay, level_path=args.level)
    
    elif args.mode == 'eval':
        stats, best_replay = evaluate_agent(
            agent,
            level_path=args.level,
            num_episodes=args.episodes,
            max_steps=args.max_steps
        )
        
        # Générer la heatmap des sauts
        print("\nGénération de la heatmap des sauts...")
        
        # On doit changer le répertoire de travail pour que DashGameEnv trouve les fichiers
        original_dir = os.getcwd()
        dash_game_dir = os.path.join(os.path.dirname(__file__), '..', 'Dash_Game')
        
        try:
            os.chdir(dash_game_dir)
            # Charger le layout pour la visualisation
            from src.layout_reader import DashLayoutReader
            layout = DashLayoutReader.read_layout_from_csv(args.level)
            
            # Initialiser la heatmap avec le layout
            # On estime la fin du niveau par la position du dernier objet + marge
            if layout.objects:
                last_obj_x = max(obj.position[0] for obj in layout.objects)
                level_len = last_obj_x + 50
            else:
                level_len = 1000.0
                
            heatmap = JumpHeatmap(level_length=level_len, layout=layout)

            # On refait quelques épisodes pour avoir des données
            for i in range(min(50, args.episodes)): # Max 50 pour pas être trop long
                 env = DashGameEnv(args.level, record_replay=True, max_steps=args.max_steps)
                 obs = env.reset()
                 done = False
                 while not done:
                     action = agent.select_action(obs, training=False)
                     obs, _, done, _ = env.step(action)
                 heatmap.add_replay(env.get_replay())
        finally:
            os.chdir(original_dir)
        
        heatmap.plot(save_path="jump_heatmap.png", show=False)
        print("Heatmap sauvegardée dans jump_heatmap.png")
        
        if best_replay:
            print("\nWatching best replay...")
            watch_replay(replay_data=best_replay, level_path=args.level)
    
    elif args.mode == 'watch':
        if args.replay:
            watch_replay(replay_path=args.replay, level_path=args.level, speed=args.speed)
        else:
            print("Please specify a replay file with --replay")
    
    elif args.mode == 'compare':
        print("Comparing agents...")
        
        agents = {
            'Random': RandomAgent(),
            'Epsilon-Greedy': EpsilonGreedyAgent(),
            'DQN': DQNAgent(),
            'Dueling DQN': DuelingDQNAgent(),
            'Stacked Frames DQN': StackedFramesDQNAgent()
        }
        
        for name, agent in agents.items():
            print(f"\nTraining {name}...")
            manager, viz = train_agent(
                agent,
                level_path=args.level,
                num_episodes=min(args.episodes, 500),
                max_steps=args.max_steps,
                print_frequency=50,
                save_path=f"checkpoints_{name.lower().replace('-', '_')}"
            )
            
            print(f"\nEvaluating {name}...")
            stats, _ = evaluate_agent(
                agent,
                level_path=args.level,
                num_episodes=50,
                max_steps=args.max_steps,
                record_best=True
            )


if __name__ == "__main__":
    main()
