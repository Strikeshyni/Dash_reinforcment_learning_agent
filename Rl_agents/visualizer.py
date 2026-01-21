"""
Visualizer - Outils de visualisation pour l'entraînement RL
Inspiré du TP1 Ruée vers l'or
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os


class TrainingPlotter:
    """Génère des graphiques pour suivre l'entraînement"""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.rewards_history = []
        self.wins_history = []
        self.progress_history = []
        self.epsilon_history = []
        self.steps_history = []
    
    def add_episode(self, reward: float, won: bool, progress: float, 
                    epsilon: float = None, steps: int = 0):
        """Enregistre les métriques d'un épisode"""
        self.rewards_history.append(reward)
        self.wins_history.append(1 if won else 0)
        self.progress_history.append(progress)
        self.steps_history.append(steps)
        if epsilon is not None:
            self.epsilon_history.append(epsilon)
    
    def plot_rewards(self, window_size: int = 100, save: bool = True, show: bool = False):
        """Plot de la récompense moyenne glissante"""
        if len(self.rewards_history) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Récompenses brutes
        ax.plot(self.rewards_history, alpha=0.3, color='blue', label='Récompense par épisode')
        
        # Moyenne glissante
        if len(self.rewards_history) >= window_size:
            moving_avg = np.convolve(self.rewards_history, 
                                     np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size-1, len(self.rewards_history)), moving_avg, 
                   color='red', linewidth=2, label=f'Moyenne glissante ({window_size} épisodes)')
        
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense')
        ax.set_title('Évolution de la récompense au cours de l\'entraînement')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'rewards.png'), dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def plot_win_rate(self, window_size: int = 100, save: bool = True, show: bool = False):
        """Plot du taux de victoire glissant"""
        if len(self.wins_history) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Moyenne glissante du taux de victoire
        if len(self.wins_history) >= window_size:
            win_rate = np.convolve(self.wins_history, 
                                   np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size-1, len(self.wins_history)), win_rate * 100, 
                   color='green', linewidth=2)
        
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Taux de victoire (%)')
        ax.set_title(f'Taux de victoire glissant ({window_size} épisodes)')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'win_rate.png'), dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def plot_progress(self, window_size: int = 100, save: bool = True, show: bool = False):
        """Plot de la progression moyenne"""
        if len(self.progress_history) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Progression brute
        ax.plot([p * 100 for p in self.progress_history], alpha=0.3, 
               color='purple', label='Progression par épisode')
        
        # Moyenne glissante
        if len(self.progress_history) >= window_size:
            moving_avg = np.convolve(self.progress_history, 
                                     np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size-1, len(self.progress_history)), moving_avg * 100, 
                   color='darkviolet', linewidth=2, 
                   label=f'Moyenne glissante ({window_size} épisodes)')
        
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Progression (%)')
        ax.set_title('Progression dans le niveau au cours de l\'entraînement')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'progress.png'), dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def plot_epsilon(self, save: bool = True, show: bool = False):
        """Plot de la décroissance d'epsilon"""
        if len(self.epsilon_history) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(self.epsilon_history, color='orange', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Décroissance d\'epsilon (exploration)')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'epsilon.png'), dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def plot_all(self, window_size: int = 100, save: bool = True, show: bool = False):
        """Génère tous les graphiques sur une seule figure"""
        if len(self.rewards_history) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Récompenses
        ax = axes[0, 0]
        ax.plot(self.rewards_history, alpha=0.3, color='blue')
        if len(self.rewards_history) >= window_size:
            moving_avg = np.convolve(self.rewards_history, 
                                     np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size-1, len(self.rewards_history)), moving_avg, 
                   color='red', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense')
        ax.set_title('Récompense')
        ax.grid(alpha=0.3)
        
        # Taux de victoire
        ax = axes[0, 1]
        if len(self.wins_history) >= window_size:
            win_rate = np.convolve(self.wins_history, 
                                   np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size-1, len(self.wins_history)), win_rate * 100, 
                   color='green', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Taux de victoire (%)')
        ax.set_title('Taux de victoire')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        
        # Progression
        ax = axes[1, 0]
        ax.plot([p * 100 for p in self.progress_history], alpha=0.3, color='purple')
        if len(self.progress_history) >= window_size:
            moving_avg = np.convolve(self.progress_history, 
                                     np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size-1, len(self.progress_history)), moving_avg * 100, 
                   color='darkviolet', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Progression (%)')
        ax.set_title('Progression')
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        
        # Epsilon
        ax = axes[1, 1]
        if self.epsilon_history:
            ax.plot(self.epsilon_history, color='orange', linewidth=2)
            ax.set_ylabel('Epsilon')
        else:
            ax.plot(self.steps_history, color='cyan', linewidth=1, alpha=0.5)
            ax.set_ylabel('Steps')
        ax.set_xlabel('Épisode')
        ax.set_title('Epsilon / Steps')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_summary.png'), 
                       dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def save_data(self, filepath: str = None):
        """Sauvegarde les données d'entraînement"""
        import pickle
        
        if filepath is None:
            filepath = os.path.join(self.save_dir, 'training_data.pkl')
        
        data = {
            'rewards': self.rewards_history,
            'wins': self.wins_history,
            'progress': self.progress_history,
            'epsilon': self.epsilon_history,
            'steps': self.steps_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Training data saved to {filepath}")
    
    def load_data(self, filepath: str):
        """Charge des données d'entraînement"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.rewards_history = data.get('rewards', [])
        self.wins_history = data.get('wins', [])
        self.progress_history = data.get('progress', [])
        self.epsilon_history = data.get('epsilon', [])
        self.steps_history = data.get('steps', [])
        
        print(f"Loaded {len(self.rewards_history)} episodes from {filepath}")


def compare_agents(results: Dict[str, Dict], window_size: int = 100,
                   save_path: str = "plots/comparison.png"):
    """
    Compare les performances de plusieurs agents
    
    Args:
        results: Dict[agent_name, dict with 'rewards', 'wins', 'progress']
        window_size: Taille de la fenêtre glissante
        save_path: Chemin de sauvegarde du graphique
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (name, data), color in zip(results.items(), colors):
        # Récompenses
        if len(data.get('rewards', [])) >= window_size:
            rewards = np.convolve(data['rewards'], 
                                 np.ones(window_size)/window_size, mode='valid')
            axes[0].plot(rewards, label=name, color=color, linewidth=2)
        
        # Taux de victoire
        if len(data.get('wins', [])) >= window_size:
            wins = np.convolve(data['wins'], 
                              np.ones(window_size)/window_size, mode='valid')
            axes[1].plot(wins * 100, label=name, color=color, linewidth=2)
        
        # Progression
        if len(data.get('progress', [])) >= window_size:
            progress = np.convolve(data['progress'], 
                                  np.ones(window_size)/window_size, mode='valid')
            axes[2].plot(progress * 100, label=name, color=color, linewidth=2)
    
    axes[0].set_title('Récompense moyenne')
    axes[0].set_xlabel('Épisode')
    axes[0].set_ylabel('Récompense')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_title('Taux de victoire')
    axes[1].set_xlabel('Épisode')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].set_title('Progression moyenne')
    axes[2].set_xlabel('Épisode')
    axes[2].set_ylabel('Progress (%)')
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Test du plotter
    plotter = TrainingPlotter("test_plots")
    
    # Simuler un entraînement
    import random
    epsilon = 1.0
    
    for i in range(500):
        reward = random.gauss(10 + i * 0.1, 5)
        won = random.random() < (i / 1000)
        progress = min(1.0, i / 400 + random.gauss(0, 0.1))
        epsilon *= 0.99
        
        plotter.add_episode(reward, won, progress, epsilon, i * 10)
    
    plotter.plot_all(show=True)
    plotter.save_data()

class JumpHeatmap:
    """Générateur de heatmap pour les sauts avec visualisation du niveau"""
    
    def __init__(self, level_length: float = 1000.0, bin_size: float = 1.0, layout=None):
        self.level_length = level_length
        self.bin_size = bin_size
        self.bins = int(level_length / bin_size) + 1
        self.jump_counts = np.zeros(self.bins)
        self.pass_counts = np.zeros(self.bins)
        self.death_positions = []  # Positions de mort
        self.layout = layout
        
    def add_replay(self, frames: List):
        """Ajoute les données d'un replay à la heatmap"""
        for i, frame in enumerate(frames):
            # Calculer la position x réelle du joueur
            player_x = frame.player_position[0] + frame.line
            idx = int(player_x / self.bin_size)
            
            if 0 <= idx < self.bins:
                self.pass_counts[idx] += 1
                if frame.action == 1:  # Saut
                    self.jump_counts[idx] += 1
        
        # Enregistrer la position de mort (dernière position si pas gagné)
        if frames and len(frames) > 0:
            last_frame = frames[-1]
            player_x = last_frame.player_position[0] + last_frame.line
            # Vérifier si c'est une mort (pas une victoire - progress < 100%)
            if hasattr(last_frame, 'progress') and last_frame.progress < 0.99:
                self.death_positions.append(player_x)
                    
    def plot(self, save_path: str = None, show: bool = True):
        """Affiche une belle heatmap superposée au niveau"""
        if np.sum(self.pass_counts) == 0:
            print("Aucune donnée pour la heatmap.")
            return
            
        # Calculer la probabilité de saut par bin
        jump_prob = np.zeros_like(self.jump_counts)
        mask = self.pass_counts > 0
        jump_prob[mask] = self.jump_counts[mask] / self.pass_counts[mask]
        
        # Déterminer les limites du niveau
        max_seen = np.max(np.where(self.pass_counts > 0)[0]) * self.bin_size + 10
        
        # Créer la figure avec style sombre
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(24, 8))
        
        # Fond dégradé
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack([gradient] * 100)
        ax.imshow(gradient, aspect='auto', cmap='Blues', alpha=0.15,
                  extent=[0, max_seen, -1, 15])
        
        # 1. Dessiner le layout avec style amélioré
        if self.layout:
            self._draw_level(ax)
        
        # 2. Dessiner la zone de jeu (sol)
        ax.axhline(y=0, color='white', linewidth=2, alpha=0.5)
        ax.fill_between([0, max_seen], -1, 0, color='#1a1a2e', alpha=0.8)
        
        # 3. Afficher les zones de saut avec effet de glow
        x_positions = np.arange(self.bins) * self.bin_size
        
        # Créer les barres de saut
        for i, (x, prob) in enumerate(zip(x_positions, jump_prob)):
            if prob > 0.05 and x < max_seen:  # Seuil minimum
                # Hauteur basée sur la probabilité
                height = 12 * prob
                
                # Couleur du bleu au rouge selon l'intensité
                color = plt.cm.plasma(prob)
                
                # Barre principale
                bar = plt.Rectangle((x, 0), self.bin_size * 0.8, height,
                                    facecolor=color, alpha=0.7,
                                    edgecolor='white', linewidth=0.5)
                ax.add_patch(bar)
                
                # Effet de glow
                for glow in range(3):
                    glow_bar = plt.Rectangle((x - glow*0.3, 0), 
                                            self.bin_size * 0.8 + glow*0.6, 
                                            height + glow*0.2,
                                            facecolor=color, alpha=0.1 - glow*0.03)
                    ax.add_patch(glow_bar)
        
        # 4. Afficher les positions de mort avec des marqueurs
        if self.death_positions:
            # Grouper les morts par zone
            death_bins = np.zeros(self.bins)
            for pos in self.death_positions:
                idx = int(pos / self.bin_size)
                if 0 <= idx < self.bins:
                    death_bins[idx] += 1
            
            # Afficher les zones de mort
            max_deaths = max(death_bins) if max(death_bins) > 0 else 1
            for i, (x, deaths) in enumerate(zip(x_positions, death_bins)):
                if deaths > 0 and x < max_seen:
                    intensity = deaths / max_deaths
                    skull_size = 100 + 200 * intensity
                    ax.scatter(x + 0.5, -0.5, s=skull_size, c='red', marker='x',
                              alpha=0.3 + 0.5 * intensity, linewidths=2)
        
        # 5. Ajouter une légende et annotations
        # Colorbar pour la probabilité de saut
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
        cbar.set_label('Probabilité de saut', fontsize=12, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Titre et labels
        total_episodes = int(np.max(self.pass_counts)) if np.max(self.pass_counts) > 0 else 0
        ax.set_title(f'🎮 Heatmap des Sauts - {len(self.death_positions)} morts analysées\n'
                    f'(Max passages par zone: {total_episodes})',
                    fontsize=16, fontweight='bold', color='white', pad=20)
        
        ax.set_xlabel('Position dans le niveau', fontsize=12, color='white')
        ax.set_ylabel('Hauteur', fontsize=12, color='white')
        
        # Limites
        ax.set_xlim(0, max_seen)
        ax.set_ylim(-1.5, 14)
        
        # Grille subtile
        ax.grid(True, alpha=0.1, linestyle='--')
        
        # Légende personnalisée
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#4a4a4a', edgecolor='white', label='Blocs'),
            Patch(facecolor='black', edgecolor='gray', label='Spikes'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                   markersize=10, label='Zones de mort', linestyle='None'),
            Patch(facecolor=plt.cm.plasma(0.7), alpha=0.7, label='Zones de saut')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 facecolor='#1a1a2e', edgecolor='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight',
                       facecolor='#0a0a1a', edgecolor='none')
            print(f"Heatmap sauvegardée: {save_path}")
            
        if show:
            plt.show()
        plt.close()
        
        # Restaurer le style par défaut
        plt.style.use('default')
    
    def _draw_level(self, ax):
        """Dessine les éléments du niveau avec un style amélioré"""
        try:
            from src.layout import DashObjectType
            
            # Couleurs améliorées
            colors = {
                DashObjectType.BLOCK: "#ffffff",
                DashObjectType.UP_SPIKE: '#4a4a4a',
                DashObjectType.DOWN_SPIKE: '#4a4a4a',
                DashObjectType.RIGHT_SPIKE: '#4a4a4a',
                DashObjectType.LEFT_SPIKE: '#4a4a4a',
                DashObjectType.YELLOW_ORB: '#FFD700',
                DashObjectType.BLUE_ORB: '#00BFFF',
                DashObjectType.PINK_ORB: '#FF69B4',
                DashObjectType.RED_ORB: '#FF4444',
                DashObjectType.GREEN_ORB: '#00FF7F',
                DashObjectType.BLACK_ORB: '#2F2F2F',
            }
            
            edge_colors = {
                DashObjectType.BLOCK: '#6a6a6a',
                DashObjectType.UP_SPIKE: '#3a3a3a',
                DashObjectType.DOWN_SPIKE: '#3a3a3a',
                DashObjectType.RIGHT_SPIKE: '#3a3a3a',
                DashObjectType.LEFT_SPIKE: '#3a3a3a',
            }
            
            for obj in self.layout.objects:
                x, y = obj.position
                color = colors.get(obj.objectType, '#8B008B')
                edge_color = edge_colors.get(obj.objectType, color)
                
                if obj.objectType == DashObjectType.BLOCK:
                    # Bloc avec style 3D léger
                    rect = plt.Rectangle((x, y), 1, 1, 
                                         facecolor=color, 
                                         edgecolor=edge_color,
                                         linewidth=0.5,
                                         alpha=0.8)
                    ax.add_patch(rect)
                    # Highlight 3D
                    highlight = plt.Rectangle((x, y + 0.8), 1, 0.2,
                                             facecolor='white', alpha=0.1)
                    ax.add_patch(highlight)
                    
                elif 'SPIKE' in str(obj.objectType):
                    # Spikes avec forme triangulaire
                    if obj.objectType == DashObjectType.UP_SPIKE:
                        triangle = plt.Polygon([(x + 0.5, y + 1), (x, y), (x + 1, y)],
                                              facecolor=color, edgecolor=edge_color,
                                              linewidth=0.5, alpha=0.9)
                    elif obj.objectType == DashObjectType.DOWN_SPIKE:
                        triangle = plt.Polygon([(x + 0.5, y), (x, y + 1), (x + 1, y + 1)],
                                              facecolor=color, edgecolor=edge_color,
                                              linewidth=0.5, alpha=0.9)
                    elif obj.objectType == DashObjectType.RIGHT_SPIKE:
                        triangle = plt.Polygon([(x + 1, y + 0.5), (x, y), (x, y + 1)],
                                              facecolor=color, edgecolor=edge_color,
                                              linewidth=0.5, alpha=0.9)
                    elif obj.objectType == DashObjectType.LEFT_SPIKE:
                        triangle = plt.Polygon([(x, y + 0.5), (x + 1, y), (x + 1, y + 1)],
                                              facecolor=color, edgecolor=edge_color,
                                              linewidth=0.5, alpha=0.9)
                    else:
                        triangle = plt.Polygon([(x + 0.5, y + 1), (x, y), (x + 1, y)],
                                              facecolor=color, alpha=0.9)
                    ax.add_patch(triangle)
                    
                else:
                    # Orbes avec effet de glow
                    # Glow extérieur
                    for r in range(3, 0, -1):
                        glow = plt.Circle((x + 0.5, y + 0.5), 0.3 + r * 0.1,
                                         facecolor=color, alpha=0.1)
                        ax.add_patch(glow)
                    # Orbe principal
                    circle = plt.Circle((x + 0.5, y + 0.5), 0.35,
                                        facecolor=color, edgecolor='white',
                                        linewidth=1, alpha=0.9)
                    ax.add_patch(circle)
                    # Reflet
                    highlight = plt.Circle((x + 0.4, y + 0.6), 0.1,
                                          facecolor='white', alpha=0.4)
                    ax.add_patch(highlight)
                    
        except Exception as e:
            print(f"Erreur lors du tracé du layout: {e}")
