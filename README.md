# Dash Reinforcement Learning Agent

L'objectif de ce projet est de créer un jeu similaire à « Geometry Dash » et d'y entraîner un agent d'apprentissage par renforcement. Nous utiliserons différentes méthodes d'agentage afin d'obtenir le meilleur agent possible pour ce type de jeu.
**Le notebook `report.ipynb`** permet d'explorer facilement l'ensemble du travail réalisé : implémentation des agents, entraînement, résultats et analyses comparatives.
## Structure du projet

```
Dash_reinforcment_learning_agent/
│   report.ipynb
├── Dash_Game/              # Le jeu Geometry Dash
│   ├── main.py             # Point d'entrée du jeu manuel
│   ├── src/
│   │   ├── game.py         # Classe principale du jeu
│   │   ├── game_env.py     # NEW Environnement RL (headless)
│   │   ├── replay_viewer.py# NEW Visualiseur de replays
│   │   ├── physics.py      # Moteur physique
│   │   ├── layout.py       # Gestion des niveaux
│   │   ├── renderer.py     # Rendu graphique
│   │   └── velocity.py     # Gestion de la vélocité
│   ├── layouts/            # Fichiers CSV des niveaux
│   └── assets/             # Images et ressources
│
└── Rl_agents/              # Agents RL
    ├── main.py             # Script principal d'entraînement
    ├── demo.py             # Démonstration rapide
    ├── agents.py           # Implémentations des agents (DQN, etc.)
    └── visualizer.py       # Outils de visualisation
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Jouer manuellement au jeu

```bash
cd Dash_Game
python main.py
```
- Touches 1-6 pour choisir le niveau
- ESPACE ou clic pour sauter

### 2. Démo rapide de l'entraînement

```bash
cd Rl_agents
python demo.py
```

### 3. Entraînement complet

```bash
cd Rl_agents

# Entraîner un agent DQN
python main.py --mode train --agent dqn --episodes 1000

# Fait un greed search pour trouver les meilleurs paramètres pour dqn
python main.py --grid-search --level layouts/level1.csv

# Entraîner avec un niveau spécifique
python main.py --mode train --level layouts/level2.csv --episodes 2000

# Évaluer un agent sauvegardé
python main.py --mode eval --load checkpoints/agent_final.pkl

# Regarder un replay
python main.py --mode watch --replay best_replay.vis

# Comparer les agents
python main.py --mode compare --episodes 500

```

### 4. Tester l'environnement

```bash
cd Rl_agents
python demo.py --test
```

## Environnement RL

### Observation (vecteur de 65 dimensions)
- Position Y du joueur (normalisée)
- Vélocité verticale
- Au sol (0/1)
- Gravité inversée (0/1)
- Progression dans le niveau
- 20 objets proches × 3 valeurs (type, relative_x, relative_y)

### Actions
- 0: Ne pas sauter
- 1: Sauter

### Récompenses
- +100: Victoire (fin du niveau)
- -10: Mort
- +0.1: Avancer
- +0.05: Bonus stabilité (au sol)

### Types d'objets détectés
| ID | Type |
|----|------|
| 1 | Block |
| 2 | Spike (haut) |
| 3 | Spike (droite) |
| 4 | Spike (bas) |
| 5 | Spike (gauche) |
| 6 | Orbe jaune |
| 7 | Orbe rose |
| 8 | Orbe bleu |
| 9 | Orbe vert |
| 10 | Orbe rouge |
| 11 | Orbe noir |

## Agents disponibles

### DQNAgent
Réseau de neurones profond avec experience replay.
```python
agent = DQNAgent(
    state_size=65,
    hidden_sizes=[128, 64],
    learning_rate=0.001,
    gamma=0.99,
    epsilon_decay=0.9995
)
```

### EpsilonGreedyAgent
Agent simple inspiré du TP1.

### RandomAgent
Baseline aléatoire.

## Système de replay

Les parties peuvent être enregistrées et rejouées:
```python
from src.game_env import DashGameEnv
from src.replay_viewer import ReplayViewer

# Enregistrer une partie
env = DashGameEnv("layouts/level1.csv", record_replay=True)
# ... jouer ...
frames = env.get_replay()

# Rejouer
viewer = ReplayViewer("layouts/level1.csv")
viewer.play_replay(frames)
```

## Entraînement accéléré

L'environnement headless permet d'entraîner ~1000 épisodes/minute sans affichage graphique. Les replays sont sauvegardés automatiquement (1/200 parties par défaut).

## Crédits

Base du jeu: 
- https://github.com/pythagon-code/Dash
- https://github.com/OJddJO/numworks-dash