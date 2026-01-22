"""
DQN Agent pour Dash Game
Agent de reinforcement learning avec Deep Q-Network
Inspiré du TP1 Ruée vers l'or
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque
from typing import List, Tuple

# Détection du device (GPU/CPU) [cite: 121]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Support pour Mac M1/M2 si disponible
if torch.backends.mps.is_available():
    device = torch.device("mps")

class DQN(nn.Module):
    """Réseau de neurones pour approximer la Q-value [cite: 19]"""
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(DQN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Construction dynamique des couches cachées
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU()) # [cite: 122]
            prev_dim = hidden_size
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, 
                 state_size: int = 95, 
                 action_size: int = 2,
                 hidden_sizes: List[int] = [256, 128], # Augmenté pour plus de capacité
                 learning_rate: float = 0.0005, # Réduit pour stabilité
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64, # [cite: 118]
                 buffer_capacity: int = 50000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = 500  # Fréquence de mise à jour du target net
        
        # Initialisation des réseaux (Policy et Target) 
        self.policy_net = DQN(state_size, action_size, hidden_sizes).to(device)
        self.target_net = DQN(state_size, action_size, hidden_sizes).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Copie initiale
        self.target_net.eval() # Le target net n'est pas entraîné directement
        
        # Optimiseur et Loss [cite: 125, 126]
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss() # ou nn.SmoothL1Loss() pour plus de robustesse
        
        # Replay Buffer (Deques sont très rapides en Python pour push/pop)
        self.memory = deque(maxlen=buffer_capacity)
        self.training_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-Greedy selection [cite: 371]"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Conversion en tenseur PyTorch sans calcul de gradient [cite: 128]
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        """Stockage de la transition et déclenchement de l'apprentissage"""
        # Stocker sous forme de tuples simples pour économiser la RAM
        self.memory.append((state, action, reward, next_state, done))
        
        # Décroissance d'Epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        # Apprentissage si assez de données
        if len(self.memory) >= self.batch_size:
            self._train_step()

    def _train_step(self):
        # 1. Échantillonnage aléatoire (Experience Replay) 
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Conversion rapide en tenseurs GPU
        state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

        # 2. Calcul des Q-values courantes Q(s, a)
        # gather récupère la Q-value correspondant à l'action prise
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # 3. Calcul des Q-values cibles (Bellman Optimality Equation) [cite: 450]
        # V*(s) = max_a Q*(s,a) avec le Target Network pour la stabilité
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            # Si done est True, le futur reward est 0
            target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # 4. Calcul de la perte et Gradient Descent [cite: 126]
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad() # Reset gradients
        loss.backward()            # Backpropagation
        # Gradient clipping pour éviter l'explosion des gradients (optionnel mais recommandé)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()      # Mise à jour des poids

        self.training_steps += 1

        # 5. Mise à jour périodique du Target Network 
        if self.training_steps % self.target_update_freq == 0:
            self.target_weights = copy.deepcopy(self.policy_net.state_dict())
            self.target_net.load_state_dict(self.target_weights)

    def save(self, filepath: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.training_steps
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('steps', 0)


class ReplayBuffer:
    """Buffer de replay pour l'apprentissage par experience replay"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Ajoute une transition au buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Échantillonne un batch aléatoire"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent_old:
    """
    Agent DQN pour le jeu Dash
    Utilise un réseau de neurones simple pour approximer la Q-fonction
    """
    
    def __init__(self, 
                 state_size: int = 95,  # 5 + 30 * 3
                 action_size: int = 2,
                 hidden_sizes: List[int] = [128, 64],
                 learning_rate: float = 0.003,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.015,
                 epsilon_decay: float = 0.999995,
                 batch_size: int = 128,
                 buffer_capacity: int = 100000):
        """
        Initialise l'agent DQN
        
        Args:
            state_size: Taille du vecteur d'observation
            action_size: Nombre d'actions possibles (2: jump/no-jump)
            hidden_sizes: Tailles des couches cachées
            learning_rate: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon_start: Epsilon initial pour exploration
            epsilon_end: Epsilon minimum
            epsilon_decay: Facteur de décroissance d'epsilon
            batch_size: Taille des batchs d'apprentissage
            buffer_capacity: Capacité du replay buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Target Network (Stability improvement)
        # On garde une copie des poids qui ne change pas à chaque étape
        self.target_update_freq = 1000  # Mise à jour du target network tous les 1000 steps
        self.last_target_update = 0
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Réseau de neurones (implémentation simple numpy)
        self.weights = self._init_weights()
        self.target_weights = copy.deepcopy(self.weights)
        
        # Statistiques
        self.training_steps = 0
        self.episode_rewards = []
    
    def _init_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Initialise les poids du réseau"""
        weights = []
        input_size = self.state_size
        
        for hidden_size in self.hidden_sizes:
            # Xavier initialization
            w = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            b = np.zeros(hidden_size)
            weights.append((w, b))
            input_size = hidden_size
        
        # Couche de sortie
        w = np.random.randn(input_size, self.action_size) * np.sqrt(2.0 / input_size)
        b = np.zeros(self.action_size)
        weights.append((w, b))
        
        return weights
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Fonction ReLU"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Dérivée de ReLU"""
        return (x > 0).astype(float)
    
    def _forward(self, state: np.ndarray, use_target: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Propagation avant
        
        Args:
            state: État d'entrée
            use_target: Si True, utilise les poids du target network
            
        Returns:
            output: Q-values pour chaque action
            activations: Activations de chaque couche (pour backprop)
        """
        x = state
        activations = [x]
        
        weights_to_use = self.target_weights if use_target else self.weights
        
        for i, (w, b) in enumerate(weights_to_use[:-1]):
            z = np.dot(x, w) + b
            x = self._relu(z)
            activations.append(x)
        
        # Couche de sortie (pas d'activation)
        w, b = weights_to_use[-1]
        output = np.dot(x, w) + b
        activations.append(output)
        
        return output, activations
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Prédit les Q-values pour un état (utilise le réseau prinicpal)"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        q_values, _ = self._forward(state, use_target=False)
        return q_values
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Sélectionne une action avec epsilon-greedy
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy
            
        Returns:
            Action sélectionnée (0 ou 1)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.predict(state)
        return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """
        Mise à jour de l'agent avec une transition
        
        Args:
            state: État initial
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: True si épisode terminé
        """
        # Ajouter au replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Apprendre si assez d'échantillons
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """Effectue un pas d'entraînement (Double DQN + Target Network)"""
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # 1. Sélection des meilleures actions pour l'état suivant avec le réseau PRINCIPAL (Double DQN)
        next_q_main, _ = self._forward(next_states, use_target=False)
        next_actions = np.argmax(next_q_main, axis=1)
        
        # 2. Évaluation de la valeur de ces actions avec le réseau CIBLE (Target Network)
        next_q_target, _ = self._forward(next_states, use_target=True)
        
        # On prend la Q-value correspondant à l'action choisie par le réseau principal
        max_next_q = next_q_target[np.arange(self.batch_size), next_actions]
        
        # Calcul des targets (Bellman equation)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 3. Calcul des prédictions actuelles (réseau PRINCIPAL) et backprop
        current_q, activations = self._forward(states, use_target=False)
        self._backprop(states, actions, targets, activations)
        
        # Mettre à jour epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1
        
        # Mise à jour du target network périodiquement
        if self.training_steps - self.last_target_update >= self.target_update_freq:
            self.target_weights = copy.deepcopy(self.weights)
            self.last_target_update = self.training_steps
    
    def _backprop(self, states: np.ndarray, actions: np.ndarray, 
                  targets: np.ndarray, activations: List[np.ndarray]):
        """Rétropropagation du gradient"""
        batch_size = len(states)
        
        # Gradient de la couche de sortie
        output = activations[-1]
        q_values = output[np.arange(batch_size), actions]
        
        # Erreur TD
        td_error = targets - q_values
        
        # Gradient initial (seulement pour l'action choisie)
        grad_output = np.zeros_like(output)
        grad_output[np.arange(batch_size), actions] = -td_error
        
        # Backprop à travers les couches
        grad = grad_output
        new_weights = []
        
        for i in reversed(range(len(self.weights))):
            w, b = self.weights[i]
            activation = activations[i]
            
            # Gradient des poids
            grad_w = np.dot(activation.T, grad) / batch_size
            grad_b = np.mean(grad, axis=0)
            
            # Gradient pour la couche précédente
            if i > 0:
                grad = np.dot(grad, w.T)
                # Appliquer dérivée ReLU
                grad = grad * self._relu_derivative(activations[i])
            
            # Mise à jour des poids
            new_w = w - self.learning_rate * grad_w
            new_b = b - self.learning_rate * grad_b
            new_weights.insert(0, (new_w, new_b))
        
        self.weights = new_weights
    
    def decay_epsilon(self):
        """Décroît epsilon manuellement"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Sauvegarde l'agent"""
        data = {
            'weights': self.weights,
            'target_weights': self.target_weights,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'last_target_update': self.last_target_update,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_sizes': self.hidden_sizes,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Charge l'agent depuis un fichier"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.weights = data['weights']
        # Compatibilité ascendante pour les anciens checkpoints
        self.target_weights = data.get('target_weights', copy.deepcopy(self.weights))
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']
        self.last_target_update = data.get('last_target_update', self.training_steps)
        print(f"Agent loaded from {filepath}")


class EpsilonGreedyAgent:
    """
    Agent simple epsilon-greedy (comme dans le TP1)
    Garde en mémoire les Q-values moyennes pour chaque état-action
    """
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.q_values = {}  # (state_hash, action) -> [rewards]
        self.action_size = 2
    
    def _state_to_hash(self, state: np.ndarray) -> str:
        """Convertit un état en hash pour stockage"""
        # Discrétiser l'état pour réduire l'espace
        discretized = (state * 10).astype(int)
        return str(discretized.tobytes())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Sélectionne une action epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_hash = self._state_to_hash(state)
        
        # Trouver la meilleure action
        best_action = 0
        best_value = float('-inf')
        
        for action in range(self.action_size):
            key = (state_hash, action)
            if key in self.q_values and len(self.q_values[key]) > 0:
                value = np.mean(self.q_values[key])
            else:
                value = 0.0
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """Met à jour les Q-values"""
        state_hash = self._state_to_hash(state)
        key = (state_hash, action)
        
        if key not in self.q_values:
            self.q_values[key] = []
        
        self.q_values[key].append(reward)
        
        # Garder seulement les N dernières récompenses
        if len(self.q_values[key]) > 100:
            self.q_values[key] = self.q_values[key][-100:]

    def save(self, filepath: str):
        """Sauvegarde l'agent"""
        data = {
            'q_values': self.q_values,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Charge l'agent depuis un fichier"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_values = data['q_values']
        self.epsilon = data['epsilon']
        print(f"Agent loaded from {filepath}")


class RandomAgent:
    """Agent aléatoire pour baseline"""
    
    def __init__(self, jump_probability: float = 0.3):
        self.jump_probability = jump_probability
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Sélectionne une action aléatoire"""
        return 1 if random.random() < self.jump_probability else 0
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """L'agent aléatoire n'apprend pas"""
        pass


class PrioritizedReplayBuffer:
    """
    Buffer avec priorité basée sur l'erreur TD
    Les transitions avec une grande erreur sont échantillonnées plus souvent
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta_start  # Importance sampling weight
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Ajoute une transition avec priorité maximale"""
        transition = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Échantillonne selon les priorités"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculer les probabilités
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Échantillonner
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                   p=probs, replace=False)
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Augmenter beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Met à jour les priorités basées sur l'erreur TD"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        return len(self.buffer)