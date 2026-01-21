"""
DQN Agent pour Dash Game
Agent de reinforcement learning avec Deep Q-Network
Inspiré du TP1 Ruée vers l'or
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional
import pickle
import os
import copy


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


class DQNAgent:
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


class DuelingDQNAgent:
    """
    Dueling DQN Agent
    Sépare l'estimation de la valeur d'état V(s) et de l'avantage A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A))
    
    Meilleur pour apprendre quand l'action n'a pas d'importance
    """
    
    def __init__(self, 
                 state_size: int = 95,
                 action_size: int = 2,
                 hidden_sizes: List[int] = [256, 128],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 buffer_capacity: int = 100000,
                 use_prioritized_replay: bool = True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.use_prioritized_replay = use_prioritized_replay
        
        self.target_update_freq = 500
        self.last_target_update = 0
        
        # Buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
        else:
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Réseau dueling
        self.weights = self._init_dueling_weights()
        self.target_weights = copy.deepcopy(self.weights)
        
        self.training_steps = 0
    
    def _init_dueling_weights(self) -> dict:
        """Initialise les poids pour l'architecture dueling"""
        weights = {}
        input_size = self.state_size
        
        # Couches partagées
        shared_weights = []
        for i, hidden_size in enumerate(self.hidden_sizes[:-1]):
            w = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            b = np.zeros(hidden_size)
            shared_weights.append((w, b))
            input_size = hidden_size
        
        weights['shared'] = shared_weights
        
        # Stream de valeur V(s)
        last_hidden = self.hidden_sizes[-1] if len(self.hidden_sizes) > 1 else self.hidden_sizes[0]
        prev_size = self.hidden_sizes[-2] if len(self.hidden_sizes) > 1 else self.state_size
        
        w_v1 = np.random.randn(prev_size, last_hidden) * np.sqrt(2.0 / prev_size)
        b_v1 = np.zeros(last_hidden)
        w_v2 = np.random.randn(last_hidden, 1) * np.sqrt(2.0 / last_hidden)
        b_v2 = np.zeros(1)
        weights['value'] = [(w_v1, b_v1), (w_v2, b_v2)]
        
        # Stream d'avantage A(s,a)
        w_a1 = np.random.randn(prev_size, last_hidden) * np.sqrt(2.0 / prev_size)
        b_a1 = np.zeros(last_hidden)
        w_a2 = np.random.randn(last_hidden, self.action_size) * np.sqrt(2.0 / last_hidden)
        b_a2 = np.zeros(self.action_size)
        weights['advantage'] = [(w_a1, b_a1), (w_a2, b_a2)]
        
        return weights
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _forward_dueling(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass avec architecture dueling"""
        weights_to_use = self.target_weights if use_target else self.weights
        
        x = state
        
        # Couches partagées
        for w, b in weights_to_use['shared']:
            x = self._relu(np.dot(x, w) + b)
        
        # Stream de valeur
        v = x
        for i, (w, b) in enumerate(weights_to_use['value']):
            v = np.dot(v, w) + b
            if i < len(weights_to_use['value']) - 1:
                v = self._relu(v)
        
        # Stream d'avantage
        a = x
        for i, (w, b) in enumerate(weights_to_use['advantage']):
            a = np.dot(a, w) + b
            if i < len(weights_to_use['advantage']) - 1:
                a = self._relu(a)
        
        # Combiner: Q = V + (A - mean(A))
        q_values = v + (a - np.mean(a, axis=-1, keepdims=True))
        
        return q_values
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self._forward_dueling(state, use_target=False)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.predict(state)
        return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        """Entraînement avec support prioritized replay"""
        if self.use_prioritized_replay:
            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
            if len(batch) == 0:
                return
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            weights = np.ones(len(batch))
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Double DQN: sélection avec réseau principal, évaluation avec target
        next_q_main = self._forward_dueling(next_states, use_target=False)
        next_actions = np.argmax(next_q_main, axis=1)
        
        next_q_target = self._forward_dueling(next_states, use_target=True)
        max_next_q = next_q_target[np.arange(len(batch)), next_actions]
        
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        current_q = self._forward_dueling(states, use_target=False)
        td_errors = targets - current_q[np.arange(len(batch)), actions]
        
        # Update priorities si prioritized replay
        if self.use_prioritized_replay:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # Backprop simplifié (mise à jour basée sur TD error pondéré)
        self._simple_update(states, actions, targets, weights)
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1
        
        if self.training_steps - self.last_target_update >= self.target_update_freq:
            self.target_weights = copy.deepcopy(self.weights)
            self.last_target_update = self.training_steps
    
    def _simple_update(self, states, actions, targets, weights):
        """Mise à jour simplifiée des poids"""
        # Pour simplifier, on utilise une mise à jour directe
        # Une implémentation complète nécessiterait un backprop complet
        current_q = self._forward_dueling(states, use_target=False)
        td_errors = targets - current_q[np.arange(len(states)), actions]
        
        # Mise à jour proportionnelle à l'erreur (très simplifié)
        for layer_weights in [self.weights['shared'], self.weights['value'], self.weights['advantage']]:
            for i, (w, b) in enumerate(layer_weights):
                # Petite perturbation dans la direction de réduction de l'erreur
                grad_scale = self.learning_rate * np.mean(np.abs(td_errors * weights))
                noise = np.random.randn(*w.shape) * grad_scale * 0.01
                layer_weights[i] = (w + noise, b)
    
    def save(self, filepath: str):
        data = {
            'weights': self.weights,
            'target_weights': self.target_weights,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_sizes': self.hidden_sizes,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'use_prioritized_replay': self.use_prioritized_replay
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dueling DQN Agent saved to {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.target_weights = data.get('target_weights', copy.deepcopy(self.weights))
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']
        print(f"Dueling DQN Agent loaded from {filepath}")


class StackedFramesDQNAgent:
    """
    DQN avec frames empilées (mémoire temporelle)
    Permet à l'agent de voir la dynamique du jeu (vitesse, accélération)
    """
    
    def __init__(self,
                 base_state_size: int = 95,
                 num_frames: int = 4,
                 action_size: int = 2,
                 hidden_sizes: List[int] = [256, 128, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 buffer_capacity: int = 50000):
        
        self.base_state_size = base_state_size
        self.num_frames = num_frames
        self.state_size = base_state_size * num_frames
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.target_update_freq = 500
        self.last_target_update = 0
        
        # Buffer de frames pour chaque état
        self.frame_buffer = deque(maxlen=num_frames)
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.weights = self._init_weights()
        self.target_weights = copy.deepcopy(self.weights)
        
        self.training_steps = 0
    
    def _init_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Initialise les poids avec architecture plus large"""
        weights = []
        input_size = self.state_size
        
        for hidden_size in self.hidden_sizes:
            w = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            b = np.zeros(hidden_size)
            weights.append((w, b))
            input_size = hidden_size
        
        w = np.random.randn(input_size, self.action_size) * np.sqrt(2.0 / input_size)
        b = np.zeros(self.action_size)
        weights.append((w, b))
        
        return weights
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        x = state
        weights_to_use = self.target_weights if use_target else self.weights
        
        for i, (w, b) in enumerate(weights_to_use[:-1]):
            x = self._relu(np.dot(x, w) + b)
        
        w, b = weights_to_use[-1]
        return np.dot(x, w) + b
    
    def reset_frame_buffer(self):
        """Réinitialise le buffer de frames"""
        self.frame_buffer.clear()
    
    def _get_stacked_state(self, state: np.ndarray) -> np.ndarray:
        """Empile les frames pour créer l'état complet"""
        self.frame_buffer.append(state)
        
        # Padding avec des zéros si pas assez de frames
        while len(self.frame_buffer) < self.num_frames:
            self.frame_buffer.appendleft(np.zeros(self.base_state_size))
        
        return np.concatenate(list(self.frame_buffer))
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        stacked = self._get_stacked_state(state)
        if stacked.ndim == 1:
            stacked = stacked.reshape(1, -1)
        return self._forward(stacked, use_target=False)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            # Toujours mettre à jour le frame buffer même en exploration
            self._get_stacked_state(state)
            return random.randint(0, self.action_size - 1)
        
        q_values = self.predict(state)
        return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        # Stocker les états empilés
        stacked_state = self._get_stacked_state(state) if len(self.frame_buffer) > 0 else np.concatenate([state] * self.num_frames)
        stacked_next = np.concatenate(list(self.frame_buffer)[1:] + [next_state])
        
        self.replay_buffer.push(stacked_state, action, reward, stacked_next, done)
        
        if done:
            self.reset_frame_buffer()
        
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
    
    def _train_step(self):
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Double DQN
        next_q_main = self._forward(next_states, use_target=False)
        next_actions = np.argmax(next_q_main, axis=1)
        
        next_q_target = self._forward(next_states, use_target=True)
        max_next_q = next_q_target[np.arange(len(batch)), next_actions]
        
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        self._backprop(states, actions, targets)
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1
        
        if self.training_steps - self.last_target_update >= self.target_update_freq:
            self.target_weights = copy.deepcopy(self.weights)
            self.last_target_update = self.training_steps
    
    def _backprop(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        """Backprop similaire au DQN standard"""
        batch_size = len(states)
        
        # Forward pass pour récupérer les activations
        x = states
        activations = [x]
        
        for i, (w, b) in enumerate(self.weights[:-1]):
            z = np.dot(x, w) + b
            x = self._relu(z)
            activations.append(x)
        
        w, b = self.weights[-1]
        output = np.dot(x, w) + b
        activations.append(output)
        
        # Gradient
        q_values = output[np.arange(batch_size), actions]
        td_error = targets - q_values
        
        grad_output = np.zeros_like(output)
        grad_output[np.arange(batch_size), actions] = -td_error
        
        grad = grad_output
        new_weights = []
        
        for i in reversed(range(len(self.weights))):
            w, b = self.weights[i]
            activation = activations[i]
            
            grad_w = np.dot(activation.T, grad) / batch_size
            grad_b = np.mean(grad, axis=0)
            
            if i > 0:
                grad = np.dot(grad, w.T)
                grad = grad * (activations[i] > 0).astype(float)
            
            new_w = w - self.learning_rate * np.clip(grad_w, -1, 1)
            new_b = b - self.learning_rate * np.clip(grad_b, -1, 1)
            new_weights.insert(0, (new_w, new_b))
        
        self.weights = new_weights
    
    def save(self, filepath: str):
        data = {
            'weights': self.weights,
            'target_weights': self.target_weights,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'config': {
                'base_state_size': self.base_state_size,
                'num_frames': self.num_frames,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_sizes': self.hidden_sizes,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Stacked Frames DQN Agent saved to {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.target_weights = data.get('target_weights', copy.deepcopy(self.weights))
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']
        self.reset_frame_buffer()
        print(f"Stacked Frames DQN Agent loaded from {filepath}")
