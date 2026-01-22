"""
DashGameEnv - Environnement de jeu pour l'entraînement RL
Version headless (sans graphiques) pour l'entraînement rapide
Avec support de replay pour visualiser certaines parties
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from collections import deque
import numpy as np

from src.layout import DashLayout, DashObject, DashObjectType
from src.layout_reader import DashLayoutReader
from src.velocity import DashVelocity


@dataclass
class ObjectInfo:
    """Information sur un objet proche du joueur"""
    object_type: int  # ID du type d'objet (1-11)
    relative_x: float  # Position X relative au joueur
    relative_y: float  # Position Y relative au joueur
    absolute_x: float  # Position X absolue
    absolute_y: float  # Position Y absolue


@dataclass
class GameState:
    """État complet du jeu pour l'agent RL"""
    player_x: float
    player_y: float
    player_velocity: float
    reversed_gravity: bool
    is_grounded: bool
    nearby_objects: List[ObjectInfo]
    progress: float  # Progression dans le niveau (0-1)
    

@dataclass 
class ReplayFrame:
    """Frame pour le système de replay"""
    player_position: Tuple[float, float]
    player_velocity: float
    reversed_gravity: bool
    line: float
    action: int  # 0: no jump, 1: jump
    progress: float  # Progression dans le niveau (0-1)
    objects_snapshot: List[Tuple[int, float, float]]  # [(type_id, x, y), ...]


class DashPhysicsHeadless:
    """Version sans graphiques de DashPhysics pour l'entraînement rapide"""
    
    GRAVITY = 1
    MAX_SPEED = 2
    GAME_SPEED = 2
    
    HITBOX_SPIKE = 0.7
    HITBOX_ORB = 0.75
    
    HITBOX_SIZE = {
        DashObjectType.BLOCK: 1,
        DashObjectType.UP_SPIKE: HITBOX_SPIKE,
        DashObjectType.RIGHT_SPIKE: HITBOX_SPIKE,
        DashObjectType.DOWN_SPIKE: HITBOX_SPIKE,
        DashObjectType.LEFT_SPIKE: HITBOX_SPIKE,
        DashObjectType.YELLOW_ORB: HITBOX_ORB,
        DashObjectType.PINK_ORB: HITBOX_ORB,
        DashObjectType.BLUE_ORB: HITBOX_ORB,
        DashObjectType.GREEN_ORB: HITBOX_ORB,
        DashObjectType.RED_ORB: HITBOX_ORB,
        DashObjectType.BLACK_ORB: HITBOX_ORB
    }
    
    REBOUND_FRAMES = 20
    JUMP_FORCE = 0.22
    YELLOW_ORB_FORCE = 0.25
    PINK_ORB_FORCE = 0.2
    BLUE_ORB_FORCE = 0.25
    GREEN_ORB_FORCE = 0.25
    RED_ORB_FORCE = 0.32
    BLACK_ORB_FORCE = 0.4
    
    def __init__(self, layout: DashLayout) -> None:
        self.layout = layout
        self.cube_position = (7.5, 0)
        self.cube_velocity = DashVelocity()
        self.line = -10
        self.progress = 0.0
        self.died = False
        self.won = False
        self.rebound = 0
        self.falling = False
        self.just_jump_frames = 0
        
        # Calculer la fin du niveau basée sur le dernier objet
        # Le player commence à x=7.5 avec line=-10, donc position initiale = -2.5
        # On utilise la position du dernier objet comme fin
        if len(layout.objects) > 0:
            self.level_end_x = max(obj.position[0] for obj in layout.objects)
        else:
            self.level_end_x = 100.0
        
        # Position de départ du joueur dans le monde
        self.start_x = self.cube_position[0] + self.line  # = -2.5 initialement
        
    def is_cube_touching(self, obj: DashObject) -> bool:
        touch_distance = self.HITBOX_SIZE[obj.objectType]
        real_dist_x = abs(self.cube_position[0] + self.line - obj.position[0])
        real_dist_y = abs(self.cube_position[1] - obj.position[1])
        return real_dist_x < touch_distance and real_dist_y < touch_distance
    
    def is_cube_grounded(self, obj: DashObject) -> bool:
        if obj.objectType != DashObjectType.BLOCK:
            return False
        grounded_distance = 1.01
        threshold = 0.4
        lower_bound = grounded_distance - threshold
        above = (self.cube_position[1] > obj.position[1]) ^ self.cube_velocity.reversed_gravity
        real_dist_x = abs(self.cube_position[0] + self.line - obj.position[0])
        real_dist_y = abs(self.cube_position[1] - obj.position[1])
        return above and real_dist_x < grounded_distance and lower_bound < real_dist_y <= grounded_distance
    
    def get_nearby_objects(self, range_ahead: float = 6.0, range_behind: float = 1.5, range_vertical: float = 6.0) -> List[ObjectInfo]:
        """Récupère les objets proches du joueur"""
        nearby = []
        player_world_x = self.cube_position[0] + self.line
        player_y = self.cube_position[1]

        for obj in self.layout.objects:
            obj_x = obj.position[0]
            obj_y = obj.position[1]

            # Distance relative
            rel_x = obj_x - player_world_x
            rel_y = obj_y - player_y

            # Filtrer les objets dans la zone de détection
            if -range_behind <= rel_x <= range_ahead and abs(rel_y) <= range_vertical:
                nearby.append(ObjectInfo(
                    object_type=obj.objectType.value,
                    relative_x=rel_x,
                    relative_y=rel_y,
                    absolute_x=obj_x,
                    absolute_y=obj_y
                ))

        # Trier par distance X (les plus proches d'abord)
        nearby.sort(key=lambda o: o.relative_x)
        return nearby
    
    def get_game_state(self) -> GameState:
        """Retourne l'état complet du jeu pour l'agent"""
        nearby = self.get_nearby_objects()
        
        # Calculer la progression basée sur la position réelle du joueur
        player_world_x = self.cube_position[0] + self.line
        
        # Progress = (position actuelle - position départ) / (fin niveau - position départ)
        total_distance = self.level_end_x - self.start_x
        if total_distance > 0:
            current_distance = player_world_x - self.start_x
            self.progress = max(0.0, min(1.0, current_distance / total_distance))
        else:
            self.progress = 1.0 if self.won else 0.0
        
        return GameState(
            player_x=self.cube_position[0] + self.line,
            player_y=self.cube_position[1],
            player_velocity=self.cube_velocity.velocity,
            reversed_gravity=self.cube_velocity.reversed_gravity,
            is_grounded=not self.falling,
            nearby_objects=nearby,
            progress=self.progress
        )
    
    def step(self, jump: bool, just_jump: bool = False) -> Tuple[bool, bool]:
        """
        Exécute un pas de simulation
        Retourne (died, won)
        """
        if self.died or self.won:
            return self.died, self.won
            
        # Vérifier si gagné
        if len(self.layout.objects) == 0 or \
           (len(self.layout.objects) > 0 and self.layout.objects[-1].position[0] - self.line <= 0):
            self.won = True
            return False, True
        
        # Vérifier les limites
        if self.cube_position[1] > 20 or (self.cube_position[1] <= 0 and self.cube_velocity.reversed_gravity):
            self.died = True
            return True, False
        
        # Nettoyer les objets passés
        while self.layout.should_remove_leftmost_object(self.line):
            self.layout.remove_leftmost_object()
        
        self.falling = self.cube_position[1] != 0 or self.cube_velocity.reversed_gravity
        
        # Mise à jour position
        new_y = max(0, self.cube_position[1] + self.cube_velocity.velocity)
        self.cube_position = (self.cube_position[0], new_y)
        
        # Gestion just_jump interne
        if jump and self.just_jump_frames <= 0:
            self.just_jump_frames = 20
        self.just_jump_frames -= 1
        actual_just_jump = just_jump or self.just_jump_frames > 0
        
        # Collisions et interactions
        for obj in self.layout.objects:
            if not DashLayout.should_display_object(obj, self.line):
                break
            
            # Spikes
            if obj.objectType in [DashObjectType.UP_SPIKE, DashObjectType.RIGHT_SPIKE, 
                                   DashObjectType.DOWN_SPIKE, DashObjectType.LEFT_SPIKE]:
                if self.is_cube_touching(obj):
                    self.died = True
                    return True, False
                    
            # Blocks
            elif obj.objectType == DashObjectType.BLOCK:
                if self.is_cube_grounded(obj):
                    self.falling = False
                    self.cube_velocity.set_speed(0)
                    unit = -1 if self.cube_velocity.reversed_gravity else 1
                    self.cube_position = (self.cube_position[0], obj.position[1] + unit)
                    self.rebound = 0
                elif self.is_cube_touching(obj):
                    self.died = True
                    return True, False
                    
            # Orbs
            elif self.rebound == 0 and actual_just_jump and self.is_cube_touching(obj):
                if obj.objectType == DashObjectType.YELLOW_ORB:
                    self.cube_velocity.set_speed(self.YELLOW_ORB_FORCE)
                    self.rebound = self.REBOUND_FRAMES
                elif obj.objectType == DashObjectType.PINK_ORB:
                    self.cube_velocity.set_speed(self.PINK_ORB_FORCE)
                    self.rebound = self.REBOUND_FRAMES
                elif obj.objectType == DashObjectType.BLUE_ORB:
                    self.cube_velocity.reverse_gravity()
                    self.rebound = self.REBOUND_FRAMES
                elif obj.objectType == DashObjectType.GREEN_ORB:
                    self.cube_velocity.reverse_gravity()
                    self.cube_velocity.set_speed(self.GREEN_ORB_FORCE)
                    self.rebound = self.REBOUND_FRAMES
                elif obj.objectType == DashObjectType.RED_ORB:
                    self.cube_velocity.set_speed(self.RED_ORB_FORCE)
                    self.rebound = self.REBOUND_FRAMES
                elif obj.objectType == DashObjectType.BLACK_ORB:
                    self.cube_velocity.set_speed(-self.BLACK_ORB_FORCE)
                    self.rebound = self.REBOUND_FRAMES
        
        # Gravité
        if self.falling:
            self.cube_velocity.fall()
        
        # Saut
        if jump and not self.falling and self.rebound == 0:
            self.cube_velocity.set_speed(0.22)
            self.rebound = self.REBOUND_FRAMES
        elif self.rebound > 0:
            self.rebound -= 1
        
        # Avancer
        self.line += 0.09173
        
        return False, False


class DashGameEnv:
    """
    Environnement de jeu compatible avec l'interface Gymnasium/OpenAI Gym
    Pour l'entraînement d'agents RL
    """
    
    # Nombre max d'objets dans l'observation
    # Augmenté pour voir plus loin et gérer les zones denses
    MAX_NEARBY_OBJECTS = 30
    
    # Actions
    ACTION_NONE = 0
    ACTION_JUMP = 1
    
    def __init__(self, level_path: str = "layouts/level1.csv", 
                 record_replay: bool = False,
                 max_steps: int = 10000):
        """
        Initialise l'environnement
        
        Args:
            level_path: Chemin vers le fichier CSV du niveau
            record_replay: Si True, enregistre les frames pour replay
            max_steps: Nombre maximum de steps avant timeout
        """
        self.level_path = level_path
        self.record_replay = record_replay
        self.max_steps = max_steps
        
        # Charger le layout original pour reset
        self.original_layout = DashLayoutReader.read_layout_from_csv(level_path)
        
        # État du jeu
        self.physics: Optional[DashPhysicsHeadless] = None
        self.current_step = 0
        self.replay_frames: List[ReplayFrame] = []
        
        # Statistiques
        self.total_episodes = 0
        self.total_wins = 0
        
        # Définition de l'espace d'observation
        # [player_y, velocity, is_grounded, reversed_gravity, progress,
        #  obj1_type, obj1_rel_x, obj1_rel_y, obj2_type, ...]
        self.observation_size = 5 + self.MAX_NEARBY_OBJECTS * 3
        
        self.reset()
    
    def _copy_layout(self) -> DashLayout:
        """Crée une copie du layout original"""
        original_objects = list(self.original_layout.objects)
        new_objects = [DashObject(obj.objectType, obj.position) for obj in original_objects]
        return DashLayout(new_objects)
    
    def reset(self) -> np.ndarray:
        """Reset l'environnement et retourne l'observation initiale"""
        layout = self._copy_layout()
        self.physics = DashPhysicsHeadless(layout)
        self.current_step = 0
        self.replay_frames = []
        self.total_episodes += 1
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Convertit l'état du jeu en vecteur d'observation"""
        state = self.physics.get_game_state()

        obs = np.zeros(self.observation_size, dtype=np.float32)

        # État du joueur
        obs[0] = state.player_y / 12.0  # Normalisé pour correspondre à range_vertical
        obs[1] = state.player_velocity / 0.4  # Normalisé par MAX_SPEED
        obs[2] = 1.0 if state.is_grounded else 0.0
        obs[3] = 1.0 if state.reversed_gravity else 0.0
        obs[4] = state.progress

        # Objets proches
        for i, obj in enumerate(state.nearby_objects[:self.MAX_NEARBY_OBJECTS]):
            base_idx = 5 + i * 3
            obs[base_idx] = obj.object_type / 11.0  # Normalisé (11 types)
            obs[base_idx + 1] = obj.relative_x / 9.0  # Normalisé pour correspondre à range_ahead
            obs[base_idx + 2] = obj.relative_y / 6.0  # Normalisé pour correspondre à range_vertical

        return obs
    
    def get_state_info(self) -> GameState:
        """Retourne l'état détaillé du jeu"""
        return self.physics.get_game_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécute une action et retourne (observation, reward, done, info)
        
        Args:
            action: 0 = ne pas sauter, 1 = sauter
            
        Returns:
            observation: Vecteur d'observation
            reward: Récompense
            done: True si l'épisode est terminé
            info: Informations supplémentaires
        """
        jump = action == self.ACTION_JUMP
        
        # Enregistrer frame si replay activé
        if self.record_replay:
            objects_snapshot = [(obj.object_type, obj.relative_x, obj.relative_y) 
                               for obj in self.physics.get_nearby_objects()[:10]]
            self.replay_frames.append(ReplayFrame(
                player_position=self.physics.cube_position,
                player_velocity=self.physics.cube_velocity.velocity,
                reversed_gravity=self.physics.cube_velocity.reversed_gravity,
                line=self.physics.line,
                action=action,
                progress=self.physics.progress,
                objects_snapshot=objects_snapshot
            ))
        
        # Exécuter le pas
        died, won = self.physics.step(jump)
        self.current_step += 1
        
        # Calculer la récompense
        reward = self._calculate_reward(died, won, jump)

        # Normalisation des récompenses (Clipping) [cite: 150]
        # Cela aide le réseau à ne pas déstabiliser ses poids face à de gros nombres
        reward = max(min(reward, 1.0), -1.0)
        
        # Vérifier si terminé
        done = died or won or self.current_step >= self.max_steps
        
        # Infos supplémentaires
        info = {
            'died': died,
            'won': won,
            'progress': self.physics.progress,
            'steps': self.current_step,
            'player_y': self.physics.cube_position[1]
        }
        
        if won:
            self.total_wins += 1
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, died: bool, won: bool, jumped: bool) -> float:
        """Reward Shaping """
        if won:
            return 1.0  # Victoire normalisée
        if died:
            return -1.0 # Mort normalisée
        
        # Récompense de survie/progression constante
        reward = 0.01 
        
        # Petite pénalité pour sauter afin d'encourager les sauts précis
        # et éviter le spamming inutile
        if jumped and not self.physics.falling:
             reward -= 0.005 
            
        return reward
    
    def get_replay(self) -> List[ReplayFrame]:
        """Retourne les frames enregistrées pour le replay"""
        return self.replay_frames
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques d'entraînement"""
        return {
            'total_episodes': self.total_episodes,
            'total_wins': self.total_wins,
            'win_rate': self.total_wins / max(1, self.total_episodes)
        }


class DashTrainingManager:
    """
    Gestionnaire d'entraînement pour lancer plusieurs parties en parallèle
    avec support de replay pour visualiser certaines parties
    """
    
    def __init__(self, level_path: str = "layouts/level1.csv",
                 replay_frequency: int = 100,
                 save_replays: bool = True):
        """
        Args:
            level_path: Chemin vers le niveau
            replay_frequency: Fréquence de sauvegarde des replays (1/N parties)
            save_replays: Si True, sauvegarde les replays
        """
        self.level_path = level_path
        self.replay_frequency = replay_frequency
        self.save_replays = save_replays
        self.saved_replays: List[List[ReplayFrame]] = []
        self.episode_count = 0
        
    def create_env(self, record_replay: bool = False) -> DashGameEnv:
        """Crée un nouvel environnement"""
        should_record = record_replay or (self.save_replays and 
                                          self.episode_count % self.replay_frequency == 0)
        env = DashGameEnv(self.level_path, record_replay=should_record)
        return env
    
    def run_episode(self, env: DashGameEnv, agent, training: bool = True) -> Dict:
        """
        Exécute un épisode complet
        
        Args:
            env: L'environnement de jeu
            agent: L'agent RL (doit avoir une méthode select_action et update)
            training: Si True, l'agent apprend
            
        Returns:
            Dict avec les statistiques de l'épisode
        """
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            if training and hasattr(agent, 'update'):
                agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_reward += reward
        
        self.episode_count += 1
        
        # Sauvegarder le replay si nécessaire
        if env.record_replay and self.save_replays:
            self.saved_replays.append(env.get_replay())
            # Garder seulement les N derniers replays
            if len(self.saved_replays) > 100:
                self.saved_replays.pop(0)
        
        return {
            'total_reward': total_reward,
            'steps': info['steps'],
            'won': info['won'],
            'died': info['died'],
            'progress': info['progress']
        }
    
    def get_replay(self, index: int = -1) -> Optional[List[ReplayFrame]]:
        """Récupère un replay sauvegardé"""
        if not self.saved_replays:
            return None
        return self.saved_replays[index]
