"""
ReplayViewer - Visualiseur de replays pour les parties enregistrées
Permet de rejouer des parties d'entraînement avec l'interface graphique
"""

import pygame
import time
from typing import List, Optional
from dataclasses import dataclass

from src.layout import DashLayout, DashObject, DashObjectType
from src.layout_reader import DashLayoutReader
from src.game_env import ReplayFrame


class ReplayViewer:
    """Visualiseur de replays avec interface graphique pygame"""
    
    SCREEN_DIMS = 960, 600
    BLOCK_SIZE = 36, 36
    ORB_SIZE = 40, 40
    
    def __init__(self, level_path: str = "layouts/level1.csv"):
        """
        Initialise le visualiseur
        
        Args:
            level_path: Chemin vers le niveau original
        """
        pygame.init()
        
        self.screen = pygame.display.set_mode(self.SCREEN_DIMS)
        self.background = pygame.image.load("assets/dash-background.png")
        pygame.display.set_caption("Dash - Replay Viewer")
        pygame.display.set_icon(pygame.image.load("assets/spike.png"))
        
        self.clock = pygame.time.Clock()
        
        # Charger les images
        self.cube = pygame.transform.scale(
            pygame.image.load("assets/cube.png"), self.BLOCK_SIZE)
        self.block = pygame.transform.scale(
            pygame.image.load("assets/block4.png"), self.BLOCK_SIZE)
        self.up_spike = pygame.transform.scale(
            pygame.image.load("assets/spike2.png"), self.BLOCK_SIZE)
        self.right_spike = pygame.transform.rotate(self.up_spike, -90)
        self.down_spike = pygame.transform.rotate(self.up_spike, 180)
        self.left_spike = pygame.transform.rotate(self.up_spike, 90)
        self.yellow_orb = pygame.transform.scale(
            pygame.image.load("assets/yellow-orb.png"), self.ORB_SIZE)
        self.pink_orb = pygame.transform.scale(
            pygame.image.load("assets/pink-orb.png"), self.ORB_SIZE)
        self.blue_orb = pygame.transform.scale(
            pygame.image.load("assets/blue-orb.png"), self.ORB_SIZE)
        self.green_orb = pygame.transform.scale(
            pygame.image.load("assets/green-orb.png"), self.ORB_SIZE)
        self.red_orb = pygame.transform.scale(
            pygame.image.load("assets/red-orb.png"), self.ORB_SIZE)
        self.black_orb = pygame.transform.scale(
            pygame.image.load("assets/black-orb.png"), self.ORB_SIZE)
        
        self.obj_to_image = {
            DashObjectType.BLOCK: self.block,
            DashObjectType.UP_SPIKE: self.up_spike,
            DashObjectType.RIGHT_SPIKE: self.right_spike,
            DashObjectType.DOWN_SPIKE: self.down_spike,
            DashObjectType.LEFT_SPIKE: self.left_spike,
            DashObjectType.YELLOW_ORB: self.yellow_orb,
            DashObjectType.PINK_ORB: self.pink_orb,
            DashObjectType.BLUE_ORB: self.blue_orb,
            DashObjectType.GREEN_ORB: self.green_orb,
            DashObjectType.RED_ORB: self.red_orb,
            DashObjectType.BLACK_ORB: self.black_orb
        }
        
        # Cubes rotatifs
        self.rotated_cubes = [
            pygame.transform.rotate(self.cube, -i) for i in range(360)]
        
        # Layout original
        self.level_path = level_path
        self.layout = DashLayoutReader.read_layout_from_csv(level_path)
        
        # Font pour afficher les infos
        self.font = pygame.font.SysFont('Arial', 16)
    
    def find_anchor_position(self, position: tuple, line: float, 
                            moves_horizontally: bool = True) -> tuple:
        """Calcule la position d'ancrage pour l'affichage"""
        effective_line = line if moves_horizontally else 0
        x = (position[0] - effective_line) * self.BLOCK_SIZE[0]
        y = self.SCREEN_DIMS[1] - position[1] * self.BLOCK_SIZE[1] - 189
        return x, y
    
    def render_frame(self, frame: ReplayFrame, frame_index: int, 
                     total_frames: int, rotation: int) -> None:
        """Affiche une frame du replay"""
        self.screen.blit(self.background, (0, 0))
        
        # Afficher les objets
        for obj in self.layout.objects:
            if obj.position[0] + frame.line > 1000:
                break
            obj_image = self.obj_to_image.get(obj.objectType)
            if obj_image:
                pos = self.find_anchor_position(obj.position, frame.line)
                self.screen.blit(obj_image, pos)
        
        # Afficher le joueur
        player_pos = self.find_anchor_position(
            frame.player_position, frame.line, moves_horizontally=False)
        cube_image = self.rotated_cubes[rotation % 360]
        
        # Retourner le cube si gravité inversée
        if frame.reversed_gravity:
            cube_image = pygame.transform.flip(cube_image, False, True)
        
        self.screen.blit(cube_image, player_pos)
        
        # Afficher les informations
        info_texts = [
            f"Frame: {frame_index}/{total_frames}",
            f"Position X: {frame.player_position[0] + frame.line:.2f}",
            f"Position Y: {frame.player_position[1]:.2f}",
            f"Velocity: {frame.player_velocity:.3f}",
            f"Action: {'JUMP' if frame.action == 1 else 'NONE'}",
            f"Gravity: {'REVERSED' if frame.reversed_gravity else 'NORMAL'}",
            f"Progress: {frame.progress*100:.1f}%",
            "",
            "Controls: SPACE=Pause, LEFT/RIGHT=Navigate, ESC=Quit"
        ]
        
        for i, text in enumerate(info_texts):
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 20))
        
        # Indicateur de saut (cercle vert si jump)
        if frame.action == 1:
            pygame.draw.circle(self.screen, (0, 255, 0), 
                             (self.SCREEN_DIMS[0] - 50, 50), 20)
    
    def play_replay(self, frames: List[ReplayFrame], speed: float = 1.0) -> None:
        """
        Joue un replay
        
        Args:
            frames: Liste des frames du replay
            speed: Vitesse de lecture (1.0 = normale, 2.0 = 2x plus rapide)
        """
        if not frames:
            print("Pas de frames à afficher")
            return
        
        # Recharger le layout
        self.layout = DashLayoutReader.read_layout_from_csv(self.level_path)
        
        frame_index = 0
        paused = False
        rotation = 0
        
        running = True
        while running and frame_index < len(frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_LEFT:
                        frame_index = max(0, frame_index - 10)
                    elif event.key == pygame.K_RIGHT:
                        frame_index = min(len(frames) - 1, frame_index + 10)
            
            current_frame = frames[frame_index]
            self.render_frame(current_frame, frame_index, len(frames), rotation)
            pygame.display.flip()
            
            if not paused:
                frame_index += 1
                rotation += 4  # Rotation du cube
            
            self.clock.tick(int(120 * speed))
        
        # Afficher la dernière frame un moment
        if running:
            time.sleep(1)
    
    def play_multiple_replays(self, replays: List[List[ReplayFrame]], 
                              speed: float = 1.0) -> None:
        """Joue plusieurs replays à la suite"""
        for i, replay in enumerate(replays):
            print(f"Playing replay {i+1}/{len(replays)}")
            self.play_replay(replay, speed)
    
    def close(self) -> None:
        """Ferme le visualiseur"""
        pygame.quit()


def view_replay_from_file(replay_file: str, level_path: str = "layouts/level1.csv"):
    """
    Charge et affiche un replay depuis un fichier
    
    Args:
        replay_file: Chemin vers le fichier de replay (pickle)
        level_path: Chemin vers le niveau
    """
    import pickle
    
    with open(replay_file, 'rb') as f:
        frames = pickle.load(f)
    
    viewer = ReplayViewer(level_path)
    viewer.play_replay(frames)
    viewer.close()


def save_replay_to_file(frames: List[ReplayFrame], filepath: str):
    """
    Sauvegarde un replay dans un fichier
    
    Args:
        frames: Liste des frames
        filepath: Chemin de sauvegarde
    """
    import pickle
    
    with open(filepath, 'wb') as f:
        pickle.dump(frames, f)
    print(f"Replay saved to {filepath}")


if __name__ == "__main__":
    # Test du visualiseur avec une partie manuelle
    from src.game_env import DashGameEnv
    import random
    
    print("Testing replay viewer with random agent...")
    
    env = DashGameEnv("layouts/level1.csv", record_replay=True)
    obs = env.reset()
    done = False
    
    while not done:
        action = random.choice([0, 0, 0, 1])  # Sauter 1 fois sur 4
        obs, reward, done, info = env.step(action)
    
    print(f"Episode finished: won={info['won']}, progress={info['progress']:.2%}")
    
    frames = env.get_replay()
    if frames:
        print(f"Recorded {len(frames)} frames")
        viewer = ReplayViewer("layouts/level1.csv")
        viewer.play_replay(frames)
        viewer.close()
