"""
Test de la décroissance d'epsilon pour différentes configurations
"""
import numpy as np
import matplotlib.pyplot as plt

def simulate_epsilon_decay(epsilon_start, epsilon_end, epsilon_decay, 
                          num_episodes, steps_per_episode, mode='step'):
    """Simule la décroissance d'epsilon"""
    epsilon_history = []
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        epsilon_history.append(epsilon)
        
        if mode == 'episode':
            # Décroît une fois par épisode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        else:  # mode == 'step'
            # Décroît à chaque training step
            for _ in range(steps_per_episode):
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    return epsilon_history


# Configurations à tester
configs = [
    # (decay, mode, label, color)
    (0.995, 'episode', 'Episode mode (0.995)', 'blue'),
    (0.997, 'episode', 'Episode mode (0.997)', 'cyan'),
    (0.999, 'episode', 'Episode mode (0.999)', 'green'),
    (0.999, 'step', 'Step mode (0.999) ~100 steps/ep', 'orange'),
    (0.9995, 'step', 'Step mode (0.9995) ~100 steps/ep - DÉFAUT', 'red'),
    (0.9998, 'step', 'Step mode (0.9998) ~100 steps/ep', 'purple'),
]

# Paramètres de simulation
epsilon_start = 1.0
epsilon_end = 0.01
num_episodes = 5000
steps_per_episode = 100  # Moyenne estimée

# Créer le graphique
plt.figure(figsize=(14, 8))

for decay, mode, label, color in configs:
    epsilon_hist = simulate_epsilon_decay(
        epsilon_start, epsilon_end, decay,
        num_episodes, steps_per_episode, mode
    )
    plt.plot(epsilon_hist, label=label, linewidth=2, color=color, alpha=0.8)

# Ajouter des lignes de référence
plt.axhline(y=epsilon_end, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Epsilon min (0.01)')
plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
plt.axhline(y=0.1, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Mise en forme
plt.xlabel('Épisode', fontsize=12)
plt.ylabel('Epsilon (ε)', fontsize=12)
plt.title('Décroissance d\'Epsilon selon différentes configurations\n'
          f'({num_episodes} épisodes, ~{steps_per_episode} training steps/épisode)',
          fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)

# Ajouter des annotations
plt.text(num_episodes * 0.5, 0.5, 'Zone de transition\n(exploration → exploitation)',
         ha='center', va='center', fontsize=10, alpha=0.6,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('epsilon_decay_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Graphique sauvegardé : epsilon_decay_comparison.png")

# Afficher quelques statistiques
print("\n📊 Statistiques de décroissance d'epsilon :")
print("=" * 70)

for decay, mode, label, _ in configs:
    epsilon_hist = simulate_epsilon_decay(
        epsilon_start, epsilon_end, decay,
        num_episodes, steps_per_episode, mode
    )
    
    # Trouver quand epsilon atteint certains seuils
    eps_50_idx = next((i for i, e in enumerate(epsilon_hist) if e <= 0.5), None)
    eps_10_idx = next((i for i, e in enumerate(epsilon_hist) if e <= 0.1), None)
    eps_min_idx = next((i for i, e in enumerate(epsilon_hist) if e <= epsilon_end + 0.001), None)
    
    print(f"\n{label}:")
    print(f"  Épisode pour ε=0.5  : {eps_50_idx if eps_50_idx else 'jamais'}")
    print(f"  Épisode pour ε=0.1  : {eps_10_idx if eps_10_idx else 'jamais'}")
    print(f"  Épisode pour ε~{epsilon_end} : {eps_min_idx if eps_min_idx else 'jamais'}")
    print(f"  Epsilon final       : {epsilon_hist[-1]:.4f}")

print("\n" + "=" * 70)
print("💡 RECOMMANDATIONS :")
print("  - Pour ~1000-2000 épisodes  : Episode mode avec decay=0.997")
print("  - Pour ~5000 épisodes       : Step mode avec decay=0.9995 (DÉFAUT)")
print("  - Pour ~10000+ épisodes     : Step mode avec decay=0.9998")

plt.show()
