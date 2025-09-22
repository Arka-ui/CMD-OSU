# 🎵 CMD-OSU

Un clone du célèbre jeu de rythme *osu!* entièrement jouable dans votre terminal. Contrôlé exclusivement au clavier avec les touches Z, Q, S, D, et des sliders à maintenir!

## ✨ Fonctionnalités Avancées

*   **Gameplay Rythmique Pur Clavier:** Expérience de jeu complète adaptée pour le terminal
*   **Système de Note Complexe:** Cercles à frapper, sliders à maintenir, et spinners à faire tourner
*   **Système de Scoring Complet:** Precision, combo, health bar, et grades (S, A, B, C, D)
*   **Mods Personnalisables:** Double Time, Half Time, Easy, Hard Rock, No Fail
*   **Génération Automatique de Maps:** Analyse audio automatique avec librosa
*   **Système de Replay:** Enregistrement et rejeu des parties
*   **Interface Terminal Immersive:** Render en temps réel avec curses

## 🛠️ Stack Technologique

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-FF7F00?style=for-the-badge)
![Librosa](https://img.shields.io/badge/Librosa-00BFFF?style=for-the-badge)
![Colorama](https://img.shields.io/badge/Colorama-000000?style=for-the-badge)

*   **Langage:** `Python 3.8+`
*   **Audio:** `pygame` pour la lecture audio et `librosa` pour l'analyse avancée
*   **Interface:** `curses` et `colorama` pour le rendu terminal
*   **Algorithmes:** Beat tracking, onset detection, et génération procédurale de maps

## 🚀 Installation & Lancement

### Pré-requis
- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

1.  **Clonez le repository:**
    ```bash
    git clone https://github.com/Arka-ui/CMD-OSU.git
    cd CMD-OSU
    ```

2.  **Installez les dépendances:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ajoutez vos musiques:**
    Placez vos fichiers audio (.mp3, .wav, .ogg) dans le dossier `songs/`

4.  **Lancez le jeu !**
    ```bash
    python main.py
    ```

### Contrôles

*   `Z`, `Q`, `S`, `D` : Frapper les cercles au bon moment
*   `Maintenir la touche` : Pour les sliders, maintenez la touche enfoncée
*   `Espace` : Mettre en pause
*   `Échap` : Quitter

## 📁 Structure du Projet





## 🧠 Processus de Développement

Ce projet démontre plusieurs compétences techniques avancées:

*   **Analyse Audio Temps-Réel:** Utilisation de librosa pour la détection de beats et onsets
*   **Génération Procédurale:** Création automatique de maps de jeu basée sur l'analyse audio
*   **Rendu Terminal Avancé:** Interface utilisateur riche avec curses et colorama
*   **Architecture de Jeu:** Système d'état, gestion d'entités, et boucle de jeu optimisée
*   **Système de Mods:** Implémentation de modificateurs de gameplay qui affectent la difficulté

**Développement Assisté par IA:** J'ai utilisé des outils comme Copilot et BlackboxAI pour:
- Résoudre des problèmes complexes d'analyse audio et de synchronisation
- Optimiser les algorithmes de génération de maps
- Structurer le code pour une maintenabilité optimale

## 👤 Contact me

*   [GitHub](https://github.com/Arka-ui)
*   [Discord](https://discord.com/users/871084043838566400)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---