import os
import sys
import json
import time
import math
import random
import shutil
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

# ====== Dépendances tierces ======
try:
    import pygame
except ImportError:
    print("Missing dependency: pygame. Install with: pip install pygame")
    sys.exit(1)

try:
    from colorama import init as colorama_init, Fore, Back, Style
except ImportError:
    print("Missing dependency: colorama. Install with: pip install colorama")
    sys.exit(1)

HAVE_LIBROSA = True
try:
    import librosa
    import numpy as np
except Exception:
    HAVE_LIBROSA = False

try:
    import msvcrt
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False
    print("Warning: msvcrt not available. Keyboard input may not work properly.")
    print("This game is optimized for Windows CMD, but will attempt to run on other platforms.")

# ====== Terminal helpers ======
colorama_init(convert=True, autoreset=False)
CSI = "\x1b["

def hidectrl(hide: bool):
    sys.stdout.write(f"{CSI}?25l" if hide else f"{CSI}?25h"); sys.stdout.flush()

def clear_screen():
    sys.stdout.write(f"{CSI}2J{CSI}H"); sys.stdout.flush()

def move_to(y: int, x: int):
    sys.stdout.write(f"{CSI}{max(1,y)};{max(1,x)}H")

def set_color(fg=None, bg=None, style=None):
    if style: sys.stdout.write(style)
    if fg: sys.stdout.write(fg)
    if bg: sys.stdout.write(bg)

def reset_color():
    sys.stdout.write(Style.RESET_ALL)

def term_size() -> Tuple[int,int]:
    sz = shutil.get_terminal_size(fallback=(110,34))
    return sz.columns, sz.lines

def draw_text(y: int, x: int, text: str, fg=None, bg=None, style=None):
    move_to(y, x); set_color(fg,bg,style); sys.stdout.write(text); reset_color(); sys.stdout.flush()

def center_x(text_len:int, cols:int) -> int:
    return max(2, (cols - text_len)//2)

# Read one logical key: returns 'UP','DOWN','LEFT','RIGHT','ENTER','SPACE','ESC', or a printable char
def read_key(timeout: float=None):
    start = time.perf_counter()
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\x1b',): return 'ESC'
            if ch in (b'\r',):    return 'ENTER'
            if ch in (b' ',):     return 'SPACE'
            if ch in (b'\xe0', b'\x00'):
                ch2 = msvcrt.getch()
                code = ch2[0]
                if code == 72: return 'UP'
                if code == 80: return 'DOWN'
                if code == 75: return 'LEFT'
                if code == 77: return 'RIGHT'
                continue
            try:
                c = ch.decode('utf-8')
                return c
            except Exception:
                continue
        if timeout is not None and (time.perf_counter()-start) >= timeout:
            return None
        time.sleep(0.01)

def wait_key_any(msg="Press any key..."):
    print()
    set_color(Fore.BLACK+Back.WHITE, style=Style.BRIGHT)
    print(msg, end="", flush=True)
    reset_color()
    while True:
        if msvcrt.kbhit(): _ = msvcrt.getch(); break
        time.sleep(0.02)

# ====== Audio wrapper ======
class Audio:
    def __init__(self, settings: 'Settings' = None):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.init()
        self._start_perf = None
        self.loaded = None
        # Sound effects
        self.hit_sound = None
        self.miss_sound = None
        self._init_sound_effects()
        self.s = settings

    def _init_sound_effects(self):
        """Generate simple sound effects for hits and misses."""
        try:
            # Hit sound: short high-pitched beep
            hit_freq = 800
            hit_duration = 0.1
            hit_samples = int(44100 * hit_duration)
            hit_buffer = pygame.mixer.Sound(buffer=bytes())
            hit_wave = [int(32767 * math.sin(2 * math.pi * hit_freq * t / 44100)) for t in range(hit_samples)]
            hit_buffer = pygame.mixer.Sound(buffer=bytes(hit_wave))
            self.hit_sound = hit_buffer

            # Miss sound: short low-pitched beep
            miss_freq = 200
            miss_duration = 0.2
            miss_samples = int(44100 * miss_duration)
            miss_buffer = pygame.mixer.Sound(buffer=bytes())
            miss_wave = [int(16383 * math.sin(2 * math.pi * miss_freq * t / 44100)) for t in range(miss_samples)]
            miss_buffer = pygame.mixer.Sound(buffer=bytes(miss_wave))
            self.miss_sound = miss_buffer
        except Exception:
            # If sound generation fails, disable sound effects
            self.hit_sound = None
            self.miss_sound = None

    def play_hit_sound(self):
        """Play hit sound effect."""
        if self.hit_sound and getattr(self, 's', None) and self.s.sound_effects:
            self.hit_sound.play()

    def play_miss_sound(self):
        """Play miss sound effect."""
        if self.miss_sound and self.s.sound_effects:
            self.miss_sound.play()

    def load(self, path: str):
        pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        self.loaded = path

    def play(self, start_sec: float=0.0):
        self._start_perf = time.perf_counter() - start_sec
        pygame.mixer.music.play(start= start_sec if start_sec>0 else 0.0)

    def stop(self):
        pygame.mixer.music.stop()
        self._start_perf = None

    def time(self) -> float:
        if self._start_perf is None: return 0.0
        return max(0.0, time.perf_counter() - self._start_perf)

# ====== Chemins & constantes ======
HERE = os.path.dirname(os.path.abspath(__file__))
SONGS_DIR = os.path.join(HERE, "songs")
MAPS_DIR  = os.path.join(HERE, "maps")
REPLAYS_DIR = os.path.join(HERE, "replays")
CONFIG_PATH = os.path.join(HERE, "term_osu.config.json")
LOG_PATH = os.path.join(HERE, "term_osu.log")
os.makedirs(SONGS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(REPLAYS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

LEVELS = ['Easy','Normal','Hard']

# Scoring constants
SCORE_PER_HIT = 100
COMBO_MULTIPLIER_BASE = 1.0
COMBO_MULTIPLIER_INCREMENT = 0.1
MAX_COMBO_MULTIPLIER = 4.0

# Grade thresholds (accuracy %)
GRADE_SS = 100.0
GRADE_S = 95.0
GRADE_A = 90.0
GRADE_B = 80.0
GRADE_C = 70.0
GRADE_D = 60.0

def calculate_hit_score(hit_time_diff: float, hit_window: float, combo: int, multiplier_enabled: bool = True) -> int:
    """Calculate score for a hit based on timing accuracy and combo."""
    # Base score
    base_score = SCORE_PER_HIT

    # Timing bonus (perfect hit gets 1.2x, good gets 1.0x, ok gets 0.8x)
    timing_ratio = abs(hit_time_diff) / hit_window
    if timing_ratio <= 0.2:  # Perfect hit
        timing_multiplier = 1.2
    elif timing_ratio <= 0.5:  # Good hit
        timing_multiplier = 1.0
    else:  # Ok hit
        timing_multiplier = 0.8

    # Combo multiplier
    combo_multiplier = 1.0
    if multiplier_enabled and combo > 0:
        combo_multiplier = min(MAX_COMBO_MULTIPLIER, COMBO_MULTIPLIER_BASE + (combo - 1) * COMBO_MULTIPLIER_INCREMENT)

    return int(base_score * timing_multiplier * combo_multiplier)

def get_grade(accuracy: float, max_combo: int, total_notes: int) -> str:
    """Determine grade based on accuracy and combo performance."""
    if accuracy >= GRADE_SS and max_combo == total_notes:
        return "SS"
    elif accuracy >= GRADE_S:
        return "S"
    elif accuracy >= GRADE_A:
        return "A"
    elif accuracy >= GRADE_B:
        return "B"
    elif accuracy >= GRADE_C:
        return "C"
    elif accuracy >= GRADE_D:
        return "D"
    else:
        return "F"



def get_active_mods_string(settings: 'Settings') -> str:
    """Get string representation of active mods."""
    mods = []
    if settings.mod_double_time: mods.append("DT")
    if settings.mod_half_time: mods.append("HT")
    if settings.mod_easy: mods.append("EZ")
    if settings.mod_hard_rock: mods.append("HR")
    if settings.mod_no_fail: mods.append("NF")
    return " + ".join(mods) if mods else "None"

# ====== Settings persistants ======
@dataclass
class Settings:
    hit_window_ms_easy: int = 110
    hit_window_ms_normal: int = 90
    hit_window_ms_hard: int = 70
    pre_spawn: float = 0.5
    post_miss: float = 0.5
    audio_offset_ms: int = 0
    show_fps: bool = False
    theme_high_contrast: bool = True
    progress_bar: bool = True
    hud_right_panel: bool = True
    auto_play: bool = False  # NEW: auto 100% accuracy
    # New settings for upgrades
    enable_scoring: bool = True
    enable_health: bool = True
    max_health: int = 100
    health_drain: int = 2  # per miss
    health_recovery: int = 1  # per hit
    combo_multiplier: bool = True
    sound_effects: bool = True
    # Mods
    mod_double_time: bool = False
    mod_half_time: bool = False
    mod_easy: bool = False
    mod_hard_rock: bool = False
    mod_no_fail: bool = False
    # Custom keys
    key_z: str = 'Z'
    key_q: str = 'Q'
    key_s: str = 'S'
    key_d: str = 'D'

def get_key_list(settings: Settings) -> List[str]:
    return [settings.key_z, settings.key_q, settings.key_s, settings.key_d]

def apply_mods_to_settings(settings: Settings) -> Settings:
    """Apply mod effects to settings."""
    # Create a copy to avoid modifying original
    modded = Settings(**settings.__dict__)

    # Speed mods
    if modded.mod_double_time:
        modded.audio_offset_ms = int(modded.audio_offset_ms * 0.5)  # Adjust timing
    elif modded.mod_half_time:
        modded.audio_offset_ms = int(modded.audio_offset_ms * 2.0)

    # Difficulty mods
    if modded.mod_easy:
        modded.hit_window_ms_easy = int(modded.hit_window_ms_easy * 1.4)
        modded.hit_window_ms_normal = int(modded.hit_window_ms_normal * 1.4)
        modded.hit_window_ms_hard = int(modded.hit_window_ms_hard * 1.4)
        modded.max_health = int(modded.max_health * 1.5)
        modded.health_drain = max(1, modded.health_drain - 1)
        modded.health_recovery = modded.health_recovery + 1

    if modded.mod_hard_rock:
        modded.hit_window_ms_easy = max(30, int(modded.hit_window_ms_easy * 0.7))
        modded.hit_window_ms_normal = max(25, int(modded.hit_window_ms_normal * 0.7))
        modded.hit_window_ms_hard = max(20, int(modded.hit_window_ms_hard * 0.7))
        modded.max_health = max(50, int(modded.max_health * 0.8))
        modded.health_drain = modded.health_drain + 1
        modded.health_recovery = max(0, modded.health_recovery - 1)

    if modded.mod_no_fail:
        modded.enable_health = False  # No health drain

    return modded

def load_settings() -> Settings:
    try:
        with open(CONFIG_PATH,"r",encoding="utf-8") as f:
            data = json.load(f)
        s = Settings()
        for k,v in data.items():
            if hasattr(s,k): setattr(s,k,v)
        return s
    except Exception:
        return Settings()

def save_settings(s: Settings):
    try:
        with open(CONFIG_PATH,"w",encoding="utf-8") as f:
            json.dump(s.__dict__, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ====== Audio analyse ======
# ====== Audio analyse (REWRITE MAX) ======
def analyze_audio(path: str):
    """
    Retourne (onset_times, bpm_avg, duration, analysis)
    - onset_times : liste des temps (s) alignés sur les vrais transitoires
    - bpm_avg     : tempo moyen
    - duration    : durée en secondes
    - analysis    : dict (sr, hop, beats_t, onset_env, intensity) pour l’étape suivante
    """
    if HAVE_LIBROSA:
        y, sr = librosa.load(path, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        hop = 512

        # Percussif (HPSS) -> meilleure pertinence des coups
        y_h, y_p = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=hop, aggregate=np.mean)

        # Beat tracking global
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop, trim=False
        )
        beats_t = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

        # Détection d’onsets + backtrack (retour sur le vrai pic)
        frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop,
            backtrack=True, pre_max=6, post_max=6, pre_avg=6, post_avg=6,
            delta=0.05, wait=1
        )
        onsets_t = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
        onsets_t = onsets_t[onsets_t >= 0.05]  # évite les clics très tôt

        # Quantification douce sur la grille locale (1/12 de temps)
        if len(beats_t) >= 2 and len(onsets_t) > 0:
            subdiv = 12
            quantized = []
            median_beat = float(np.median(np.diff(beats_t))) if len(beats_t) >= 2 else (60.0 / (tempo if tempo > 0 else 120.0))
            for t in onsets_t:
                i = np.searchsorted(beats_t, t) - 1
                if 0 <= i < len(beats_t) - 1:
                    b0, b1 = beats_t[i], beats_t[i+1]
                    beat_len = b1 - b0 if (b1 > b0) else median_beat
                else:
                    b0 = (t // median_beat) * median_beat
                    beat_len = median_beat
                step = beat_len / subdiv
                k = int(round((t - b0) / step))
                g = b0 + k * step
                # snap seulement si proche (≤ 40% du pas ou ≤ 40 ms)
                tau = min(0.04, 0.4 * step)
                quantized.append(g if abs(g - t) <= tau else t)
            onsets_t = np.array(quantized, dtype=float)

        # Fusion < 22 ms pour éviter doubles coups
        if len(onsets_t) >= 2:
            merged = [float(onsets_t[0])]
            for t in onsets_t[1:]:
                t = float(t)
                if t - merged[-1] < 0.022:
                    merged[-1] = t
                else:
                    merged.append(t)
            onsets_t = np.array(merged, dtype=float)

        # Intensité (normalisée 0..1) via enveloppe d’onset
        if len(frames) > 0:
            env_vals = onset_env[np.clip(frames, 0, len(onset_env)-1)]
        else:
            env_vals = np.array([], dtype=float)
        if env_vals.size > 0:
            p5, p95 = np.percentile(env_vals, [5, 95])
            denom = (p95 - p5) if (p95 - p5) > 1e-9 else 1.0
            intensity = np.clip((env_vals - p5) / denom, 0.0, 1.0).astype(float)
        else:
            intensity = np.zeros_like(onsets_t, dtype=float)

        bpm = float(tempo if tempo and tempo > 0 else 120.0)

        analysis = {
            "sr": int(sr),
            "hop": int(hop),
            "beats_t": beats_t.astype(float).tolist(),
            "onset_env": onset_env.astype(float).tolist(),
        }

        # Recalcule l’intensité exactement aux temps (post-quantif) par interpolation
        if onsets_t.size > 0 and onset_env.size > 0:
            def env_at_times(ts):
                f = (np.array(ts) * sr) / hop
                idx0 = np.clip(np.floor(f).astype(int), 0, len(onset_env)-1)
                idx1 = np.clip(idx0 + 1, 0, len(onset_env)-1)
                frac = f - idx0
                return (1 - frac) * onset_env[idx0] + frac * onset_env[idx1]
            env_q = env_at_times(onsets_t)
            p5, p95 = np.percentile(env_q, [5, 95])
            denom = (p95 - p5) if (p95 - p5) > 1e-9 else 1.0
            intensity = np.clip((env_q - p5) / denom, 0.0, 1.0).astype(float)
        analysis["intensity"] = intensity.astype(float).tolist()

        return onsets_t.astype(float).tolist(), bpm, float(duration), analysis

    # ====== Fallback sans librosa : simple grille (mieux que rien) ======
    try:
        pygame.mixer.init()
        snd = pygame.mixer.Sound(path)
        duration = float(snd.get_length())
    except Exception:
        duration = 120.0
    bpm_avg = 120.0
    step = 60.0 / bpm_avg
    onsets = []
    t = 0.5
    while t < max(0.0, duration - 0.05):
        onsets.append(float(t))
        t += step
    analysis = {"sr": None, "hop": None, "beats_t": [], "onset_env": [], "intensity": [0.5] * len(onsets)}
    return onsets, bpm_avg, duration, analysis

# ====== Utils fichiers ======
def list_songs(base_dir: str) -> List[str]:
    exts = (".mp3",".ogg",".wav",".flac",".m4a",".aac",".opus",".wma")
    files = []
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            p = os.path.join(base_dir, name)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                files.append(p)
    return sorted(files)

def map_path_for_song(song_path: str) -> str:
    base = os.path.splitext(os.path.basename(song_path))[0]
    return os.path.join(MAPS_DIR, base + ".map.json")

def load_map_for_song(song_path: str) -> Optional[dict]:
    p = map_path_for_song(song_path)
    if not os.path.exists(p): return None
    try:
        with open(p,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def level_hit_window(settings: Settings, level: str) -> int:
    return {"Easy": settings.hit_window_ms_easy,
            "Normal": settings.hit_window_ms_normal,
            "Hard": settings.hit_window_ms_hard}[level]

# ====== Construction des maps (REWRITE) ======
from typing import Optional

def _compute_base_min_gap(level: str, bpm_avg: float) -> float:
    """Ecart mini de base selon niveau (en s), borné par le tempo."""
    beat = 60.0 / max(30.0, min(240.0, bpm_avg))
    if level == 'Easy':   return max(0.18, beat / 2.0)   # ~1/8 à 120 BPM
    if level == 'Normal': return max(0.12, beat / 3.0)   # ~triolet
    return max(0.06, beat / 4.0)                         # ~1/16 en Hard

def _local_rate(onsets: List[float], i: int, radius: float = 0.75) -> float:
    """Notes par seconde dans une fenêtre centrée autour de onsets[i]."""
    t = onsets[i]
    left, right = t - radius, t + radius
    cnt = 0
    j = i
    while j >= 0 and onsets[j] >= left:
        cnt += 1; j -= 1
    j = i + 1
    while j < len(onsets) and onsets[j] <= right:
        cnt += 1; j += 1
    return cnt / (2 * radius + 1e-9)

def filter_and_assign(onsets: List[float], level: str, bpm_avg: float, settings: 'Settings', analysis: Optional[dict] = None) -> List[dict]:
    """
    Fidèle au son : on ne crée PAS de notes, on filtre juste quand c'est injouable.
    - Snap déjà effectué dans analyze_audio()
    - Filtrage min_gap adaptatif à la densité locale + difficulté (Hard tolère des jacks)
    - Types:
        * tap par défaut
        * slider si l'écart au prochain coup est long ET énergie soutenue
        * spinner si l'écart est très long ET faible énergie (segments “ambiants”)
    """
    import math
    # Base min_gap selon diff + mods EZ/HR (DT/HT restent runtime)
    base_gap = _compute_base_min_gap(level, bpm_avg)
    if getattr(settings, "mod_easy", False):      base_gap *= 1.15
    if getattr(settings, "mod_hard_rock", False): base_gap *= 0.85

    onsets = sorted(float(t) for t in onsets if t >= 0.0)

    # --- Sélection des onsets : pas d'enrichissement, seulement un filtrage jouable ---
    kept: List[float] = []
    last_keep = -1e9
    for i, t in enumerate(onsets):
        rate = _local_rate(onsets, i, radius=0.75)  # notes/s autour du point
        # Plus c'est dense, plus on autorise court (plancher 40 ms)
        density_factor = 1.0 / (1.0 + 0.08 * max(0.0, rate - 2.0))
        min_gap_i = max(0.04, base_gap * density_factor)
        if t - last_keep >= min_gap_i:
            kept.append(t); last_keep = t
        else:
            # Hard: tolère un jack si c'est "presque" suffisant
            if level == 'Hard' and (t - last_keep) >= (0.5 * min_gap_i):
                kept.append(t); last_keep = t
            # Easy/Normal: on drop

    # --- Energie locale pour décider sliders/spinners ---
    def env_norm(time_s: float) -> float:
        """Énergie normalisée 0..1 autour de time_s (si analyze_audio a fourni onset_env)."""
        if not analysis or not analysis.get("onset_env") or not analysis.get("sr") or not analysis.get("hop"):
            return 0.0
        try:
            import numpy as np
        except Exception:
            return 0.0
        env_arr = np.array(analysis["onset_env"], dtype=float)
        if env_arr.size == 0:
            return 0.0
        sr = int(analysis["sr"]); hop = int(analysis["hop"])
        # stats pour normaliser
        p5, p95 = np.percentile(env_arr, [5, 95])
        scale = max(1e-9, (p95 - p5))
        # interpolation linéaire à l'instant demandé
        f = (time_s * sr) / hop
        i0 = int(np.clip(math.floor(f), 0, len(env_arr)-1))
        i1 = min(len(env_arr)-1, i0+1)
        frac = f - i0
        v = (1 - frac) * env_arr[i0] + frac * env_arr[i1]
        return float(max(0.0, min(1.0, (v - p5) / scale)))

    # --- Assignation des touches + type de note ---
    keys = get_key_list(settings)  # ex: ['Z','Q','S','D']
    notes: List[dict] = []
    last_key = None

    # Seuils par difficulté
    sustain_thresh = 0.45 if level == 'Easy' else (0.35 if level == 'Normal' else 0.28)
    spinner_thresh = 1.60 if level == 'Easy' else (1.40 if level == 'Normal' else 1.20)

    for i, t in enumerate(kept):
        # Choix de la touche : alternance par défaut, jack en Hard si très serré
        if last_key is not None:
            gap_prev = t - kept[i-1]
            if level == 'Hard' and gap_prev < base_gap * 0.75:
                k = last_key  # jack
            else:
                idx = (keys.index(last_key) + 1) % len(keys)
                k = keys[idx]
        else:
            k = keys[0]
        last_key = k

        # Type + durée
        note_type = "tap"
        duration = 0.0
        next_t = kept[i+1] if i+1 < len(kept) else None

        if next_t is not None:
            gap_next = next_t - t
            energy = env_norm(t + 0.05)  # un poil après l'attaque

            # Spinner: très long écart + faible énergie (sustains ambiants)
            if gap_next >= spinner_thresh and energy < 0.35:
                note_type = "spinner"
                duration = min(gap_next - 0.10, 4.0)

            # Slider: long écart + énergie soutenue
            elif gap_next >= sustain_thresh and energy >= 0.55:
                note_type = "slider"
                duration = min(gap_next - 0.05, 2.5)

            # Sinon: tap
        # Dernière note: tap

        notes.append({
            "t": round(float(t), 6),
            "key": k,
            "type": note_type,
            "duration": round(float(max(0.0, duration)), 6)
        })

    return notes

def make_map_for_song(song_path: str, settings: Settings):
    onsets, bpm_avg, duration, analysis = analyze_audio(song_path)  # <— CHANGEMENT
    levels = {}
    for L in LEVELS:
        notes = filter_and_assign(onsets, L, bpm_avg, settings, analysis=analysis)  # <— CHANGEMENT
        levels[L] = {
            "hit_window_ms": int(level_hit_window(settings, L)),
            "notes": notes
        }
    mp = {
        "version": 3,  # <— CHANGEMENT
        "song_file": os.path.basename(song_path),
        "duration": float(duration),
        "bpm_avg": float(bpm_avg),
        "levels": levels,
        "analysis": {  # meta utile mais non requis au runtime
            "has_librosa": bool(HAVE_LIBROSA),
            "beats_count": len(analysis.get("beats_t", [])) if analysis else 0
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(map_path_for_song(song_path), "w", encoding="utf-8") as f:
        json.dump(mp, f, ensure_ascii=False, indent=2)

def scan_and_build_maps(settings: Settings):
    files = list_songs(SONGS_DIR)
    if not files: return
    clear_screen()
    cols, rows = term_size()
    title = "Scanning songs & building maps..."
    draw_text(2, center_x(len(title), cols), title, fg=Fore.MAGENTA, style=Style.BRIGHT)
    for path in files:
        mapp = map_path_for_song(path)
        need = True
        try:
            if os.path.exists(mapp):
                need = (os.path.getmtime(mapp) < os.path.getmtime(path))
        except Exception:
            need = True
        line = f"• {os.path.basename(path)}"
        draw_text(4, 4, line, fg=Fore.CYAN)
        if need:
            try:
                make_map_for_song(path, settings)
                draw_text(5, 6, "map generated", fg=Fore.GREEN)
            except Exception as e:
                draw_text(5, 6, f"map failed: {e}", fg=Fore.RED)
        else:
            draw_text(5, 6, "up to date", fg=Fore.GREEN)
        time.sleep(0.3)

# ====== Visuel ======
def format_time_ms(sec: float) -> str:
    if sec < 0: sec = 0.0
    m = int(sec // 60); s = int(sec % 60); ms = int((sec-int(sec))*1000)
    return f"{m:02d}:{s:02d}:{ms:03d}"

def big_key_box(char: str) -> List[str]:
    c = char.upper()
    return ["┌───┐", f"│ {c} │", "└───┘"]

# ====== Menu UI ======
def draw_menu(title: str, items: List[str], selected: int, footer: Optional[str]=None):
    clear_screen()
    cols, rows = term_size()
    draw_text(2, center_x(len(title), cols), title, fg=Fore.MAGENTA, style=Style.BRIGHT)
    top = max(5, rows//2 - len(items)//2)
    for i, label in enumerate(items):
        prefix = "▶ " if i==selected else "  "
        line = prefix + label
        fg = Fore.CYAN if i==selected else Fore.WHITE
        draw_text(top+i, center_x(len(line), cols), line, fg=fg, style=Style.BRIGHT if i==selected else None)
    if footer:
        draw_text(rows-2, center_x(len(footer), cols), footer, fg=Fore.WHITE)

def menu_select(title: str, options: List[str], footer: Optional[str]=None) -> Optional[int]:
    idx = 0
    while True:
        draw_menu(title, options, idx, footer)
        k = read_key()
        if k=='UP': idx = (idx-1) % len(options)
        elif k=='DOWN': idx = (idx+1) % len(options)
        elif k in ('ENTER',):
            return idx
        elif k in ('ESC',):
            return None

# ====== Settings UI ======
def settings_screen(s: Settings):
    items = [
        ("Hit window Easy (ms)", "value", "hit_window_ms_easy", 30, 200, 5),
        ("Hit window Normal (ms)", "value", "hit_window_ms_normal", 30, 200, 5),
        ("Hit window Hard (ms)", "value", "hit_window_ms_hard", 20, 180, 5),
        ("Pre-spawn (s)", "valuef", "pre_spawn", 0.2, 1.5, 0.05),
        ("Post-miss (s)", "valuef", "post_miss", 0.2, 1.5, 0.05),
        ("Audio offset (ms)", "value", "audio_offset_ms", -250, 250, 5),
        ("High contrast theme", "toggle", "theme_high_contrast", None, None, None),
        ("Show FPS", "toggle", "show_fps", None, None, None),
        ("HUD right panel", "toggle", "hud_right_panel", None, None, None),
        ("Progress bar", "toggle", "progress_bar", None, None, None),
        ("Auto-play (100%)", "toggle", "auto_play", None, None, None),
        # New upgrade settings
        ("Enable Scoring", "toggle", "enable_scoring", None, None, None),
        ("Enable Health", "toggle", "enable_health", None, None, None),
        ("Max Health", "value", "max_health", 50, 200, 10),
        ("Health Drain/Miss", "value", "health_drain", 1, 10, 1),
        ("Health Recovery/Hit", "value", "health_recovery", 0, 5, 1),
        ("Combo Multiplier", "toggle", "combo_multiplier", None, None, None),
        ("Sound Effects", "toggle", "sound_effects", None, None, None),
        # Mods
        ("Double Time", "toggle", "mod_double_time", None, None, None),
        ("Half Time", "toggle", "mod_half_time", None, None, None),
        ("Easy", "toggle", "mod_easy", None, None, None),
        ("Hard Rock", "toggle", "mod_hard_rock", None, None, None),
        ("No Fail", "toggle", "mod_no_fail", None, None, None),
        # Custom Keys
        ("Key Z", "key", "key_z", None, None, None),
        ("Key Q", "key", "key_q", None, None, None),
        ("Key S", "key", "key_s", None, None, None),
        ("Key D", "key", "key_d", None, None, None),
        ("Back", "action", None, None, None, None),
    ]
    idx = 0
    while True:
        clear_screen()
        cols, rows = term_size()
        title = "Settings"
        draw_text(2, center_x(len(title), cols), title, fg=Fore.MAGENTA, style=Style.BRIGHT)
        top = 5
        for i,(label,typ,attr,lo,hi,step) in enumerate(items):
            if typ=="value":
                val = getattr(s, attr)
                line = f"{label}: {val}"
            elif typ=="valuef":
                val = getattr(s, attr)
                line = f"{label}: {val:.2f}"
            elif typ=="toggle":
                val = getattr(s, attr)
                line = f"{label}: {'ON' if val else 'OFF'}"
            elif typ=="key":
                val = getattr(s, attr)
                line = f"{label}: [{val}]"
            else:
                line = label
            prefix = "▶ " if i==idx else "  "
            fg = Fore.CYAN if i==idx else Fore.WHITE
            draw_text(top+i, center_x(len(prefix+line), cols), prefix+line, fg=fg, style=Style.BRIGHT if i==idx else None)
        draw_text(rows-2, center_x(60, cols), "↑/↓ select  •  ←/→ adjust  •  SPACE toggle  •  ENTER apply  •  ESC back", fg=Fore.WHITE)
        k = read_key()
        if k=='UP': idx = (idx-1) % len(items)
        elif k=='DOWN': idx = (idx+1) % len(items)
        elif k=='LEFT':
            label,typ,attr,lo,hi,step = items[idx]
            if typ=="value":
                cur = getattr(s, attr); cur = max(lo, cur - step); setattr(s, attr, cur)
            if typ=="valuef":
                cur = getattr(s, attr); cur = max(lo, round(cur - step, 2)); setattr(s, attr, cur)
        elif k=='RIGHT':
            label,typ,attr,lo,hi,step = items[idx]
            if typ=="value":
                cur = getattr(s, attr); cur = min(hi, cur + step); setattr(s, attr, cur)
            if typ=="valuef":
                cur = getattr(s, attr); cur = min(hi, round(cur + step, 2)); setattr(s, attr, cur)
        elif k=='SPACE':
            label,typ,attr,lo,hi,step = items[idx]
            if typ=="toggle":
                setattr(s, attr, not getattr(s, attr))
            elif typ=="key":
                draw_text(rows-2, center_x(40, cols), f"Press new key for {label}...", fg=Fore.CYAN)
                new_key = read_key()
                if new_key and len(new_key) == 1:
                    setattr(s, attr, new_key.upper())
                draw_text(rows-2, center_x(40, cols), " "*40, fg=Fore.WHITE)  # Clear message
        elif k=='ENTER':
            if items[idx][1]=="action":
                save_settings(s); return
        elif k=='ESC':
            save_settings(s); return

# ====== Sélecteurs ======
def pick_song_menu() -> Optional[str]:
    files = list_songs(SONGS_DIR)
    if not files:
        clear_screen()
        msg = f"Add songs in: {SONGS_DIR}"
        cols, _ = term_size()
        draw_text(6, center_x(len(msg), cols), msg, fg=Fore.YELLOW)
        wait_key_any("No songs found. Press any key...")
        return None
    labels = [os.path.basename(p) for p in files]
    idx = menu_select("Select a song", labels, footer="↑/↓ select • ENTER play • ESC back")
    return files[idx] if idx is not None else None

def pick_level_menu() -> Optional[str]:
    idx = menu_select("Difficulty", ["Easy","Normal","Hard"], footer="↑/↓ select • ENTER confirm • ESC back")
    return ["Easy","Normal","Hard"][idx] if idx is not None else None

def tutorial_mode(settings: Settings):
    """Simple tutorial mode to teach basic gameplay."""
    cols, rows = term_size()
    clear_screen()

    tutorial_steps = [
        ("Welcome to Terminal osu!", "This is a rhythm game where you hit notes to the beat."),
        ("Controls", f"Use {settings.key_z}, {settings.key_q}, {settings.key_s}, {settings.key_d} to hit notes."),
        ("Timing", "Hit the notes when they reach the center. Perfect timing gives bonus points!"),
        ("Combos", "Hitting notes in sequence builds your combo multiplier."),
        ("Health", "Missing notes drains your health. Don't let it reach zero!"),
        ("Note Types", "Tap: Single press | Slider: Hold the key | Spinner: Multiple presses"),
        ("Mods", "Try different mods in settings to change difficulty and speed."),
        ("Ready?", "Press ENTER to start a practice round, or ESC to return to menu.")
    ]

    step = 0
    while True:
        clear_screen()
        title, desc = tutorial_steps[step]
        draw_text(2, center_x(len(title), cols), title, fg=Fore.MAGENTA, style=Style.BRIGHT)
        draw_text(4, center_x(len(desc), cols), desc, fg=Fore.WHITE)

        if step < len(tutorial_steps) - 1:
            draw_text(rows-2, center_x(40, cols), "Press ENTER for next, ESC to skip", fg=Fore.CYAN)
        else:
            draw_text(rows-2, center_x(40, cols), "Press ENTER to practice, ESC to menu", fg=Fore.CYAN)

        k = read_key()
        if k == 'ENTER':
            if step < len(tutorial_steps) - 1:
                step += 1
            else:
                # Start practice round
                return True
        elif k == 'ESC':
            return False

def list_replays() -> List[str]:
    files = []
    for name in os.listdir(REPLAYS_DIR):
        p = os.path.join(REPLAYS_DIR, name)
        if os.path.isfile(p) and name.lower().endswith(".json"):
            files.append(p)
    return sorted(files)

def pick_replay_menu() -> Optional[str]:
    reps = list_replays()
    if not reps:
        wait_key_any("No replays found. Press any key...")
        return None
    labels=[]
    metas=[]
    for rp in reps:
        try:
            with open(rp,"r",encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 1)
            lab = f"{os.path.basename(rp)}  | {data.get('song_file','?')}  | {data.get('level','?')}  | acc {data.get('accuracy','?')}%"
            if version >= 2:
                lab += f"  | score {data.get('score', 0):,}  | grade {data.get('grade', 'N/A')}"
        except Exception:
            lab = os.path.basename(rp)
        labels.append(lab)
    idx = menu_select("Play Replay", labels, footer="↑/↓ select • ENTER play • ESC back")
    return reps[idx] if idx is not None else None

# ====== Gameplay ======
class Game:
    def __init__(self, settings: Settings):
        self.s = settings
        self.audio = Audio(settings)
        # New attributes for upgrades
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.health = settings.max_health if settings.enable_health else 100
        self.hit_effects = []  # List of (x, y, intensity, time_left)

    def poll_keys(self) -> List[str]:
        pressed=[]
        key_list = get_key_list(self.s)
        while msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\x1b', b'\x03'):
                pressed.append('ESC'); continue
            try:
                k = ch.decode('utf-8').upper()
                if k == ' ':
                    pressed.append(' ')
                elif k in key_list:
                    pressed.append(k)
            except Exception:
                pass
        return pressed

    def get_held_keys(self) -> Set[str]:
        """Get currently held keys (for sliders/spinners)."""
        held = set()
        # This is a simplified version - in a real implementation you'd track key states
        # For now, we'll use a simple approach
        return held

    def draw_hud(self, cols:int, rows:int, accuracy:float, now:float, song_name:str, level:str, bpm_avg:float, hits:int, misses:int, combo:int, total_notes:int, progress_ratio:float, fps:Optional[float]):
        # Acc bar
        acc = max(0.0, min(100.0, accuracy))
        bar_w = cols - 4
        fill = int((acc/100.0)*bar_w)
        color = Fore.GREEN if acc>=80 else (Fore.YELLOW if acc>=60 else Fore.RED)
        draw_text(1, 2, "[" + "#"*fill + "-"*(bar_w-fill) + f"] {acc:5.1f}%", fg=color, style=Style.BRIGHT)
        # timer
        tstr = format_time_ms(now)
        draw_text(2, center_x(len(tstr), cols), tstr, fg=Fore.CYAN, style=Style.BRIGHT)
        # info
        mods_str = get_active_mods_string(self.s)
        info = f"{song_name}  |  {level}  |  BPM≈{bpm_avg:.1f}  |  Mods: {mods_str}"
        draw_text(3, center_x(len(info), cols), info, fg=Fore.MAGENTA, style=Style.BRIGHT)
        # Upgrade: score and health display
        if self.s.enable_scoring:
            score_str = f"Score: {self.score:,}"
            draw_text(4, center_x(len(score_str), cols), score_str, fg=Fore.YELLOW, style=Style.BRIGHT)
        if self.s.enable_health:
            health_bar_w = 20
            health_fill = int((self.health / self.s.max_health) * health_bar_w)
            health_color = Fore.GREEN if self.health > 50 else (Fore.YELLOW if self.health > 25 else Fore.RED)
            health_str = f"HP: [{'█'*health_fill}{'-'*(health_bar_w-health_fill)}] {self.health}/{self.s.max_health}"
            draw_text(5, center_x(len(health_str), cols), health_str, fg=health_color, style=Style.BRIGHT)
        # stats
        stats = f"Combo: {self.combo}   Hits: {hits}   Misses: {misses}   Notes: {total_notes}"
        if self.s.hud_right_panel:
            draw_text(6 if (self.s.enable_scoring or self.s.enable_health) else 4, 2, stats, fg=Fore.WHITE, style=Style.BRIGHT)
        # progress
        if self.s.progress_bar:
            pr_w = cols - 4
            pr_fill = int(pr_w * max(0.0, min(1.0, progress_ratio)))
            draw_text(rows-1, 2, "[" + "■"*pr_fill + "-"*(pr_w-pr_fill) + "]", fg=Fore.BLUE, style=Style.BRIGHT)
        if self.s.show_fps and fps is not None:
            draw_text(2, cols-10, f"{fps:5.1f}fps", fg=Fore.YELLOW)

    def draw_scene(self, cols:int, rows:int, drawables: List[Tuple[int,int,List[str],str]], hit_effects: List[Tuple[int,int,float,float]] = None):
        clear_screen()
        for (x, y_top, lines, state) in drawables:
            if y_top < 5 or y_top+len(lines) > rows-2:  # éviter HUD/bas
                continue
            if state=="green": fg,bg = Fore.BLACK, Back.GREEN
            elif state=="red": fg,bg = Fore.WHITE, Back.RED
            elif state=="blue": fg,bg = Fore.WHITE, Back.BLUE
            else: fg,bg = (Fore.WHITE, None)
            for i,ln in enumerate(lines):
                draw_text(y_top+i, x, ln, fg=fg, bg=bg, style=Style.BRIGHT)

        # Draw hit effects
        if hit_effects:
            for (x, y, intensity, _time_left) in hit_effects:
                if 0 <= x < cols and 5 <= y < rows-2:
                    effect_char = "★" if intensity > 0.5 else "✦"
                    color = Fore.YELLOW if intensity > 0.5 else Fore.CYAN
                    draw_text(y, x, effect_char, fg=color, style=Style.BRIGHT)

    def build_runtime_positions(self, notes_data: List[dict], cols:int, rows:int, seed:int, pre:float, post:float):
        rng = random.Random(seed)
        top_area = 5
        bottom_margin = 2
        min_x, max_x = 4, max(10, cols-10)
        min_y, max_y = top_area, rows - bottom_margin - 2
        runtime=[]
        active_positions=[]
        for nd in notes_data:
            due = float(nd["t"])
            if due < pre: continue
            spawn_time = due - pre
            active_positions = [p for p in active_positions if p[2] > spawn_time]
            placed=False
            for _ in range(24):
                x = rng.randint(min_x, max_x)
                y = rng.randint(min_y, max_y)
                ok=True
                for (px,py,dead) in active_positions:
                    if abs(px-x) < 4 and abs(py-y) < 2:
                        ok=False; break
                if ok:
                    duration = nd.get("duration", 0.0)
                    runtime.append({"due":due,"key":nd["key"],"type":nd["type"],"x":x,"y":y,"hit":False,"miss":False,"removed":False,"duration":duration,"end_time":due + duration,"held":False})
                    active_positions.append((x,y, due + post))
                    placed=True; break
            if not placed:
                duration = nd.get("duration", 0.0)
                runtime.append({"due":due,"key":nd["key"],"type":nd["type"],"x":min_x,"y":min_y,"hit":False,"miss":False,"removed":False,"duration":duration,"end_time":due + duration,"held":False})
        return runtime

    def countdown(self, level:str):
        cols, rows = term_size()
        for i in range(5,0,-1):
            clear_screen()
            msg = f"Starting in {i}s"
            draw_text(rows//2, center_x(len(msg), cols), msg, fg=Fore.MAGENTA, style=Style.BRIGHT)
            draw_text(rows//2 + 2, center_x(len(level)+8, cols), f"Level: {level}", fg=Fore.CYAN, style=Style.BRIGHT)
            time.sleep(1.0)
        clear_screen()

    def play_round(self, song_path: str, map_data: dict, level: str, auto_play: bool=False, replay_driver: Optional[dict]=None) -> dict:
        logging.info(f"Starting round: {song_path} - {level}")
        # replay_driver:
        #   {"seed":int, "notes":[{"t":...,"key":"Z","type":"tap","hit":True/False,"hit_time":float or null}],
        #    "settings":{...}}  -> si fourni, rejoue exactement les hits/misses selon hit_time
        cols, rows = term_size()

        # Apply mods to settings
        modded_settings = apply_mods_to_settings(self.s)
        pre = modded_settings.pre_spawn
        post = modded_settings.post_miss
        hit_win = map_data["levels"][level]["hit_window_ms"]/1000.0
        notes_data = map_data["levels"][level]["notes"]
        song_name = os.path.basename(map_data["song_file"])
        bpm_avg = map_data["bpm_avg"]
        duration = map_data["duration"]

        # seed pour positions
        if replay_driver is not None:
            seed = int(replay_driver.get("seed", 123456))
            # si notes dans replay, les timings et keys viennent de là
            if "notes" in replay_driver:
                notes_data = replay_driver["notes"]
        else:
            seed = int(time.time()*1000) & 0xFFFFFFFF

        runtime_notes = self.build_runtime_positions(notes_data, cols, rows, seed, pre, post)
        total_notes = len(runtime_notes)

        # countdown 5s
        self.countdown(level)

        # lancer audio
        self.audio.load(os.path.join(SONGS_DIR, song_name))
        start_at = 0.0 + max(0.0, self.s.audio_offset_ms/1000.0)
        self.audio.play(start_at)

        last_frame = time.perf_counter()
        fps = None
        hits = 0; misses = 0; combo = 0
        # Initialize upgrade stats
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.health = self.s.max_health if self.s.enable_health else 100
        self.hit_effects = []
        self.paused = False
        self.pause_start_time = 0.0

        # replay control
        replay_mode = (replay_driver is not None)
        replay_index_by_keytime = None
        if replay_mode and "notes" in replay_driver:
            # fabrique dict par (t,key) approx -> hit_time
            replay_index_by_keytime = {}
            for nd in replay_driver["notes"]:
                t = round(float(nd["t"]),6); k = nd["key"].upper()
                replay_index_by_keytime[(t,k)] = nd.get("hit_time", None) if nd.get("hit") else None

        try:
            while True:
                dt = 1/60.0
                now = self.audio.time()
                if now > duration + 2.0:
                    break

                # Handle pause
                if self.paused:
                    clear_screen()
                    pause_msg = "PAUSED - Press SPACE to resume, ESC to quit"
                    draw_text(rows//2, center_x(len(pause_msg), cols), pause_msg, fg=Fore.YELLOW, style=Style.BRIGHT)
                    draw_text(rows//2 + 2, center_x(30, cols), f"Score: {self.score:,}", fg=Fore.CYAN)
                    draw_text(rows//2 + 3, center_x(30, cols), f"Combo: {self.combo}", fg=Fore.CYAN)
                    time.sleep(0.1)
                    continue

                # jugements auto/replay
                for n in runtime_notes:
                    if n["removed"]: continue
                    due = n["due"]
                    # REPLAY: applique exactement à hit_time
                    if replay_mode and replay_index_by_keytime is not None:
                        key = (round(due,6), n["key"].upper())
                        ht = replay_index_by_keytime.get(key, None)
                        if ht is not None and (not n["hit"]) and now >= float(ht):
                            n["hit"]=True; n["removed"]=True; hits += 1; combo += 1
                        elif ht is None:
                            # miss quand fin fenêtre
                            if (not n["hit"]) and (not n["miss"]):
                                should_miss = False
                                if n["type"] in ["slider", "spinner"]:
                                    if now > n["end_time"] + hit_win:
                                        should_miss = True
                                else:
                                    if now > due + hit_win:
                                        should_miss = True

                                if should_miss:
                                    n["miss"]=True; misses += 1; combo=0
                                    # Upgrade: combo break and health drain
                                    self.combo = 0
                                    if self.s.enable_health:
                                        self.health = max(0, self.health - self.s.health_drain)
                                    if self.s.sound_effects:
                                        self.audio.play_miss_sound()
                    elif auto_play:
                        # auto: 100% -> hit à due
                        if (not n["hit"]) and now >= due:
                            if n["type"] == "slider":
                                # Auto-play slider: hold until end
                                if now >= n["end_time"]:
                                    n["hit"]=True; n["removed"]=True; hits += 1; combo += 1
                                    # Upgrade: scoring and health for auto-play
                                    if self.s.enable_scoring:
                                        hit_score = calculate_hit_score(0.0, hit_win, combo, self.s.combo_multiplier)
                                        self.score += hit_score
                                    self.combo += 1
                                    self.max_combo = max(self.max_combo, self.combo)
                                    if self.s.enable_health:
                                        self.health = min(self.s.max_health, self.health + self.s.health_recovery)
                            elif n["type"] == "spinner":
                                # Auto-play spinner: spin until end
                                if now >= n["end_time"]:
                                    n["hit"]=True; n["removed"]=True; hits += 1; combo += 1
                                    # Upgrade: scoring and health for auto-play
                                    if self.s.enable_scoring:
                                        hit_score = calculate_hit_score(0.0, hit_win, combo, self.s.combo_multiplier)
                                        self.score += hit_score
                                    self.combo += 1
                                    self.max_combo = max(self.max_combo, self.combo)
                                    if self.s.enable_health:
                                        self.health = min(self.s.max_health, self.health + self.s.health_recovery)
                            else:
                                # Regular tap
                                n["hit"]=True; n["removed"]=True; hits += 1; combo += 1
                                # Upgrade: scoring and health for auto-play
                                if self.s.enable_scoring:
                                    hit_score = calculate_hit_score(0.0, hit_win, combo, self.s.combo_multiplier)
                                    self.score += hit_score
                                self.combo += 1
                                self.max_combo = max(self.max_combo, self.combo)
                                if self.s.enable_health:
                                    self.health = min(self.s.max_health, self.health + self.s.health_recovery)
                    else:
                        # joueur: auto-miss si dépasse fenêtre
                        if (not n["hit"]) and (not n["miss"]):
                            should_miss = False
                            if n["type"] in ["slider", "spinner"]:
                                # Special miss conditions for sliders/spinners
                                if now > n["end_time"] + hit_win:
                                    should_miss = True
                            else:
                                if now > due + hit_win:
                                    should_miss = True

                            if should_miss:
                                n["miss"]=True; misses += 1; combo=0
                                # Upgrade: combo break and health drain
                                self.combo = 0
                                if self.s.enable_health:
                                    self.health = max(0, self.health - self.s.health_drain)
                                if self.s.sound_effects:
                                    self.audio.play_miss_sound()

                    # remove après mort
                    death_time = n["end_time"] + post if n["type"] in ["slider", "spinner"] else due + post
                    if now > death_time:
                        n["removed"] = True

                # inputs joueur (ignorés en auto/replay)
                if not auto_play and not replay_mode:
                    pressed = self.poll_keys()
                    if 'ESC' in pressed: break
                    if ' ' in pressed:  # SPACE for pause
                        self.paused = not self.paused
                        if self.paused:
                            self.pause_start_time = now
                            self.audio.stop()
                        else:
                            # Resume audio from where we left off
                            resume_at = now
                            self.audio.play(resume_at)
                        continue
                    for K in pressed:
                        # hit si dans fenêtre
                        cand=None; best_abs=None
                        for n in runtime_notes:
                            if n["removed"] or n["hit"] or n["miss"]: continue
                            if n["key"] != K: continue
                            ad = abs(now - n["due"])
                            if ad <= hit_win:
                                if cand is None or ad < best_abs:
                                    cand=n; best_abs=ad
                        if cand is not None:
                            if cand["type"] == "slider":
                                # Slider: mark as held if not already hit
                                if not cand["held"]:
                                    cand["held"] = True
                                    # Check if held until end
                                    if now >= cand["end_time"] - hit_win:
                                        cand["hit"] = True
                                        cand["removed"] = True
                                        hits += 1; combo += 1
                                        # Upgrade: scoring and health
                                        if self.s.enable_scoring:
                                            hit_score = calculate_hit_score(best_abs, hit_win, combo, self.s.combo_multiplier)
                                            self.score += hit_score
                                        self.combo += 1
                                        self.max_combo = max(self.max_combo, self.combo)
                                        if self.s.enable_health:
                                            self.health = min(self.s.max_health, self.health + self.s.health_recovery)
                                        if self.s.sound_effects:
                                            self.audio.play_hit_sound()
                                        # Add hit effect
                                        self.hit_effects.append((cand["x"], cand["y"], 1.0, 0.3))  # x, y, intensity, duration
                            elif cand["type"] == "spinner":
                                # Spinner: require multiple alternating presses
                                if not hasattr(cand, "spin_count"):
                                    cand["spin_count"] = 0
                                cand["spin_count"] += 1
                                if cand["spin_count"] >= 5 and now >= cand["end_time"] - hit_win:  # Require 5 spins
                                    cand["hit"] = True
                                    cand["removed"] = True
                                    hits += 1; combo += 1
                                    # Upgrade: scoring and health
                                    if self.s.enable_scoring:
                                        hit_score = calculate_hit_score(best_abs, hit_win, combo, self.s.combo_multiplier)
                                        self.score += hit_score
                                    self.combo += 1
                                    self.max_combo = max(self.max_combo, self.combo)
                                    if self.s.enable_health:
                                        self.health = min(self.s.max_health, self.health + self.s.health_recovery)
                                    if self.s.sound_effects:
                                        self.audio.play_hit_sound()
                                    # Add hit effect
                                    self.hit_effects.append((cand["x"], cand["y"], 1.0, 0.3))
                            else:
                                # Regular tap
                                cand["hit"]=True; cand["removed"]=True
                                hits += 1; combo += 1
                                # Upgrade: scoring and health
                                if self.s.enable_scoring:
                                    hit_score = calculate_hit_score(best_abs, hit_win, combo, self.s.combo_multiplier)
                                    self.score += hit_score
                                self.combo += 1
                                self.max_combo = max(self.max_combo, self.combo)
                                if self.s.enable_health:
                                    self.health = min(self.s.max_health, self.health + self.s.health_recovery)
                                if self.s.sound_effects:
                                    self.audio.play_hit_sound()
                                # Add hit effect
                                self.hit_effects.append((cand["x"], cand["y"], 1.0, 0.3))
                else:
                    # autoriser quitter
                    k = read_key(timeout=0.0)
                    if k=='ESC': break

                judged = hits + misses
                accuracy = 100.0 * (hits / judged) if judged>0 else (100.0 if (auto_play or replay_mode) else 0.0)
                progress_ratio = min(1.0, now / max(1e-6, duration))

                # build drawables
                drawables=[]
                for n in runtime_notes:
                    if n["removed"]: continue
                    spawn = n["due"] - pre; death = n["due"] + post
                    if n["type"] in ["slider", "spinner"]:
                        death = max(death, n["end_time"] + post)
                    if now < spawn or now > death: continue

                    if n["type"] == "slider":
                        # Slider: show extended box
                        box = ["┌─────┐", f"│  {n['key']}  │", "└─────┘"]
                    elif n["type"] == "spinner":
                        # Spinner: show circular symbol
                        spin_char = "○" if int(now * 10) % 2 == 0 else "●"
                        box = ["┌───┐", f"│ {spin_char} │", "└───┘"]
                    else:
                        # Tap: normal box
                        box = big_key_box(n["key"])

                    if n["miss"] and now <= death: state="red"
                    elif n["type"] in ["slider", "spinner"] and n["held"]: state="blue"
                    else: state = "green" if abs(now - n["due"]) <= hit_win else "white"
                    drawables.append((n["x"], n["y"]-1, box, state))

                # Update hit effects
                new_effects = []
                for x, y, intensity, time_left in self.hit_effects:
                    new_time = time_left - dt
                    if new_time > 0:
                        new_intensity = max(0, intensity - dt * 3)
                        new_effects.append((x, y, new_intensity, new_time))
                self.hit_effects = new_effects

                # draw
                self.draw_scene(cols, rows, drawables, self.hit_effects)
                self.draw_hud(cols, rows, accuracy, now, os.path.basename(song_path), level, bpm_avg, hits, misses, combo, total_notes, progress_ratio, fps)

                # fps
                nowp = time.perf_counter()
                dt = nowp - last_frame; last_frame = nowp
                if dt>0:
                    fps = (0.9*fps + 0.1*(1.0/dt)) if fps else (1.0/dt)
                time.sleep(max(0.0, (1/60.0) - (time.perf_counter()-nowp)))

        finally:
            self.audio.stop()

        accuracy = (100.0*hits/(hits+misses)) if (hits+misses)>0 else (100.0 if (auto_play or replay_mode) else 0.0)
        grade = get_grade(accuracy, self.max_combo, total_notes) if self.s.enable_scoring else "N/A"
        result = {
            "hits": hits,
            "misses": misses,
            "accuracy": accuracy,
            "score": self.score,
            "max_combo": self.max_combo,
            "grade": grade,
            "final_health": self.health,
            "duration": duration,
            "seed": seed,
            "notes": [{"t": round(n["due"],6), "key": n["key"], "type": n["type"], "hit": n["hit"]} for n in runtime_notes]
        }
        logging.info(f"Round completed: score={self.score}, accuracy={result['accuracy']:.1f}%, grade={result['grade']}")
        return result

# ====== Replays ======
def next_replay_name() -> str:
    existing = [name for name in os.listdir(REPLAYS_DIR) if name.lower().startswith("replay_") and name.lower().endswith(".json")]
    nums=[]
    for n in existing:
        try:
            core = os.path.splitext(n)[0]
            num = int(core.split("_")[1]); nums.append(num)
        except Exception:
            pass
    nxt = (max(nums)+1) if nums else 1
    return f"replay_{nxt:03d}.json"

def save_replay(map_data: dict, level: str, result: dict, accuracy: float, settings: Settings, custom_name: Optional[str]=None):
    if settings.auto_play:
        return False, "Auto-play enabled: replay cannot be saved."
    name = custom_name.strip() if custom_name else next_replay_name()
    if not name.lower().endswith(".json"): name += ".json"
    out = {
        "version": 2,  # Updated version for upgrades
        "song_file": map_data["song_file"],
        "level": level,
        "duration": map_data["duration"],
        "bpm_avg": map_data["bpm_avg"],
        "seed": result["seed"],
        "settings": {
            "pre_spawn": settings.pre_spawn,
            "post_miss": settings.post_miss,
            "hit_window_ms": map_data["levels"][level]["hit_window_ms"],
            "enable_scoring": settings.enable_scoring,
            "enable_health": settings.enable_health,
            "max_health": settings.max_health
        },
        "accuracy": round(float(accuracy),2),
        "score": result.get("score", 0),
        "max_combo": result.get("max_combo", 0),
        "grade": result.get("grade", "N/A"),
        "final_health": result.get("final_health", settings.max_health),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # notes with whether they were hit and when (approx = due for hits we registered on key; we don't store precise time, but due is enough to replay).
        "notes": [{"t": n["t"], "key": n["key"], "type": n["type"], "hit": n["hit"], "hit_time": (n["t"] if n["hit"] else None)} for n in result["notes"]]
    }
    path = os.path.join(REPLAYS_DIR, name)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return True, path

def load_replay(path: str) -> Optional[dict]:
    try:
        with open(path,"r",encoding="utf-8") as f:
            data = json.load(f)
        # Backward compatibility for version 1
        if data.get("version", 1) == 1:
            data["score"] = data.get("score", 0)
            data["max_combo"] = data.get("max_combo", 0)
            data["grade"] = data.get("grade", "N/A")
            data["final_health"] = data.get("final_health", 100)
            data["settings"]["enable_scoring"] = data["settings"].get("enable_scoring", True)
            data["settings"]["enable_health"] = data["settings"].get("enable_health", True)
            data["settings"]["max_health"] = data["settings"].get("max_health", 100)
        return data
    except Exception:
        return None

# ====== Flow ======
def play_song_flow(settings: Settings):
    song = pick_song_menu()
    if not song: return
    # assurer map
    mp = load_map_for_song(song)
    if (not mp) or (os.path.getmtime(map_path_for_song(song)) < os.path.getmtime(song)):
        make_map_for_song(song, settings)
        mp = load_map_for_song(song)
    level = pick_level_menu()
    if not level: return
    # résumé
    clear_screen()
    cols, rows = term_size()
    head = "Ready!"
    draw_text(2, center_x(len(head), cols), head, fg=Fore.GREEN, style=Style.BRIGHT)
    info = f"Song: {os.path.basename(song)}"
    draw_text(4, center_x(len(info), cols), info, fg=Fore.WHITE)
    sub = f"Duration: {mp['duration']:.1f}s   BPM≈{mp['bpm_avg']:.1f}   Notes({level}): {len(mp['levels'][level]['notes'])}"
    draw_text(5, center_x(len(sub), cols), sub, fg=Fore.WHITE)
    wait_key_any("ENTER to start, any key...")
    # play
    g = Game(settings)
    res = g.play_round(song, mp, level, auto_play=settings.auto_play, replay_driver=None)
    # results
    clear_screen()
    title = "Results"
    draw_text(2, center_x(len(title), cols), title, fg=Fore.MAGENTA, style=Style.BRIGHT)
    draw_text(4, center_x(30, cols), f"Accuracy: {res['accuracy']:.2f}%", fg=Fore.GREEN)
    hits = sum(1 for n in res["notes"] if n["hit"])
    misses = len(res["notes"]) - hits
    draw_text(5, center_x(30, cols), f"Hits: {hits}   Misses: {misses}", fg=Fore.WHITE)
    # Upgrade: display score, grade, max combo, final health
    if settings.enable_scoring:
        draw_text(6, center_x(30, cols), f"Score: {res['score']:,}", fg=Fore.YELLOW)
        draw_text(7, center_x(30, cols), f"Grade: {res['grade']}", fg=Fore.CYAN)
    if settings.enable_health:
        draw_text(8 if settings.enable_scoring else 6, center_x(30, cols), f"Max Combo: {res['max_combo']}", fg=Fore.WHITE)
        draw_text(9 if settings.enable_scoring else 7, center_x(30, cols), f"Final Health: {res['final_health']}/{settings.max_health}", fg=Fore.MAGENTA)
    else:
        draw_text(6, center_x(30, cols), f"Max Combo: {res['max_combo']}", fg=Fore.WHITE)
    # Save replay?
    if settings.auto_play:
        draw_text(10 if (settings.enable_scoring or settings.enable_health) else 7, center_x(52, cols), "Auto-play was ON — replay not savable.", fg=Fore.YELLOW)
        wait_key_any("Press any key...")
        return
    # prompt save
    prompt_line = 10 if (settings.enable_scoring or settings.enable_health) else 7
    draw_text(prompt_line, center_x(60, cols), "Save replay?  ENTER = Yes  •  ESC = No", fg=Fore.CYAN)
    while True:
        k = read_key()
        if k=='ENTER':
            # ask name
            default = next_replay_name()
            draw_text(9, center_x(60, cols), f"Name (empty = {default}): ", fg=Fore.WHITE)
            # simple line input (blocking)
            reset_color(); move_to(9, center_x(60, cols)+len(f"Name (empty = {default}): "))
            name = input().strip()
            ok, path = save_replay(mp, level, res, res["accuracy"], settings, custom_name=(name if name else default))
            clear_screen()
            if ok:
                draw_text(6, center_x(len("Replay saved!"), cols), "Replay saved!", fg=Fore.GREEN, style=Style.BRIGHT)
                draw_text(8, center_x(len(path), cols), path, fg=Fore.WHITE)
            else:
                draw_text(6, center_x(len(str(path)), cols), str(path), fg=Fore.RED)
            wait_key_any("Press any key...")
            return
        elif k=='ESC':
            return

def play_replay_flow(settings: Settings):
    rp = pick_replay_menu()
    if not rp: return
    data = load_replay(rp)
    if not data:
        wait_key_any("Failed to load replay.")
        return
    song_file = data.get("song_file")
    level = data.get("level","Normal")
    song_path = os.path.join(SONGS_DIR, song_file)
    if not os.path.exists(song_path):
        wait_key_any(f"Song file not found: {song_file}")
        return
    # construire map minimal pour driver (hit window du replay)
    map_stub = {
        "song_file": song_file,
        "duration": data.get("duration", 120.0),
        "bpm_avg": data.get("bpm_avg", 120.0),
        "levels": {
            level: {
                "hit_window_ms": data.get("settings",{}).get("hit_window_ms", 90),
                "notes": [{"t": n["t"], "key": n["key"], "type": n.get("type","tap")} for n in data.get("notes",[])]
            }
        }
    }
    driver = {
        "seed": data.get("seed", 123456),
        "notes": [{"t": n["t"], "key": n["key"], "type": n.get("type","tap"), "hit": n.get("hit",False), "hit_time": n.get("hit_time")} for n in data.get("notes",[])]
    }
    # jouer en mode replay (inputs ignorés)
    g = Game(settings)
    res = g.play_round(song_path, map_stub, level, auto_play=False, replay_driver=driver)
    # fin
    clear_screen()
    cols,_ = term_size()
    head = "Replay finished"
    draw_text(3, center_x(len(head), cols), head, fg=Fore.MAGENTA, style=Style.BRIGHT)
    acc = f"Accuracy (replay): {data.get('accuracy','?')}%"
    draw_text(5, center_x(len(acc), cols), acc, fg=Fore.GREEN)
    # Upgrade: display replay stats
    if data.get("version", 1) >= 2:
        score = f"Score: {data.get('score', 0):,}"
        draw_text(6, center_x(len(score), cols), score, fg=Fore.YELLOW)
        grade = f"Grade: {data.get('grade', 'N/A')}"
        draw_text(7, center_x(len(grade), cols), grade, fg=Fore.CYAN)
        combo = f"Max Combo: {data.get('max_combo', 0)}"
        draw_text(8, center_x(len(combo), cols), combo, fg=Fore.WHITE)
        health = f"Final Health: {data.get('final_health', 100)}/{data['settings'].get('max_health', 100)}"
        draw_text(9, center_x(len(health), cols), health, fg=Fore.MAGENTA)
    wait_key_any("Press any key...")

# ====== Main menu ======
def main_menu():
    s = load_settings()
    # pré-scan
    scan_and_build_maps(s)

    while True:
        files = list_songs(SONGS_DIR)
        maps_count = len([n for n in os.listdir(MAPS_DIR) if n.endswith(".json")])
        reps_count = len([n for n in os.listdir(REPLAYS_DIR) if n.endswith(".json")])

        opts = [
            f"Play  (songs: {len(files)}, maps: {maps_count})",
            f"Play Replay  (replays: {reps_count})",
            "Tutorial",
            "Re-scan songs & rebuild maps",
            "Settings",
            "Quit"
        ]
        choice = menu_select("osu! terminal (ZQSD) — Main Menu", opts,
                             footer="↑/↓ select • ENTER choose • ESC exit")
        if choice is None or choice==5:
            clear_screen(); print("Bye!"); break
        if choice==0:
            play_song_flow(s)
        elif choice==1:
            play_replay_flow(s)
        elif choice==2:
            if tutorial_mode(s):
                # Start tutorial with a simple practice round
                tutorial_song = pick_song_menu()
                if tutorial_song:
                    mp = load_map_for_song(tutorial_song)
                    if mp:
                        g = Game(s)
                        res = g.play_round(tutorial_song, mp, "Easy", auto_play=False, replay_driver=None)
                        # Show tutorial results
                        clear_screen()
                        cols, rows = term_size()
                        draw_text(2, center_x(20, cols), "Tutorial Complete!", fg=Fore.GREEN, style=Style.BRIGHT)
                        draw_text(4, center_x(30, cols), f"Final Score: {res['score']:,}", fg=Fore.YELLOW)
                        draw_text(5, center_x(30, cols), f"Accuracy: {res['accuracy']:.1f}%", fg=Fore.CYAN)
                        wait_key_any("Press any key to return to menu")
        elif choice==3:
            scan_and_build_maps(s); wait_key_any("Done.")
        elif choice==4:
            settings_screen(s)

if __name__ == "__main__":
    try:
        logging.info("Terminal osu! started")
        main_menu()
    except KeyboardInterrupt:
        logging.info("Game interrupted by user")
        hidectrl(False); reset_color(); print("\nExit.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        hidectrl(False); reset_color()
        print(f"\nAn error occurred: {e}")
        print("Check term_osu.log for details.")
        sys.exit(1)
