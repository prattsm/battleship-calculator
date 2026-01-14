# Board and cell constants (legacy layout defaults)
BOARD_SIZE = 8

EMPTY = "."
MISS = "o"
HIT = "x"

WORLD_SAMPLE_TARGET = 10000
WORLD_MAX_ATTEMPTS_FACTOR = 30

# Enumeration: if product of allowed placements for ships is <= this,
# we enumerate all valid layouts exactly instead of sampling.
ENUMERATION_PRODUCT_LIMIT = 80000

NO_SHIP = 0
HAS_SHIP = 1

NO_SHOT = 0
SHOT_MISS = 1
SHOT_HIT = 2

DISP_RADIUS = 4

SHIP_ORDER = ["square2", "L3", "line3", "line2"]

# Configuration for tunable models (used by the Model Details / custom sim dialog)
PARAM_SPECS = {
    # Softmax temperature T appears in exp((p - pmax) / T).
    # Since p in [0, 1], values above ~1 quickly approach near-uniform sampling,
    # so we cap sweeps to [0.01, 1.00] to keep the grid meaningful.
    "softmax_greedy": [
        {"key": "temperature", "label": "Temperature (T)", "default": 0.10, "min": 0.01, "max": 1.00, "step": 0.01}
    ],

    # Endpoint Targeter weights are linear multipliers; extreme values just drown out the others.
    # A practical, interpretable range is [0, 2] for each term.
    "endpoint_phase": [
        {"key": "w_prob", "label": "Prob Weight", "default": 1.00, "min": 0.00, "max": 2.00, "step": 0.05},
        {"key": "w_neighbor", "label": "Cluster Bonus", "default": 0.20, "min": 0.00, "max": 2.00, "step": 0.05},
        {"key": "w_endpoint", "label": "Endpoint Bonus", "default": 0.40, "min": 0.00, "max": 2.00, "step": 0.05},
    ],

    # UCB-style exploration bonus coefficient.
    "ucb_explore": [
        {"key": "ucb_c", "label": "Explore Bonus (c)", "default": 0.35, "min": 0.00, "max": 1.00, "step": 0.05}
    ],

    # Rollout model parameters (kept small for practical simulation speed).
    "rollout_mcts": [
        {"key": "rollouts", "label": "Rollouts", "default": 6, "min": 2, "max": 30, "step": 2, "is_int": True},
        {"key": "top_k", "label": "Top Cells", "default": 5, "min": 2, "max": 12, "step": 1, "is_int": True},
        {"key": "max_shots", "label": "Max Sim Shots", "default": 64, "min": 12, "max": 128, "step": 4, "is_int": True},
    ],
}
