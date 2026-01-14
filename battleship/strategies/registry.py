from typing import Dict, List


def model_defs() -> List[Dict[str, object]]:
    # Keep in sync with the models supported by strategies.selection.
    return [
        {
            "key": "random",
            "name": "Random",
            "description": "Chooses uniformly among unknown cells.",
            "notes": "A sanity-check baseline. Useful to confirm your world-sampling and UI plumbing work; performance should be the worst.",
        },
        {
            "key": "greedy",
            "name": "Greedy (Probability)",
            "description": "Shoots the cell with the highest immediate hit probability.",
            "notes": "Strong finisher once the posterior is peaked, but can be myopic in open-board hunt mode.",
        },
        {
            "key": "entropy1",
            "name": "Info Gain (1-ply)",
            "description": "Shoots to maximize expected information gain (entropy reduction).",
            "notes": "Great at hunting because it values learning. Can be slightly slower than Greedy at finishing once a ship is basically found.",
        },
        {
            "key": "weighted_sample",
            "name": "Weighted Sample",
            "description": "Randomly samples among cells proportional to their hit probability.",
            "notes": "Adds controlled randomness while still respecting the posterior. If all probabilities are 0, it falls back to uniform random.",
        },
        {
            "key": "softmax_greedy",
            "name": "Softmax (Stochastic)",
            "description": "Selects shots stochastically via a softmax over probabilities (temperature-controlled).",
            "notes": "Temperature -> 0 behaves like Greedy; higher temperature increases exploration. Useful for avoiding deterministic patterns.",
        },
        {
            "key": "parity_greedy",
            "name": "Parity Greedy",
            "description": "Chooses Greedy, but constrained to the parity color with higher total probability mass.",
            "notes": "A parity-flavored variant of Greedy. Helps in hunt mode when minimum ship length makes parity efficient.",
        },
        {
            "key": "random_checkerboard",
            "name": "Random Checkerboard",
            "description": "Randomly hunts only one checkerboard color (parity).",
            "notes": "Simple parity baseline. If your smallest ship length is 2+, parity typically beats pure random.",
        },
        {
            "key": "systematic_checkerboard",
            "name": "Systematic Checkerboard",
            "description": "Deterministically hunts checkerboard cells in row-major order.",
            "notes": "A deterministic parity sweep. Good for debugging and reproducibility; not always optimal vs posterior-driven methods.",
        },
        {
            "key": "diagonal_stripe",
            "name": "Diagonal Stripe",
            "description": "Hunts using diagonal stripes (mod patterns) before relaxing to a wider set.",
            "notes": "Another coverage heuristic. Can reduce redundant coverage early but is typically weaker than posterior-based hunt strategies.",
        },
        {
            "key": "dynamic_parity",
            "name": "Dynamic Parity",
            "description": "Adapts its parity step based on remaining ships (e.g., step=3 when only length-3 remains).",
            "notes": "A practical endgame accelerator. If only a length-3 ship remains, checking every 3rd cell can reduce wasted shots.",
        },
        {
            "key": "hybrid_phase",
            "name": "Hybrid (Hunt/Target)",
            "description": "Uses Info Gain to hunt, switches to Greedy when in target mode.",
            "notes": "A balanced default: information-seeking early, ruthless finishing once the posterior spikes.",
        },
        {
            "key": "endpoint_phase",
            "name": "Endpoint Targeter",
            "description": "Uses Info Gain to hunt; when targeting, prefers endpoints of aligned hit clusters.",
            "notes": "Adds geometric intuition: if hits line up, shoot the ends; if they clump, shoot the frontier. Tunable weights let you trade off geometry vs posterior probability.",
        },
        {
            "key": "center_weighted",
            "name": "Center-Weighted Greedy",
            "description": "Greedy probability, but slightly favors central cells (distance penalty).",
            "notes": "A mild bias that can help in symmetric boards where placements are roughly uniform, but it can hurt if your placement prior is not center-heavy.",
        },
        {
            "key": "adaptive_skew",
            "name": "Adaptive Skew",
            "description": "Greedy probability with an early-game center penalty that fades later.",
            "notes": "Tries to avoid edge-chasing when the board is mostly unknown, then becomes closer to Greedy as information accumulates.",
        },
        {
            "key": "thompson_world",
            "name": "Thompson Sampling",
            "description": "Samples one consistent world and plays as if it is true (then resamples next turn).",
            "notes": "Naturally balances exploration/exploitation. Often very strong without explicit hunt/target heuristics.",
        },
        {
            "key": "two_ply",
            "name": "Two-ply (Legacy)",
            "description": "Evaluates the expected information gain two moves ahead (1‑ply + 2‑ply lookahead).",
            "notes": "A legacy heuristic that simulates the split of possible worlds after a shot, then scores the expected reduction one step further.",
        },
    ]
