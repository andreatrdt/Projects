
import pandas as pd
import numpy as np

# modify these as needed for your experiments
MARKET_OPEN_SECONDS = 9.5 * 3600
from utils import t_rel
t_start = t_rel(13, 0, 0)
PLOT = True
SAVE_DIR = "cache"
_ANCHOR_DATE = pd.to_datetime("1970-01-01")

# Do ONLY edit the above two lines. The rest of the code is imported by main.ipynb and should not be changed.

COMPONENTS = ["L_b","E_b","HE_b","D_b","L_a","E_a","HE_a","D_a"]
COMPONENTS_4D = ["L_b", "M_b", "L_a", "M_a"]
COMPONENTS_2D = ["E_b", "E_a"]
MACROTYPE  = ["L", "E","HE", "D", "L", "E", "HE", "D"]

BETA_IDX = np.arange(8, dtype=int)
ALPHA_SAME_KEYS = [
    ("L","L"), ("L","E"), ("L","HE"), ("L","D"),
    ("E","L"), ("E","E"), ("E","HE"), ("E","D"),
    ("HE","L"), ("HE","E"), ("HE","HE"), ("HE","D"),
    ("D","L"), ("D","E"), ("D","HE"), ("D","D"),
]
ALPHA_CROSS_KEYS = [
    ("L","E"),
    ("L","HE"),
    ("L","D"),
    ("E","L"),
    ("E","E"),
    ("E","D"),
    ("D","L"),
    ("D","E"),
    ("D","HE"),
    ("D","D"),
]
N_SAME  = 16
N_CROSS = len(ALPHA_CROSS_KEYS)
N_ALPHA = N_SAME + N_CROSS
BETA_SIZE = int(BETA_IDX.max() + 1)