import sys
# ROOM CONFIG
M_MIN           = 5000 
M_MAX           = 8000
N_MIN           = 5000
N_MAX           = 18000
GRID_SIZE       = 32
PIXEL2MM        = 50


# FURNITURE CONFIG
# thống kê theo database
# W_MIN         = 20
# W_MAX         = 4800
# D_MIN         = 33
# D_MAX         = 1803

# config thực tế
W_MIN           = 0
W_MAX           = max(M_MAX, N_MAX)
D_MIN           = 0
D_MAX           = max(M_MAX, N_MAX)


# HYPERPARAMETER CONFIG
GAMMA           = 0.99          # discount factor
GAE_LAMBDA      = 0.95          # GAE λ
ACTOR_LR        = 1e-4          # learning rate actor
CRITIC_LR       = 1e-3          # learning rate critic
CLIP_RANGE      = 0.2           # PPO clip ratio
TOTAL_EPOCHS    = 500_000
PENALTY         = 10           # penalty
STEPS_PER_UPDATE= 2048          # rollout length mỗi update
BATCH_SIZE      = 64
N_EPOCHS        = 10            # số epoch trong mỗi update PPO
ENT_COEF        = 0.0           # entropy coef (ce)
VF_COEF         = 0.5           # value function coef (cv)
MAX_GRAD_NORM   = 0.5
SEED            = 42

# SYSTEM CONFIG
MODEL_FOLDER    = "./models"
LOG_FOLDER      = "logs"
INF_POS_NUM     = sys.maxsize
NUM_ENVS        = 12             # số môi trường song song (tùy chỉnh)