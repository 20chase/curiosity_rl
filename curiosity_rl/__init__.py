# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms

from curiosity_rl.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from curiosity_rl.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from curiosity_rl.algos.pytorch.sac.sac import sac as sac_pytorch
from curiosity_rl.algos.pytorch.td3.td3 import td3 as td3_pytorch
from curiosity_rl.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from curiosity_rl.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

# Loggers
from curiosity_rl.utils.logx import Logger, EpochLogger

# Version
from curiosity_rl.version import __version__