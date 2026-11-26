from envs.CardBoardEnv import RobotEnv as CardBoardenv
from envs.MoveEnv import RobotEnv as MoveEnv

env = 0

if env == 0:
    RobotEnv = CardBoardenv
    agentfile = "CardboardBox"
elif env == 1:
    RobotEnv = MoveEnv
    agentfile = "Move"