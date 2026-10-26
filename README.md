## Lancer une calibration : 
uv run src/vision_deporte/tools/cameraCalibration.py 

## Lancer une détection + visualisation :
uv run src/vision_deporte/main.py 

## Lancer un entrainement SB PPO:
uv run uv run src/vision_deporte/SBenv.py -t

## Lancer une visualisation SB PPO:
uv run uv run src/vision_deporte/SBenv.py -v

## Tensorboard PPO trainsing :
uv run tensorboard --logdir src/vision_deporte/tensorboard_log