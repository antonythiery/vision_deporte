### Lancer une calibration : 
<code>uv run src/vision_deporte/tools/cameraCalibration.py</code> 

### Lancer une détection + visualisation :
<code>uv run src/vision_deporte/main.py </code>

### Lancer un entrainement SB PPO:
<code>uv run uv run src/vision_deporte/SBenv.py -t</code>

### Lancer une visualisation SB PPO:
<code>uv run uv run src/vision_deporte/SBenv.py -v</code>

### Tensorboard PPO trainsing :
<code>uv run tensorboard --logdir src/vision_deporte/tensorboard_log</code>