#!/usr/bin/env bash
set -e

cd ~/ws_daniel
source daniel_setup.bash

PARAM_FILE="$HOME/ws_daniel/src/ur_softhand_dual/src/ur_dual_pick_place/config/oak_depth_octomap_tuned.yaml"

echo "[INFO] Cargando parámetros desde:"
echo "$PARAM_FILE"

ros2 param load /oak_cam/oak "$PARAM_FILE"

echo "[INFO] Reiniciando pipeline de OAK para aplicar parámetros i_..."
ros2 service call /oak_cam/oak/stop_camera std_srvs/srv/Trigger "{}"
sleep 2
ros2 service call /oak_cam/oak/start_camera std_srvs/srv/Trigger "{}"

echo "[OK] Parámetros aplicados."
