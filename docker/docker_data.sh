#!/bin/bash

echo ""
echo "Run docker container detached, add data volumes, and execute python script."
echo ""

docker run --gpus "device=0" -d -it --rm --name create_vtr_dataset --shm-size=64g \
-v <path_to_intheedark_dataset_on_computer>:/home/<user_name>/data/inthedark:ro \
-v <path_to_multiseason_dataset_on_computer>:/home/<user_name>/data/multiseason:ro \
-v /home/<user_name>/code:/home/<user_name>/code \
-v /home/<user_name>/results:/home/<user_name>/results \
-v /home/<user_name>/networks:/home/<user_name>/networks \
-v /home/<user_name>/datasets:/home/<user_name>/datasets \
docker_image_name \
bash -c "cd /home/<user_name>/code/deep_learned_visual_features && python -m data.build_train_test_dataset_loc --config /home/<user_name>/code/deep_learned_visual_features/config/data.json"
