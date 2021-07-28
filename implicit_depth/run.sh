xhost +local:root
sudo docker run --gpus 1 -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all --privileged --device=/dev:/dev -h $HOSTNAME implicit_depth
