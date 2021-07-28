# Docker files for ClearGrasp

Use `build.sh` to construct the Docker image and `run.sh` to run the Docker image (on a Unix system).

Only depends on the checkpoints of ClearGrasp, download them from http://clkgum.com/shreeyak/cleargrasp-checkpoints. Unzip the downloaded archive so the directory structure looks as follows:
```
This directory (containing the Dockerfile, run.sh, build.sh)
├── cleargrasp-checkpoints
│   ├── mask_segmentation
│   ├── outlines
│   ├── surface_normals
```

Run using `run.sh`. To start the live demo, execute the following commands to start the RealSense preview and ClearGrasp demo:
```
cd cleargrasp/live_demo
./realsense/build/realsense &
QT_X11_NO_MITSHM=1 python live_demo.py -c config/config.yaml
```
