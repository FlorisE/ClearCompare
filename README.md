# ClearCompare

_work in progress_

Comparisons / benchmarks of different transparent object point cloud completion / pose estimation systems, especially using common household items that have transparency (e.g. plastic bottles).

Docker files and preview tools for:
* implicit\_depth (https://github.com/NVlabs/implicit_depth)

To be added:
* ClearGrasp (https://github.com/Shreeyak/cleargrasp)
* LIT (https://sites.google.com/umich.edu/prolit)

Dataset of household objects with transparency.

Generation tool for locating and replacing missing depth data using scanned models (with paint applied to their transparent areas).

## How to use the Docker files

To use the docker files, navigate to the directory and run `build.sh`. Then if the build is successful run `run.sh`. The docker container should contain everything you need to start the Intel RealSense camera (e.g. `realsense-viewer`) and generate outputs using the specific depth estimator.

If you encounter any problem, feel free to open an issue.
