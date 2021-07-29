# ClearCompare

_Work in progress, the only reason why it is public is so we can easily clone it from a docker image... If you want to use this in your own research please message us._

Comparisons / benchmarks of different transparent object point cloud completion / pose estimation systems, especially using common household items that have transparency (e.g. plastic bottles).

Docker files and (live) preview tools for:
* ClearGrasp (https://github.com/Shreeyak/cleargrasp) - preview tool included out of the box
* implicit\_depth (https://github.com/NVlabs/implicit_depth) - preview tool being ported from ClearGrasp
* KeyPose (https://arxiv.org/abs/1912.02805) - no preview tool yet

To be added:
* LIT (https://sites.google.com/umich.edu/prolit)

Potentially more to be added in the future, as this is a very active research field.

We will also add a dataset of household objects with transparency and a generation tool for locating and replacing missing depth data using scanned models (we will apply paint to their transparent areas to make them scannable).

Work supported by the Japan Science and Technology Agency (JST) Moonshot project.

## How to use the Docker files

To use the docker files, navigate to the directory and run `build.sh`. Then if the build is successful run `run.sh`. The docker container should contain everything you need to start the Intel RealSense camera (e.g. `realsense-viewer`) and generate outputs using the specific depth estimator.

If you encounter any problem, feel free to open an issue.

## FAQ

**Why do you use Conda within Docker?**

Docker containers are already mostly separate from the host system, but we still use Conda within them to separate project specific code from the container level installation of Python.

**How about CUDA?**

We try to maintain containers for the latest version of CUDA. Use `nvidia-docker` (https://github.com/NVIDIA/nvidia-docker) and make sure your graphics card has the Ampere architecture.
