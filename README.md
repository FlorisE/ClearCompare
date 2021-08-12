# ClearCompare

_Work in progress, the only reason why it is public is so we can easily clone it from a docker image... If you want to use this in your own research please message us._

Comparisons / benchmarks of different transparent object point cloud completion / pose estimation methods, especially using common household items that have transparency (e.g. plastic bottles).

Thanks to the authors for their work and for releasing their preprints and code.

| Name            | Output type | Preview tool available | Links |
| --------------- | ----------- | ---------------------- | ----- |
| ClearGrasp      | Point Cloud | :heavy_check_mark:     | [:paperclip:](https://sites.google.com/view/cleargrasp) [:notebook:](https://arxiv.org/abs/1910.02550) [:octocat:](https://github.com/Shreeyak/cleargrasp) |
| implicit\_depth | Point Cloud |                        | [:paperclip:](https://research.nvidia.com/publication/2021-03_RGB-D-Local-Implicit) [:notebook:](https://arxiv.org/abs/2104.00622) [:octocat:](https://github.com/NVlabs/implicit_depth) |
| KeyPose         | Keypoints   |                        | [:paperclip:](https://sites.google.com/view/keypose) [:notebook:](https://arxiv.org/abs/1912.02805)  [:octocat:](https://github.com/google-research/google-research/tree/master/keypose) |

:paperclip:: Link to project web site
:notebook:: Link to paper
:octocat:: Link to GitHub repo

To be added:
* LIT (https://sites.google.com/umich.edu/prolit)
* Transparent Specular Grasping (https://sites.google.com/view/transparent-specular-grasping)

Some more classical works:
* Saxena, Driemeyer, and Ng, “Robotic Grasping of Novel Objects Using Vision.” The International Journal of Robotics Research, 2008. (no source code available)
* Lysenkov and Rabaud, “Pose Estimation of Rigid Transparent Objects in Transparent Clutter.” ICRA 2013. (http://wg-perception.github.io/transparent_objects/)


Potentially more to be added in the future, as this is a very active research field. We are mostly focussing on deep learning approaches vs approaches using hardcoded features. While there are some non-learning approaches that seem to work decently, they often require a known model (or a small set of known models) during execution time.

Benchmarks will be aimed at simulated and real grasps, with state of the art methods applied on the above methods to translate their output into grasps (where needed).

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
