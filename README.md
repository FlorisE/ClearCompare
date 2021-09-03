# ClearCompare

_Work in progress_

Comparisons / benchmarks of different transparent object point cloud completion / pose estimation methods, especially using common household items that have transparency (e.g. plastic bottles). We mostly focus on systems that take as single RGB(D) image as input and provide output in a format that facilitates grasping by a robot. Also we only incorporate works that have open source code released.

Thanks to the authors for their work and for releasing their preprints and code.

| Name            | Input type | Output type | Preview tool available | Links |
| --------------- | ---------- | ----------- | ---------------------- | ----- |
| ClearGrasp      | RGBD       | Point Cloud | :heavy_check_mark:     | [:paperclip:](https://sites.google.com/view/cleargrasp) [:notebook:](https://arxiv.org/abs/1910.02550) [:octocat:](https://github.com/Shreeyak/cleargrasp) |
| implicit\_depth | RGBD       | Point Cloud |                        | [:paperclip:](https://research.nvidia.com/publication/2021-03_RGB-D-Local-Implicit) [:notebook:](https://arxiv.org/abs/2104.00622) [:octocat:](https://github.com/NVlabs/implicit_depth) |
| KeyPose         | RGBD       | Keypoints   |                        | [:paperclip:](https://sites.google.com/view/keypose) [:notebook:](https://arxiv.org/abs/1912.02805)  [:octocat:](https://github.com/google-research/google-research/tree/master/keypose) |

:paperclip:: Link to project web site
:notebook:: Link to paper
:octocat:: Link to GitHub repo

For each system we maintain a docker file so it's easy to start and test it. We are also working on creating a preview tool using the Intel RealSense camera for each system, based on the preview team that is part of ClearGrasp.

We also included a Docker image for depth2depth, which is a point cloud completion system discussed in Zhang and Funkhouser: _Deep Depth Completion of a Single RGB-D Image_, CVPR 2018.

To be added:
* Transparent Specular Grasping (https://sites.google.com/view/transparent-specular-grasping)

Some other recent works that are of interest:
* LIT (https://sites.google.com/umich.edu/prolit) (dataset available but no source code available)
* Xu et al., “6DoF Pose Estimation of Transparent Object from a Single RGB-D Image”. Sensors, 2020. (no source code available)
* Li, Yen and Chandraker: “Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes”. CVPR 2020. (multi-view reconstruction method, dataset and source code available)

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
