# Docker files for implicit_depth

Use `build.sh` to construct the Docker image and `run.sh` to run the Docker image (on a Unix system).

Follow the instructions from https://github.com/NVlabs/implicit_depth regarding dataset preparation. Be sure to prepare the ClearGrasp Training Dataset, ClearGrasp Testing and Validation Dataset, RGB-D LIDF checkpoints and Omniverse Object Dataset before running `build.sh`.
You will need a lot of free space (~1TB). 

The directory structure should look as follows:
```
This directory (containing the Dockerfile, run.sh, build.sh)
├── checkpoints
├── cleargrasp
│   ├── cleargrasp-dataset-train
│   ├── cleargrasp-dataset-test-val
├── omniverse_v3-train-20200904 (extract the downloaded tar)
├── omniverse_v3-train-20200910 (extract the downloaded tar)

```

The live demo has been ported from the ClearGrasp project (https://github.com/Shreeyak/cleargrasp).
