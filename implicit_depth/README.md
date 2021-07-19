# Docker files for implicit_depth

Use `build.sh` to construct the Docker image and `run.sh` to run the Docker image (on a Unix system).

Follow the instructions from https://github.com/NVlabs/implicit_depth regarding dataset preparation. Be sure to prepare the ClearGrasp Training Dataset, ClearGrasp Testing and Validation Dataset and RGB-D LIDF checkpoints before running `build.sh`.

The directory structure should look as follows:
```
${DATASET_ROOT_DIR}
├── cleargrasp
│   ├── cleargrasp-dataset-train
│   ├── cleargrasp-dataset-test-val
├── checkpoints
```

The live demo has been ported from the ClearGrasp project (https://github.com/Shreeyak/cleargrasp).
