# Sagemaker custom training template

Example script to run a job on the SageMaker Training.

Sagemaker Training offers a convenient environment to run a job in a docker container. You can perform ML model training, but it's not limited to that.

You can use [pre-build container image maintained by AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) or build your own.

## Run a job on SageMaker Training
The following environment variables are required
| Variables                 |                                                                                                                                  |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| AWS_S3_BUCKET             | S3 bucket to place training codes.                                                                                                |
| AWS_SM_EXECUTION_ROLE_ARN | IAM role ARN used in a training job. Appropriate access permission is necessary. |
| AWS_ECR_REPOSITORY        | ECR repository for training docker image. You can use your own image or pre-built SageMaker images.                              |
| IMAGE_TAG                 | Tag of the image.                                                                                                                |


Run
```
python main.py
```

Run with docker build and push
```
python main.py --build-image
```
The ECR repository should be prepared in advance.

Dependencies and training scripts are read from `trainer` directory, so the image build is not necessary every time you add a library or modify training codes.


## Prepare training codes
The `trainer` directory contains actual training codes. `train.py` is the entry point script.

In `trainer`, dependencies should be written in `requirements.txt`.
