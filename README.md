# Sagemaker custom training template

## Prepare training codes
The `trainer` directory contains actual training codes. `train.py` is the entry point script.

In `trainer`, dependencies should be written in `requirements.txt`. We recommend to use `uv` as dependency management, and export its lock information by
```
cd trainer
uv export -o requirements.txt --no-hashes
```

## Run on SageMaker Training Jobs
The following environment variables are required
| Variables                 |                                                                                                                                  |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| AWS_S3_BUCKET             | S3 bucket to place training codes.                                                                                                |
| AWS_SM_EXECUTION_ROLE_ARN | IAM role ARN to execute training jobs. Appropriate access permission is necessary for S3 or other resources used during the job. |
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

