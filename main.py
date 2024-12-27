import base64
import json
import os
import tarfile
from datetime import datetime
from logging import getLogger
from typing import Any, Self, TypedDict

import boto3
import docker
import sagemaker
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

SOURCE_DIR = "trainer"
S3_CODE_PREFIX = "code"
S3_CODE_BASENAME = "trainer"
TRAIN_CONFIG_FILENAME = "config.yaml"

logger = getLogger(__name__)
sagemaker_session = sagemaker.session.Session()  # type: ignore
ecr_client = boto3.client("ecr")
docker_client = docker.from_env()


class SagemakerTrainingSettings(BaseSettings):
    AWS_S3_BUCKET: str
    AWS_SM_EXECUTION_ROLE_ARN: str
    AWS_ECR_REPOSITORY: str
    IMAGE_TAG: str
    RUN_ID: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S%f")
    )


def json_encode_hyperparameters(hyperparameters: dict[str, Any]) -> dict[str, str]:
    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}


class EstimatorConfig(BaseModel):
    entry_point: str
    instance_count: int
    instance_type: str
    base_job_name: str
    use_spot_instances: bool
    max_run: int | None = None
    max_wait: int | None = None
    hyperparameters: dict[str, Any]

    def to_estimator_args(self) -> dict[str, Any]:
        args = self.model_dump()
        if not self.use_spot_instances:
            args.pop("max_run")
            args.pop("max_wait")
        args["hyperparameters"] = json_encode_hyperparameters(args["hyperparameters"])
        return args

    @classmethod
    def from_yaml(cls, filename: str) -> Self:
        with open(filename, "r") as f:
            return cls(**yaml.safe_load(f))


def create_tar_file(source_dir: str, target_filename: str):
    with tarfile.open(target_filename, "w:gz") as tar:
        for root, _, files in os.walk(source_dir):
            if root.endswith(".venv"):
                continue
            for file in files:
                full_path = os.path.join(root, file)
                tar.add(full_path, arcname=os.path.relpath(full_path, source_dir))


def prepare_training_code_on_s3(sm_settings: SagemakerTrainingSettings) -> str:
    trainer_filename = f"{S3_CODE_BASENAME}_{sm_settings.RUN_ID}.tar.gz"
    try:
        create_tar_file(SOURCE_DIR, trainer_filename)
        sources = sagemaker_session.upload_data(
            trainer_filename, sm_settings.AWS_S3_BUCKET, S3_CODE_PREFIX
        )
    finally:
        os.remove(trainer_filename)

    return sources


class DockerAuthConfig(TypedDict):
    username: str
    password: str
    registry: str


class ECRLoginError(Exception):
    pass


def login_to_ecr(docker_client: docker.DockerClient) -> DockerAuthConfig:
    response = ecr_client.get_authorization_token()
    auth_data = response["authorizationData"][0]

    if "authorizationToken" not in auth_data:
        raise ECRLoginError("No authorizationToken in response")
    token = auth_data["authorizationToken"]

    username, password = base64.b64decode(token).decode("utf-8").split(":")

    if "proxyEndpoint" not in auth_data:
        raise ECRLoginError("No proxyEndpoint in response")
    registry = auth_data["proxyEndpoint"]

    logger.info(f"Logging in to {registry}")
    docker_client.login(username, password, registry=registry, reauth=True)

    return {"username": username, "password": password, "registry": registry}


def build_docker_image(repository: str, tag: str) -> str:
    image_uri = f"{repository}:{tag}"
    docker_client.images.build(
        path=".",
        dockerfile="Dockerfile",
        tag=image_uri,
        platform="linux/amd64",
        quiet=False,
    )
    return image_uri


def push_docker_image(image_uri: str, auth_config: DockerAuthConfig):
    resp = docker_client.images.push(
        image_uri, stream=True, decode=True, auth_config=auth_config
    )
    for chunk in resp:
        logger.debug(chunk)
        if "errorDetail" in chunk:
            raise RuntimeError(chunk["errorDetail"]["message"])


def build_and_push_image(repository: str, tag: str) -> str:
    auth_config = login_to_ecr(docker_client)
    container_image_uri = build_docker_image(repository, tag)
    push_docker_image(container_image_uri, auth_config)
    return container_image_uri


def prepare_training_image(
    sm_settings: SagemakerTrainingSettings, build_image: bool
) -> str:
    if build_image:
        return build_and_push_image(
            sm_settings.AWS_ECR_REPOSITORY, sm_settings.IMAGE_TAG
        )
    else:
        return f"{sm_settings.AWS_ECR_REPOSITORY}:{sm_settings.IMAGE_TAG}"


def main(build_image: bool):
    sm_settings = SagemakerTrainingSettings()  # type: ignore
    logger.info(f"Run ID: {sm_settings.RUN_ID}")

    container_image_uri = prepare_training_image(sm_settings, build_image)
    trainer_sources = prepare_training_code_on_s3(sm_settings)
    trainer_config = EstimatorConfig.from_yaml(TRAIN_CONFIG_FILENAME)

    logger.info("Start training")
    estimator = sagemaker.estimator.Estimator(
        container_image_uri,
        sm_settings.AWS_SM_EXECUTION_ROLE_ARN,
        source_dir=trainer_sources,
        **trainer_config.to_estimator_args(),
    )
    estimator.fit(wait=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-image", action="store_true")
    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
