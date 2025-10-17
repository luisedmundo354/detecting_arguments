from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import boto3
import numpy as np
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

@dataclass
class ModernBertConfig:
    endpoint_name: str
    model_id: str = "answerdotai/modernbert-base"
    instance_type: str = "ml.m5.xlarge"
    tei_image_version: str | None = None
    tei_task: str = "text-embedding"
    extra_env: Dict[str, str] | None = None
    role_arn: str | None = None
    region_name: str | None = None


def tei_image_for(instance_type: str, region: str, version: str | None) -> str:
    """Resolve the TEI container image based on hardware capabilities."""
    backend = "huggingface-tei" if instance_type.startswith(("ml.g", "ml.p")) else "huggingface-tei-cpu"
    kwargs = {"backend": backend, "region": region}
    if version:
        kwargs["version"] = version
    return get_huggingface_llm_image_uri(**kwargs)


class ModernBertClient:
    def __init__(self, config: ModernBertConfig):
        session = boto3.Session(region_name=config.region_name)
        self.config = config
        self.session = sagemaker.Session(boto_session=session)
        self.sagemaker = session.client("sagemaker")
        self.runtime = session.client("sagemaker-runtime")
        role = config.role_arn or os.getenv("SAGEMAKER_EXECUTION_ROLE")
        if not role:
            raise RuntimeError("SAGEMAKER_EXECUTION_ROLE is required")
        self.role = role

    def endpoint_exists(self) -> bool:
        try:
            self.sagemaker.describe_endpoint(EndpointName=self.config.endpoint_name)
            return True
        except ClientError as error:
            if error.response["Error"].get("Code") == "ValidationException":
                return False
            raise

    def deploy(self) -> None:
        region = self.config.region_name or self.session.boto_session.region_name
        image_uri = tei_image_for(self.config.instance_type, region, self.config.tei_image_version)
        print(f"Deploying {image_uri}")
        resolved_env = {
            "MODEL_ID": self.config.model_id,
            "HF_MODEL_ID": self.config.model_id,
            "HF_TASK": self.config.tei_task,
        }
        if self.config.extra_env:
            resolved_env.update(self.config.extra_env)
        model = HuggingFaceModel(
            role=self.role,
            image_uri=image_uri,
            env=resolved_env,
            sagemaker_session=self.session,
        )
        model.deploy(
            initial_instance_count=1,
            instance_type=self.config.instance_type,
            endpoint_name=self.config.endpoint_name,
        )

    def embeddings(self, texts: Sequence[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        outputs: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            body = json.dumps({"inputs": batch})
            response = self.runtime.invoke_endpoint(
                EndpointName=self.config.endpoint_name,
                ContentType="application/json",
                Body=body,
            )
            payload = response["Body"].read().decode("utf-8")
            outputs.extend(json.loads(payload))
        array = np.asarray(outputs, dtype=np.float32)
        if not normalize:
            return array
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return array / norms


_default_client: ModernBertClient | None = None


def get_default_client() -> ModernBertClient:
    global _default_client
    if _default_client is None:
        endpoint = os.getenv("MODERN_BERT_ENDPOINT")
        if not endpoint:
            raise RuntimeError("MODERN_BERT_ENDPOINT is required")
        model_id = os.getenv("MODERN_BERT_MODEL_ID", "answerdotai/modernbert-base")
        instance_type = os.getenv("MODERN_BERT_INSTANCE", "ml.m5.xlarge")
        tei_version = os.getenv("MODERN_BERT_TEI_VERSION")
        tei_task = os.getenv("MODERN_BERT_TASK", "text-embedding")
        extra_env: Dict[str, str] | None = None
        env_overrides = os.getenv("MODERN_BERT_ENV")
        if env_overrides:
            try:
                parsed = json.loads(env_overrides)
            except json.JSONDecodeError as error:
                raise RuntimeError("MODERN_BERT_ENV must be valid JSON") from error
            if not isinstance(parsed, dict):
                raise RuntimeError("MODERN_BERT_ENV must decode to a JSON object")
            extra_env = {str(key): str(value) for key, value in parsed.items()}
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        role = os.getenv("MODERN_BERT_ROLE_ARN") or os.getenv("SAGEMAKER_EXECUTION_ROLE")
        config = ModernBertConfig(
            endpoint_name=endpoint,
            model_id=model_id,
            instance_type=instance_type,
            tei_image_version=tei_version,
            tei_task=tei_task,
            extra_env=extra_env,
            region_name=region,
            role_arn=role,
        )
        client = ModernBertClient(config)
        if not client.endpoint_exists():
            client.deploy()
        _default_client = client
    return _default_client
