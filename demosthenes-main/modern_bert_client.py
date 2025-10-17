"""Utility for retrieving Modern-BERT embeddings from a SageMaker TEI endpoint."""

from __future__ import annotations

import json
import os
from typing import Sequence

import boto3
import numpy as np
from botocore.exceptions import ClientError


class ModernBertEndpoint:
    """Thin wrapper around a SageMaker endpoint serving Modern-BERT embeddings."""

    def __init__(self, endpoint_name: str | None = None, *, batch_size: int = 16):
        self.endpoint_name = endpoint_name or os.getenv("MODERN_BERT_ENDPOINT")
        if not self.endpoint_name:
            raise RuntimeError("MODERN_BERT_ENDPOINT is required")

        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        session = boto3.Session(region_name=region)
        self.sagemaker = session.client("sagemaker")
        self.runtime = session.client("sagemaker-runtime")
        self.batch_size = batch_size

        try:
            self.sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
        except ClientError as error:  # pragma: no cover - requires live AWS
            if error.response.get("Error", {}).get("Code") == "ValidationException":
                raise RuntimeError(
                    f"SageMaker endpoint '{self.endpoint_name}' not found."
                ) from error
            raise

    def embeddings(self, texts: Sequence[str], *, normalize: bool = True) -> np.ndarray:
        """Fetch embeddings for a batch of texts using the configured endpoint."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        outputs: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = json.dumps({"inputs": batch})
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=payload,
            )
            body = response["Body"].read().decode("utf-8")
            outputs.extend(json.loads(body))

        array = np.asarray(outputs, dtype=np.float32)
        if not normalize:
            return array

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return array / norms


_default_endpoint: ModernBertEndpoint | None = None


def get_default_endpoint() -> ModernBertEndpoint:
    """Return a cached Modern-BERT endpoint client."""

    global _default_endpoint
    if _default_endpoint is None:
        _default_endpoint = ModernBertEndpoint()
    return _default_endpoint

