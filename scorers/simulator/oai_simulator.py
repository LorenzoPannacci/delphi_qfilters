from typing import Union

import torch

from delphi.latents.latents import ActivatingExample, NonActivatingExample

from ..scorer import Scorer, ScorerResult
from .oai_autointerp import (
    ActivationRecord,
    ExplanationNeuronSimulator,
    LogprobFreeExplanationTokenSimulator,
    simulate_and_score,
)


class OpenAISimulator(Scorer):
    """
    Simple wrapper for the LogProbFreeExplanationTokenSimulator.
    """

    name = "simulator"

    def __init__(
        self,
        client,
        tokenizer,
        all_at_once=True,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.all_at_once = all_at_once

    async def __call__(self, record):  # type: ignore
        # Simulate and score the explanation.
        cls = (
            ExplanationNeuronSimulator
            if self.all_at_once
            else LogprobFreeExplanationTokenSimulator
        )
        simulator = cls(
            self.client,
            record.explanation,
        )

        valid_activation_records = self.to_activation_records(record.test)  # type: ignore
        if len(record.not_active) > 0:
            non_activation_records = self.to_activation_records(record.not_active)  # type: ignore
        else:
            non_activation_records = []

        result = await simulate_and_score(
            simulator, valid_activation_records, non_activation_records
        )

        return ScorerResult(
            record=record,
            score=result,
        )

    def to_activation_records(
        self, examples: list[Union[ActivatingExample, NonActivatingExample]]
    ) -> list[ActivationRecord]:
        # Filter Nones
        result = []
        for example in examples:
            if example is None:
                continue

            if example.normalized_activations is None:
                # Use zeros for non-activating examples
                example.normalized_activations = torch.zeros_like(example.activations)

            result.append(
                ActivationRecord(
                    self.tokenizer.batch_decode(example.tokens),
                    example.normalized_activations.half(),
                    quantile=(
                        example.quantile
                        if isinstance(example, ActivatingExample)
                        else None
                    ),
                )
            )

        return result
