from __future__ import annotations

from medreason_docker.schema import MedReasonCase, MedReasonPrediction
from medreason_docker.systems.base import MedReasonSystem


class CustomMedReasonSystem(MedReasonSystem):
    """Participant implementation entry point.

    Replace this class with your complete MedReason system. Your system may be a
    single MLLM, an ensemble, a retrieval-augmented workflow, an agentic pipeline,
    or any other self-contained system.

    Official evaluation constraints:
    - The system must run inside the submitted Docker container.
    - Do not rely on internet access, external APIs, or manual interaction.
    - Return one prediction for every input case.
    - For MCQ cases, `answer` must be one of the official option labels.
    - For open-ended cases, both `reasoning_trace` and `answer` are required.
    """

    def setup(self) -> None:
        # Load your models, processors, retrieval indices, or other resources here.
        # Example:
        #   self.model = ...
        #   self.processor = ...
        pass

    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        # TODO: implement your complete MedReason system.
        # The placeholder below keeps the template runnable, but it is not a
        # meaningful baseline.
        if case.task_type == "mcq":
            answer = case.options[0].label
            reasoning_trace = "Placeholder custom system selected the first option."
        else:
            answer = "Unable to determine with the placeholder custom system."
            reasoning_trace = (
                "Placeholder custom system has not been implemented and did not inspect the image evidence."
            )

        return MedReasonPrediction(
            case_id=case.case_id,
            task_type=case.task_type,
            answer=answer,
            reasoning_trace=reasoning_trace,
            confidence=None,
            metadata={"system": "custom_placeholder"},
        )
