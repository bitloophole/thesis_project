from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    name: str
    problem_type: str
    label_space: str
    expected_num_classes: int
    class_names: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


THESIS_MULTICLASS_TASK = TaskDefinition(
    name="thesis_multiclass_category_ids",
    problem_type="multiclass_classification",
    label_space="8 encoded classes already present in the prepared CICIoT2023 CSV files",
    expected_num_classes=8,
    class_names=(
        "label_0",
        "label_1",
        "label_2",
        "label_3",
        "label_4",
        "label_5",
        "label_6",
        "label_7",
    ),
    notes=(
        "This project is locked to multiclass IDS rather than binary benign-vs-attack classification.",
        "The proposal describes a category-level setting aligned with benign plus seven attack super-categories.",
        "Numeric label IDs are already encoded in the provided CSV files.",
        "Exact semantic mapping from numeric IDs to attack-category names should be verified against the upstream preprocessing source before thesis write-up tables use human-readable class names.",
    ),
)
