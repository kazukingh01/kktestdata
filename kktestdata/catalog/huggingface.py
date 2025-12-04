from dataclasses import dataclass


@dataclass(frozen=True)
class HFSpec:
    name: str
    repo_id: str
    split: str
    features: list[str]
    target: str | list[str]
    task: str
    n_data: int | None
    description: str
    n_classes: int = 0
    group: str | None = None
    load_kwargs: dict | None = None


HF_SPECS: tuple[HFSpec, ...] = (
    HFSpec(
        name="boatrace_2020_2021",
        repo_id="kazukingh01/boatrace_2020_2021",
        split="train",
        features=[
            "race_id",
            "number",
            "player_no",
            "motor_no",
            "boat_no",
            "weight",
            "adjust_weight",
            "propeller",
            "mount_angle",
            "exhibition_course",
            "exhibition_start_time",
            "exhibition_time",
        ],
        target="rank_web",
        group="race_id",
        task="rank",
        n_data=667152,
        n_classes=6,
        description="Boat race results for 2020-2021 published on Hugging Face (kazukingh01/boatrace_2020_2021).",
    ),
)

SPEC_BY_NAME: dict[str, HFSpec] = {spec.name: spec for spec in HF_SPECS}
