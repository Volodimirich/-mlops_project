from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureParams:
    numerical: Optional[list]
    str_features: Optional[list]
