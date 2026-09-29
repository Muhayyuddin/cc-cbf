"""Controller registry in the order used throughout the paper."""

from typing import Dict, Type

from algorithms.base_controller import BaseController
from algorithms.c3bf import C3BFController
from algorithms.cc_cbf import CCCBFController
from algorithms.geometric_cri import GeometricCRIController
from algorithms.rule_based_colreg import RuleBasedCOLREGController
from algorithms.turning_circle_cbf import TurningCircleCBFController

CONTROLLERS: Dict[str, Type[BaseController]] = {
    "CC-CBF": CCCBFController,               # proposed
    "C3BF": C3BFController,                  # B1
    "Rule-COLREG": RuleBasedCOLREGController,  # B2
    "Geo-CRI": GeometricCRIController,       # B3
    "TC-CBF": TurningCircleCBFController,    # B4
}


def make_controller(name: str) -> BaseController:
    """Instantiate a controller by its registry name."""
    try:
        return CONTROLLERS[name]()
    except KeyError:
        raise ValueError(f"Unknown controller {name!r}; available: {list(CONTROLLERS)}") from None
