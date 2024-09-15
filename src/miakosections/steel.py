"""Module for steel material."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

STEEL_CLASSES = {
    "B500A": {
        "f_yk": 500.0,
        "k": 1.05,
        "epsilon_uk": 0.025,
    },
    "B500B": {
        "f_yk": 500.0,
        "k": 1.08,
        "epsilon_uk": 0.050,
    },
    "B500C": {
        "f_yk": 500.0,
        "k": 1.15,
        "epsilon_uk": 0.075,
    },
}


class Steel:
    """
    Steel material properties.

    :ivar steel_grade: Steel grade according to EN 1992-1-1:2004.
    :ivar f_yk: Characteristic yield strength of steel [MPa].
    :ivar k: Coefficient for the design yield strength of steel.
    :ivar epsilon_uk: Ultimate strain of steel.
    :ivar gamma_s: Partial safety factor for steel.
    :ivar e: Young's modulus of steel [MPa].
    """
    def __init__(
        self,
        steel_grade: str | None = None,
        f_yk: float | None = None,
        k: float = 1.08,
        epsilon_uk: float = 0.050,
        gamma_s: float = 1.15,
    ) -> None:
        """Initialise steel material properties."""
        self.steel_grade = steel_grade

        if steel_grade is not None:
            if steel_grade not in STEEL_CLASSES.keys():
                raise ValueError(f"Steel grade '{steel_grade}' not recognised. "
                                 f"Choose from: {[c for c in STEEL_CLASSES.keys()]}.")
            self.f_yk = STEEL_CLASSES.get(steel_grade).get("f_yk")
            self.k = STEEL_CLASSES.get(steel_grade).get("k")
            self.epsilon_uk = STEEL_CLASSES.get(steel_grade).get("epsilon_uk")
        else:
            if f_yk is None:
                raise ValueError("Steel grade or yield strength must be provided.")
            if f_yk <= 0:
                raise ValueError("Yield strength of steel must be positive.")
            self.f_yk = f_yk
            self.k = k
            self.epsilon_uk = epsilon_uk

        self.e: float = 200_000.0
        self.gamma_s = gamma_s

    @property
    def f_yd(self) -> float:
        """Design yield strength of steel [MPa]."""
        return self.f_yk / self.gamma_s

    @property
    def f_yd2(self) -> float:
        """Design strength of steel [MPa]."""
        eps_uk = self.epsilon_uk
        eps_ud = self.epsilon_ud
        f_yd = self.f_yd
        eps_sy = self.epsilon_syd
        k = self.k
        f_yd2 = (
            f_yd * (- k * eps_sy + eps_ud + eps_uk)
            / (eps_sy - eps_uk)
        )
        return f_yd2


    @property
    def f_yk2(self) -> float:
        """Characteristic strength of steel [MPa]."""
        return self.k * self.f_yk

    @property
    def epsilon_ud(self) -> float:
        """Design ultimate strain of steel."""
        return 0.9 * self.epsilon_uk

    @property
    def epsilon_syk(self) -> float:
        """Yield strain of steel."""
        return self.f_yk / self.e

    @property
    def epsilon_syd(self) -> float:
        return self.f_yd / self.e

