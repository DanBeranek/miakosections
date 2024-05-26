from __future__ import annotations

from typing import TYPE_CHECKING

from enum import StrEnum

import numpy as np
import matplotlib.pyplot as plt

from miakosections.utils import plotting_context

if TYPE_CHECKING:
    import numpy.typing as npt
    from miakosections.concrete import Concrete
    from miakosections.steel import Steel
    import matplotlib.axes


class ConcreteCompressionSSP(StrEnum):
    """
    Enum for the compression stress-strain profile types.

    * ``ELASTIC``: Elastic concrete stress-strain profile without limit.
    * ``STRESS_BLOCK``: Rectangular stress block according to EN 1992-1-1:2004, 3.1.7(3).
    * ``EC_BILINEAR``: Eurocode bi-linear concrete ssp according to EN 1992-1-1:2004, 3.1.7(2).
    * ``EC_PARABOLIC_RECTANGULAR``: Eurocode parabolic-rectangular concrete ssp according to EN 1992-1-1:2004, 3.1.7(1).
    * ``EC_NONLINEAR``: Eurocode non-linear concrete ssp according to EN 1992-1-1:2004, 3.1.5(1).
    """
    ELASTIC = "elastic"
    STRESS_BLOCK = 'block'
    BILINEAR = "bilinear"
    PARABOLIC_RECTANGULAR = "parabolic"
    NONLINEAR = "nonlinear"


class ConcreteTensionSSP(StrEnum):
    """
    Enum for the tension stress-strain profile types.

    * ``NONE``: No tension stress-strain profile.
    * ``ELASTIC``: Elastic concrete stress-strain profile without limit.
    * ``ELASTIC_BRITTLE``: Elastic concrete stress-strain profile with brittle behaviour.
    * ``ELASTIC_PLASTIC_WITH_SOFTENING``: Elastic concrete stress-strain profile with softening.
    """
    NONE = "none"
    ELASTIC = "elastic"
    ELASTIC_BRITTLE = "brittle"
    ELASTIC_PLASTIC_WITH_SOFTENING = "softening"


class ReinforcementSSP(StrEnum):
    """
    Enum for the reinforcement stress-strain profile types.

    * ``ELASTIC``: Elastic stress-strain profile without limit.
    * ``ELASTIC_PLASTIC``: Elastic-plastic stress-strain profile without limit.
    * ``ELASTIC_PLASTIC_WITH_HARDENING``: Elastic-plastic stress-strain profile with hardening.
    """
    ELASTIC = "elastic"
    ELASTIC_PLASTIC = "elastic_plastic"
    ELASTIC_PLASTIC_WITH_HARDENING = "hardening"


class StressStrainProfile:
    """
    Base class for a material stress-strain profile.

    Negative stresses and strains are compression.

    :param stresses: Stresses in the material (MPa).
    :param strains: Strains in the material (1).
    """
    def __init__(
        self,
        strains: list[float] | npt.NDArray[np.float64],
        stresses: list[float] | npt.NDArray[np.float64],
    ) -> None:
        self.strains = np.array(strains)
        self.stresses = np.array(stresses)

    def _validate_input(self) -> None:
        """
        Validate the input stresses and strains.

        * Stresses and strains must have the same length.
        * Stresses and strains must have at least two values.
        * Strain values must be increasing.

        :raises ValueError: If the input stresses and strains are invalid.
        """
        if len(self.stresses) != len(self.strains):
            raise ValueError("Stresses and strains must have the same length.")

        if len(self.strains < 2):
            raise ValueError("Stresses and strains must have at least two values.")

        # Check if strains are non-decreasing
        if not np.all(np.diff(self.strains) > 0):
            raise ValueError("Strain values mut be increasing.")

    @property
    def min_strain(self) -> float:
        """Return the minimum (maximum compression) strain in the material."""
        return float(np.min(self.strains))

    @property
    def max_strain(self) -> float:
        """Return the maximum (maximum tension) strain in the material."""
        return float(np.max(self.strains))

    def get_stresses(
        self,
        strains: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Return the stresses in the material for given strains.

        If the strain is outside the range of the interval, the stress is set to zero.

        :param strains: Strains in the material.
        :return: Stresses in the material.
        """
        return np.interp(
            x=strains,
            xp=self.strains,
            fp=self.stresses,
            left=0.0,
            right=0.0
        )

    def plot_profile(
        self,
        title: str = "Stress-Strain Profile",
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the stress-strain profile.

                Args:
                    title: Plot title
                    fmt: Plot format string
                    kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

                Returns:
                    Matplotlib axes object
                """
        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (fig, ax):
            assert ax
            ax.plot(self.strains, self.stresses, fmt)
            plt.xlabel("Strain")
            plt.ylabel("Stress")
            plt.grid(True)

        return ax


class ConcreteStressStrainProfile(StressStrainProfile):
    """
    Base class for a concrete stress-strain profile.

    :param concrete: Instance of the :class:`Concrete` class.
    :param limit_state: Limit state for the concrete stress-strain profile.
    :param compression_profile: Type of the compression stress-strain profile.
    :param tension_profile: Type of the tension stress-strain profile.
    :param tension_softening_stiffness: Slope of the linear tension softening branch (MPa).
    :param n_points_1: Number of points to discretize the curve prior to the peak stress
    :param n_points_2: Number of points to discretize the curve after the peak stress
    """
    def __init__(
        self,
        concrete: Concrete,
        limit_state: str,
        compression_profile: ConcreteCompressionSSP = ConcreteCompressionSSP.NONLINEAR,
        tension_profile: ConcreteTensionSSP = ConcreteTensionSSP.ELASTIC_PLASTIC_WITH_SOFTENING,
        tension_softening_stiffness: float = 10.0e3,
        n_points_1: int = 10,
        n_points_2: int = 5,
    ) -> None:
        self.concrete = concrete

        if limit_state.upper() not in ["ULS", "SLS"]:
            raise ValueError(f"Limit state must be either 'ULS' or 'SLS', not '{limit_state}'.")

        if limit_state.upper() == "ULS":
            if compression_profile == ConcreteCompressionSSP.NONLINEAR:
                raise ValueError("Non-linear compression profile is not allowed for ULS.")
            if tension_profile != ConcreteTensionSSP.NONE:
                raise ValueError("Tension profile is not allowed for ULS.")

        if limit_state.upper() == "SLS":
            if compression_profile == ConcreteCompressionSSP.ELASTIC:
                raise ValueError("Elastic compression profile is not allowed for SLS.")
            if compression_profile == ConcreteCompressionSSP.STRESS_BLOCK:
                raise ValueError("Stress block compression profile is not allowed for SLS.")

        self.limit_state = limit_state
        self.compression_profile = compression_profile
        self.tension_profile = tension_profile
        self.tension_softening_stiffness = tension_softening_stiffness
        self._n_points_1 = n_points_1
        self._n_points_2 = n_points_2

        tension_strain, tension_stress = self._get_tension_strain_stress_values()
        compression_strain, compression_stress = self._get_compression_strain_stress_values()

        super().__init__(
            strains=compression_strain+tension_strain,
            stresses=compression_stress+tension_stress,
        )

    def _get_tension_strain_stress_values(self) -> tuple[list[float], list[float]]:
        """
        Return the tension strain-stress profile for the concrete.

        :return: Stresses and strains in the tension stress-strain profile.
        """
        match self.tension_profile:
            case ConcreteTensionSSP.NONE:
                return [0.0], [0.0]

            case ConcreteTensionSSP.ELASTIC:
                return [0.0, 1.0], [0.0, self.concrete.e_cm]

            case ConcreteTensionSSP.ELASTIC_BRITTLE:
                return [0.0, self._get_cracking_strain_in_tension()], [0.0, self.concrete.f_ctm]

            case ConcreteTensionSSP.ELASTIC_PLASTIC_WITH_SOFTENING:
                if self.tension_softening_stiffness is None:
                    raise ValueError("Concrete tension stress-strain profile requires tension softening stiffness.")
                strain_crack = self._get_cracking_strain_in_tension()
                strain_zero_tension = strain_crack + self.concrete.f_ctm / self.tension_softening_stiffness
                return [0.0, strain_crack, strain_zero_tension], [0.0, self.concrete.f_ctm, 0.0]

            case _:
                raise ValueError(f"Invalid tension stress-strain profile: '{self.tension_profile}'")

    def _get_cracking_strain_in_tension(self) -> float:
        """
        Return the cracking strain in tension for the concrete.

        :return: Cracking strain in tension.
        """
        f_c = self.concrete.f_ck
        f_t = self.concrete.f_ctm

        match self.compression_profile:
            case ConcreteCompressionSSP.ELASTIC:
                raise ValueError('Cracking strain in tension is not defined for elastic compression profile.')

            case ConcreteCompressionSSP.STRESS_BLOCK:
                raise ValueError('Cracking strain in tension is not defined for stress block compression profile.')

            case ConcreteCompressionSSP.BILINEAR:
                eps_c = self.concrete.eps_c3
                return f_t * (eps_c / f_c)

            case ConcreteCompressionSSP.PARABOLIC_RECTANGULAR:
                eps_c = self.concrete.eps_c2
                return f_t * (eps_c / (f_c * self.concrete.n))

            case ConcreteCompressionSSP.NONLINEAR:
                return f_t / (1.05 * self.concrete.e_cm)

            case _:
                raise ValueError(f"Invalid compression stress-strain profile: '{self.compression_profile}'")

    def _get_compression_strain_stress_values(self) -> tuple[list[float], list[float]]:
        if self.limit_state == "ULS":
            f_c = self.concrete.f_cd
        else:
            if self.compression_profile == ConcreteCompressionSSP.NONLINEAR:
                f_c = self.concrete.f_cm
            else:
                f_c = self.concrete.f_ck

        match self.compression_profile:
            case ConcreteCompressionSSP.ELASTIC:
                return [-1.0], [-self.concrete.e_cm]

            case ConcreteCompressionSSP.STRESS_BLOCK:
                eps_cu3 = self.concrete.eps_cu3
                lamb = self.concrete.lamb
                return [
                    - eps_cu3,
                    - eps_cu3 * (1 - lamb),
                    - eps_cu3 * (1 - lamb),
                    ], [
                    - f_c,
                    - f_c,
                    0.0,
                ]

            case ConcreteCompressionSSP.BILINEAR:
                return [
                    - self.concrete.eps_cu3,
                    - self.concrete.eps_c3,
                ], [
                    - f_c,
                    - f_c,
                ]

            case ConcreteCompressionSSP.PARABOLIC_RECTANGULAR:
                strains = np.linspace(-self.concrete.eps_c2, 0.0, self._n_points_1)[:-1]
                stresses = - (f_c *
                              (1 - (1 - np.abs(strains) / self.concrete.eps_c2) ** self.concrete.n)
                              )

                strains = strains.tolist()
                strains.insert(0, - self.concrete.eps_cu2)
                stresses = stresses.tolist()
                stresses.insert(0, - f_c)

                return strains, stresses

            case ConcreteCompressionSSP.NONLINEAR:
                strains_2 = np.linspace(-self.concrete.eps_cu1, -self.concrete.eps_c1, self._n_points_2)
                strains_1 = np.linspace(-self.concrete.eps_c1, 0.0, self._n_points_1)[1:-1]

                strains = np.concatenate([strains_2, strains_1])

                k = 1.05 * self.concrete.e_cm * np.abs(self.concrete.eps_c1) / self.concrete.f_cm
                eta = np.abs(strains / self.concrete.eps_c1)

                stresses = - (k * eta - eta ** 2) / (1 + (k - 2) * eta) * self.concrete.f_cm

                return strains.tolist(), stresses.tolist()

            case _:
                raise ValueError(f"Invalid compression stress-strain profile: '{self.compression_profile}'")


class ReinforcementStressStrainProfile(StressStrainProfile):
    """
    Base class for a reinforcement stress-strain profile.

    :param steel: Instance of the :class:`Steel` class.
    :param limit_state: Limit state for the reinforcement stress-strain profile.
    :param profile: Type of the reinforcement stress-strain profile.
    """
    def __init__(
        self,
        steel: Steel,
        limit_state: str,
        profile: ReinforcementSSP = ReinforcementSSP.ELASTIC_PLASTIC,
    ) -> None:
        self.steel = steel
        self.profile = profile

        if limit_state.upper() not in ["ULS", "SLS"]:
            raise ValueError(f"Limit state must be either 'ULS' or 'SLS', not '{limit_state}'.")

        self.limit_state = limit_state

        strains, stresses = self._get_strain_stress_values()

        super().__init__(
            strains=strains,
            stresses=stresses,
        )

    def _get_strain_stress_values(self) -> tuple[list[float], list[float]]:
        if self.limit_state.upper() == "ULS":
            yield_strain = self.steel.epsilon_syd
            yield_stress = self.steel.f_yd
            strength_strain = self.steel.epsilon_ud
            strength_stress = self.steel.f_yd2
        else:
            yield_strain = self.steel.epsilon_syk
            yield_stress = self.steel.f_yk
            strength_strain = self.steel.epsilon_uk
            strength_stress = self.steel.f_yk2

        match self.profile:
            case ReinforcementSSP.ELASTIC:
                return [-1.0, 1.0], [-self.steel.e, self.steel.e]

            case ReinforcementSSP.ELASTIC_PLASTIC:
                return [
                    - 100.0,
                    - yield_strain,
                    + yield_strain,
                    + 100.0
                ], [
                    - yield_stress,
                    - yield_stress,
                    + yield_stress,
                    + yield_stress,
                ]

            case ReinforcementSSP.ELASTIC_PLASTIC_WITH_HARDENING:
                return [
                    - strength_strain,
                    - yield_strain,
                    + yield_strain,
                    + strength_strain
                ], [
                    - strength_stress,
                    - yield_stress,
                    + yield_stress,
                    + strength_stress,
                ]

            case _:
                raise ValueError(f"Invalid reinforcement stress-strain profile: '{self.profile}'")


def main() -> None:
    from miakosections.concrete import Concrete
    from miakosections.steel import Steel
    conc = Concrete("C20/25")
    steel = Steel("B500B")
    # ssp = ConcreteStressStrainProfile(concrete=conc)
    # ssp.plot_profile()
    #
    # for c_profile in ConcreteCompressionSSP:
    #     for t_profile in ConcreteTensionSSP:
    #         try:
    #             ssp = ConcreteStressStrainProfile(concrete=conc, compression_profile=c_profile, tension_profile=t_profile)
    #             ssp.plot_profile(title=f"{c_profile.value} - {t_profile.value}")
    #         except ValueError as e:
    #             pass

    for s_profile in ReinforcementSSP:
        ssp = ReinforcementStressStrainProfile(steel=steel, profile=s_profile)
        ssp.plot_profile(title=s_profile.value)


if __name__ == "__main__":
    main()
