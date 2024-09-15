from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import matplotlib.pyplot as plt

from typing import TYPE_CHECKING

from miakosections.utils import plotting_context


if TYPE_CHECKING:
    import numpy.typing as npt
    import matplotlib.axes

    from miakosections.miako_section import ConcreteMesh
    from miakosections.miako_section import SteelMesh


@dataclass
class MomentCurvatureResults:
    theta: float
    n_target: float
    curvature: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    moments: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    axial_force_error: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    _failure_mesh: list[ConcreteMesh | SteelMesh] = field(init=False, repr=False)
    _failure: bool = field(default=False, init=False)

    def plot_results(
        self,
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots the moment curvature results.

        Args:
            m_scale: Scaling factor to apply to bending moment
            fmt: Plot format string
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """

        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (fig, ax):
            assert ax
            ax.plot(self.curvature, self.moments * m_scale, fmt)
            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

        return ax

    @staticmethod
    def plot_multiple_results(
        moment_curvature_results: list[MomentCurvatureResults],
        labels: list[str],
        m_scale: float = 1e-6,
        fmt: str = "o-",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plots multiple moment curvature results.

        Args:
            moment_curvature_results: List of moment curvature results objects
            labels: List of labels for each moment curvature diagram
            m_scale: Scaling factor to apply to bending moment
            fmt: Plot format string
            kwargs: Passed to :func:`~concreteproperties.post.plotting_context`

        Returns:
            Matplotlib axes object
        """
        # create plot and setup the plot
        with plotting_context(title="Moment-Curvature", **kwargs) as (fig, ax):
            assert ax
            idx = 0

            # for each M-k curve
            for idx, mk_result in enumerate(moment_curvature_results):
                # scale results
                kappas = np.array(mk_result.curvature)
                moments = np.array(mk_result.moments) * m_scale

                ax.plot(kappas, moments, fmt, label=labels[idx])

            plt.xlabel("Curvature")
            plt.ylabel("Moment")
            plt.grid(True)

            # if there is more than one curve show legend
            if idx > 0:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

