from __future__ import annotations

import math

import numpy as np

from sectionproperties.pre.library import rectangular_section
from concreteproperties.pre import add_bar_rectangular_array, add_bar
from concreteproperties import ConcreteSection

from concreteproperties import Concrete
from concreteproperties import RectangularStressBlock
from concreteproperties import EurocodeNonLinear
from concreteproperties import SteelElasticPlastic

from concreteproperties import SteelBar

from miakosections.helpers import POTBeam
from miakosections.helpers import MiakoBlock
from miakosections.helpers import get_bars_positions


PRECAST_CONCRETE = Concrete(
    name="C25/30",
    density=2400.0,
    stress_strain_profile=EurocodeNonLinear(
        elastic_modulus=35e3,
        ultimate_strain=0.0035,
        compressive_strength=25.0,
        compressive_strain=0.0023,
        tensile_strength=3.5,
        tension_softening_stiffness=7e3,
    ),
    ultimate_stress_strain_profile=RectangularStressBlock(
        compressive_strength=25.0,
        alpha=1.0,
        gamma=0.8,
        ultimate_strain=0.0035,
    ),
    flexural_tensile_strength=3.4,
    colour='darkgrey',
)

INSITU_CONCRETE = Concrete(
    name="C25/30",
    density=2400.0,
    stress_strain_profile=EurocodeNonLinear(
        elastic_modulus=35e3,
        ultimate_strain=0.0035,
        compressive_strength=25.0,
        compressive_strain=0.0023,
        tensile_strength=3.5,
        tension_softening_stiffness=7e3,
    ),
    ultimate_stress_strain_profile=RectangularStressBlock(
        compressive_strength=25.0,
        alpha=1.0,
        gamma=0.8,
        ultimate_strain=0.0035,
    ),
    flexural_tensile_strength=3.4,
    colour='lightgrey',
)

STEEL = SteelBar(
    name="BSt 500 M",
    density=7850.0,
    stress_strain_profile=SteelElasticPlastic(
        yield_strength=500.0,
        elastic_modulus=200.e3,
        fracture_strain=0.1,
    ),
    colour="blue",
)


def create_miako_section(
    pot_label: str = "POT 400",
    n_pots: int = 1,
    miako_height: float = 190.0,
    pot_distances: tuple[float, float] = (625.0, 625.0),
    section_height: float = 250.0,
    precast_concrete: Concrete = PRECAST_CONCRETE,
    insitu_concrete: Concrete = INSITU_CONCRETE,
    precast_steel: SteelBar = STEEL,
    consider_top_pot_reinforcement: bool = True,
    consider_top_reinforcement: bool = True,
    concrete_cover: float = 20.0,
    wwf_steel: SteelBar | None = None,
    wwf_diameter: float = 6.0,
    wwf_spacing: float = 100.0,
    top_reinforcement_steel: SteelBar | None = None,
    top_reinforcement_diameter: float = 10.0,
    top_reinforcement_number: int = 0,
) -> ConcreteSection:
    """
    Create a Miako section with a POT beam and Miako blocks.

    :param pot_label: Label of the POT beam.
    :param n_pots: Number of POT beams.
    :param miako_height: Height of the Miako blocks in mm.
    :param pot_distances: Distances between simple POT beams in mm.
    :param section_height: Total height of the section in mm.
    :param precast_concrete: Concrete material for the POT beam.
                             Must be an instance of the :class:`concreteproperties.material.Concrete` class.
    :param insitu_concrete: Concrete material for the monolithic part.
                            Must be an instance of the :class:`concreteproperties.material.Concrete` class.
    :param precast_steel: Steel material for the POT beam.
                          Must be an instance of the :class:`concreteproperties.material.SteelBar` class.
    :param consider_top_pot_reinforcement: Flag to consider top reinforcement in the POT beam.
    :param consider_top_reinforcement: Flag to consider top reinforcement in the monolithic part.
    :param concrete_cover: Cover of the reinforcement in mm.
    :param wwf_steel: Steel material for the welded wire fabric reinforcement.
                      If `None`, no wwf reinforcement is added.
                      Else must be an instance of the :class:`concreteproperties.material.SteelBar` class.
    :param wwf_diameter: Diameter of the welded wire fabric reinforcement in mm.
    :param wwf_spacing: Spacing of the welded wire fabric reinforcement in mm.
    :param top_reinforcement_steel: Steel material for the top reinforcement.
                                    If `None`, no top reinforcement is added.
                                    Else must be an instance of
                                    the :class:`concreteproperties.material.SteelBar` class.
    :param top_reinforcement_diameter: Diameter of the top reinforcement in mm.
    :param top_reinforcement_number: Number of top reinforcement bars.

    :return: A :class:`concreteproperties.section.ConcreteSection` object.
    """
    b_eff = sum(pot_distances) / 2 + (n_pots - 1) * 160.0
    slab_height = section_height - miako_height

    rec = rectangular_section(d=section_height, b=b_eff, material=insitu_concrete)

    miako_left = (
        MiakoBlock(height=miako_height, pot_distance=pot_distances[0])
        .geometry
        .mirror_section(axis="y")
        .align_to(other=rec, on="bottom", inner=True)
        .align_to(other=rec, on="left", inner=True)
    )

    miako_right = (
        MiakoBlock(height=miako_height, pot_distance=pot_distances[1])
        .geometry
        .align_to(other=rec, on="bottom", inner=True)
        .align_to(other=rec, on="right", inner=True)
    )

    rec2 = rectangular_section(d=60, b=b_eff, material=None)

    if consider_top_reinforcement:
        if wwf_steel:
            top_reinforcement_diameter = top_reinforcement_diameter if top_reinforcement_steel else 0.0

            # round down number of wwf bars that fit in the section
            n_x = int(math.floor(b_eff / wwf_spacing))

            # calculate center position of the first wwf bar
            anchor = (
                (b_eff - (n_x - 1) * wwf_spacing) / 2,
                section_height - concrete_cover - top_reinforcement_diameter + wwf_diameter / 2
            )

            # add bars as rectangular array
            rec = add_bar_rectangular_array(
                geometry=rec,
                area=math.pi*wwf_diameter**2/4,
                material=wwf_steel,
                n_x=n_x,
                x_s=wwf_spacing,
                n_y=1,
                y_s=69.0,
                anchor=anchor
            )

        if top_reinforcement_steel:
            # we must prevent overlapping of the wwf and the top reinforcement
            # assuming, the top reinforcement can be placed in the rib width
            # extended 45 degrees to the sides at the slab

            if wwf_steel:
                # get the wwf x-positions:
                x_wwf = np.arange(anchor[0], anchor[0] + n_x * wwf_spacing, wwf_spacing).tolist()

                extension_45_deg = slab_height - concrete_cover - top_reinforcement_diameter
                # calculate acceptable interval for the top reinforcement
                tr_x_interval = [
                    pot_distances[0] / 2 - 50 - extension_45_deg,
                    b_eff - pot_distances[1] / 2 + 50 + extension_45_deg
                ]

                x_pos = get_bars_positions(
                    x_wwf_rebars=x_wwf,
                    wwf_diameter=wwf_diameter,
                    tr_number=top_reinforcement_number,
                    tr_diameter=top_reinforcement_diameter,
                    tr_interval=tr_x_interval
                )

            else:
                x_pos = np.linspace(
                    start=pot_distances[0] / 2 - concrete_cover,
                    stop=pot_distances[0] / 2 + (n_pots - 1) * 160 + concrete_cover,
                    num=top_reinforcement_number
                )

            for x in x_pos:
                rec = add_bar(
                    geometry=rec,
                    area=math.pi*top_reinforcement_diameter**2/4,
                    material=top_reinforcement_steel,
                    x=x,
                    y=section_height - concrete_cover - top_reinforcement_diameter / 2
                )

    geom = rec - miako_left - miako_right - rec2

    pot_section = (POTBeam(
        label=pot_label,
        concrete_material=precast_concrete,
        rebar_material=precast_steel,
        consider_upper_rebar=consider_top_pot_reinforcement
    ).geometry
                   .align_to(other=geom, on="bottom", inner=True)
                   .shift_section(y_offset=-45))

    for n in range(n_pots):
        shift_x = pot_distances[0] / 2 - 130 / 2 + n * 160.0
        shifted_pot = pot_section.shift_section(x_offset=shift_x)
        geom = (geom - shifted_pot) + shifted_pot  # Subtract and add the POT beam section

    return ConcreteSection(geometry=geom)
