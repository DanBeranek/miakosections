from __future__ import annotations

from sectionproperties.pre.library import rectangular_section
from sectionproperties.pre.library import triangular_section
from sectionproperties.pre import Geometry
from sectionproperties.pre import CompoundGeometry
from concreteproperties.pre import add_bar
from concreteproperties.pre import add_bar_rectangular_array

import math

from typing import TYPE_CHECKING

from miakosections.resources import POT_BEAMS

if TYPE_CHECKING:
    from concreteproperties import Concrete
    from concreteproperties import SteelBar


class POTBeam:
    """
    A class to represent a POT beam section.

    :param label: The label of the POT beam section.
    :ivar length: The length of the POT beam section in mm.
    :ivar height: The height of the POT beam section in mm.
    :ivar width: The width of the POT beam section in mm.
    :ivar ceramic_cover_width: The ceramic cover width of the POT beam section in mm.
    :ivar d_top: The diameter of the top rebar of the POT beam section in mm.
    :ivar d_w: The diameter of the shear reinforcement of the POT beam section in mm.
    :ivar d_bottom_sides: The diameter of the bottom edge rebars of the POT beam section in mm.
    :ivar d_bottom_middle: The diameter of the bottom middle rebar of the POT beam section in mm.
    """
    def __init__(self,
                 label: str,
                 concrete_material: Concrete,
                 rebar_material: SteelBar,
                 consider_upper_rebar: bool = False,
                 ) -> None:
        """
        Initialize the POTBeam class.

        :param label: The label of the POT beam section.
        :param concrete_material: The concrete material of the POT beam section.
        :param rebar_material: The rebar material of the POT beam section.
        :param consider_upper_rebar: A boolean to consider the upper rebar of the POT beam section, default False.
        """
        if label not in POT_BEAMS.keys():
            raise ValueError(f"Invalid POT beam label. Choose from {list(POT_BEAMS.keys())}.")

        self.label = label
        self.length = POT_BEAMS[label]["length"]
        self.height = POT_BEAMS[label]["height"]
        self.width = POT_BEAMS[label]["width"]
        self.ceramic_cover_width = POT_BEAMS[label]["ceramic_cover_width"]
        self.consider_upper_rebar = consider_upper_rebar
        self.d_top = POT_BEAMS[label]["d_top"]
        self.height_w_rebar = POT_BEAMS[label]["height_with_rebar"]
        self.d_w = POT_BEAMS[label]["d_w"]
        self.d_bottom_sides = POT_BEAMS[label]["d_bottom_sides"]
        self.d_bottom_middle = POT_BEAMS[label]["d_bottom_middle"]

        self.concrete_material = concrete_material
        self.rebar_material = rebar_material

        self.concrete_cover = 15.0

    @property
    def geometry(self) -> CompoundGeometry:
        """Return the geometry of the POT beam section."""
        d = self.height - self.ceramic_cover_width
        b = self.width - 2 * self.ceramic_cover_width
        geom = rectangular_section(d=d, b=b, material=self.concrete_material)

        if self.consider_upper_rebar:
            y = self.height_w_rebar - self.ceramic_cover_width
            geom = add_bar(
                geometry=geom,
                area=math.pi*self.d_top**2/4,
                material=self.rebar_material,
                x=b/2,
                y=y
            )

        if self.d_bottom_middle:
            y = self.concrete_cover + self.d_bottom_middle/2
            geom = add_bar(
                geometry=geom,
                area=math.pi*self.d_bottom_middle**2/4,
                material=self.rebar_material,
                x=b/2,
                y=y
            )

        x_s = b - 2*self.concrete_cover - self.d_bottom_sides
        bottom_left_anchor = (
            self.concrete_cover + self.d_bottom_sides/2,
            self.concrete_cover + self.d_bottom_sides/2
        )
        geom = add_bar_rectangular_array(
            geometry=geom,
            area=math.pi*self.d_bottom_sides**2/4,
            material=self.rebar_material,
            n_x=2,
            x_s=x_s,
            n_y=1,
            y_s=420.0,
            anchor=bottom_left_anchor
        )

        return geom


class MiakoBlock:
    def __init__(
        self,
        height: float,
        pot_distance: float
    ) -> None:
        """
        Initialize the MiakoBlock class.

        :param height: The height of the block in mm.
        :param pot_distance: Axial distance of single POT beams in mm.
        """
        if height not in [80.0, 150.0, 190.0, 230.0, 250.0]:
            raise ValueError(f"Invalid block height. Choose from [80.0, 150.0, 190.0, 230.0, 250.0].")

        if pot_distance not in [500.0, 625.0]:
            raise ValueError(f"Invalid block width. Choose from [500.0, 625.0].")

        self.height = height
        self.width = pot_distance - 100.0

    @property
    def geometry(self) -> Geometry:
        """Return the right half geometry of the Miako block."""
        if math.isclose(self.height, 80.0):
            return self.geometry_additional_block()
        if math.isclose(self.height, 250.0):
            return self.geometry_full_height_block()
        return self.geometry_basic_block()

    def geometry_additional_block(self) -> Geometry:
        """Return right half geometry of the Miako block with 80 mm height."""
        rec = rectangular_section(d=self.height, b=self.width/2, material=None)
        pot_mask = (rectangular_section(d=60.0, b=30.0, material=None)
                    .align_to(other=rec, on="bottom", inner=True)
                    .align_to(other=rec, on="left", inner=True))

        return rec - pot_mask

    def geometry_basic_block(self) -> Geometry:
        """Return right half geometry of the Miako block with 150, 190, or 230 mm height."""
        rec = rectangular_section(d=self.height, b=self.width / 2, material=None)

        pot_mask = (rectangular_section(d=60.0, b=35.0, material=None)
                    .align_to(other=rec, on="bottom", inner=True)
                    .align_to(other=rec, on="left", inner=True))

        nose_mask = (triangular_section(b=15.0, h=-40.0, material=None)
                     .align_to(other=rec, on="top", inner=True)
                     .align_to(other=rec, on="left", inner=True)
                     .shift_section(0.0, -12.0))

        return rec - pot_mask - nose_mask

    def geometry_full_height_block(self) -> Geometry:
        """Return right half geometry of the Miako block with 250 mm height (BNK)."""
        rec = rectangular_section(d=self.height, b=self.width / 2, material=None)

        pot_mask = (rectangular_section(d=60.0, b=35.0, material=None)
                    .align_to(other=rec, on="bottom", inner=True)
                    .align_to(other=rec, on="left", inner=True))

        rec1 = (rectangular_section(d=16.0, b=84.0, material=None)
                .align_to(other=rec, on="top", inner=True)
                .align_to(other=rec, on="left", inner=True))

        rec2 = (rectangular_section(d=26, b=99, material=None)
                .align_to(other=rec1, on="bottom", inner=False)
                .align_to(other=rec, on="left", inner=True))

        tri = (triangular_section(b=99, h=-33, material=None)
               .align_to(other=rec2, on="bottom", inner=False).align_to(other=rec, on="left", inner=True))

        nose_mask = rec1 + rec2 + tri

        return rec - pot_mask - nose_mask


def get_bars_positions(
    x_wwf_rebars: list[float],
    wwf_diameter: float,
    tr_number: int,
    tr_diameter: float,
    tr_interval: list[float],
) -> list[float]:
    """
    Place rebars in the section.

    :param x_wwf_rebars: The x-coordinates of the WWF rebars.
    :param wwf_diameter: The diameter of the WWF rebars.
    :param tr_number: The number of the top rebars.
    :param tr_diameter: The diameter of the top rebars.
    :param tr_interval: The interval of the top rebars.
    :return: The x-coordinates of the top rebars.
    """
    interval_width = tr_interval[1] - tr_interval[0]

    # minimal distance between rebars
    min_distance = wwf_diameter / 2 + tr_diameter / 2 + 20.0

    # Add buffer zone around each WWF rebar within the top rebar interval
    buffer_zones = [(x - min_distance, x + min_distance) for x in x_wwf_rebars]

    # filter buffer zones intervals
    filtered_buffer_zones = []
    for i, (xi, xj) in enumerate(buffer_zones):
        if xi < tr_interval[0] < xj:
            filtered_buffer_zones.append((tr_interval[0], xj))
        elif xi < tr_interval[1] < xj:
            filtered_buffer_zones.append((xi, tr_interval[1]))
        elif tr_interval[0] < xi < xj < tr_interval[1]:
            filtered_buffer_zones.append((xi, xj))

    # get intervals for top rebars
    intervals = []
    for start, end in zip(filtered_buffer_zones[:-1], filtered_buffer_zones[1:]):
        intervals.append((start[1], end[0]))

    # sort intervals from the middle
    intervals = sort_from_middle(intervals)

    # Get the number of rebars that can be placed in each interval
    interval_widths = [end - start for start, end in intervals]
    max_n_rebars = [math.floor(width / min_distance)+1 for width in interval_widths]

    if sum(max_n_rebars) < tr_number:
        raise ValueError("Not enough space to place all top rebars.")

    n_rebars_per_interval = [0 for _ in range(len(intervals))]
    remaining_rebars = tr_number

    while remaining_rebars:
        for i in range(len(intervals)):
            if n_rebars_per_interval[i] < max_n_rebars[i]:
                n_rebars_per_interval[i] += 1
                remaining_rebars -= 1
                if remaining_rebars == 0:
                    break

    x_positions = []
    for n, (start, end), width in zip(n_rebars_per_interval, intervals, interval_widths):
        if n == 1:
            x_positions.append((start + end) / 2)
        else:
            step = width / (n - 1)
            for i in range(n):
                x_positions.append(start + i * step)

    return x_positions


def sort_from_middle(lst):
    """
    Sorts a list starting from the middle and then alternating between the left and right.

    :param lst: List of elements to be sorted.
    :return: List sorted from the middle outward.
    """
    n = len(lst)
    mid = n // 2  # Integer division to find the middle index

    # Using list slicing and stepping to create left and right parts
    left_part = lst[:mid][::-1]  # Reverse the left half
    right_part = lst[mid:] if n % 2 == 0 else lst[mid + 1:]  # Exclude middle if odd

    # Initialize result list
    result = []

    # Interleave elements from the left and right parts
    if n % 2 != 0:
        result.append(lst[mid])  # Start with the middle element if odd length

    # Add elements alternating from left and right
    for l, r in zip(left_part, right_part):
        result.append(l)
        result.append(r)

    # Add any leftover elements if the lists were uneven
    if len(left_part) > len(right_part):
        result.extend(left_part[len(right_part):])
    elif len(right_part) > len(left_part):
        result.extend(right_part[len(left_part):])

    return result
