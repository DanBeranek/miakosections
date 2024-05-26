from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from typing import TYPE_CHECKING

from miakosections.concrete import Concrete
from miakosections.steel import Steel
from miakosections.stress_strain_profile import ConcreteStressStrainProfile
from miakosections.stress_strain_profile import ReinforcementStressStrainProfile
from miakosections.stress_strain_profile import ConcreteCompressionSSP
from miakosections.stress_strain_profile import ConcreteTensionSSP
from miakosections.stress_strain_profile import ReinforcementSSP
from miakosections.resources import POT_BEAMS
from miakosections.results import MomentCurvatureResults

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class MiakoSection:
    """Create a Miako section with a POT beam and Miako blocks.

    :param pot_label: Label of the POT beam.
    :param n_pots: Number of POT beams.
    :param miako_height: Height of the Miako blocks in mm.
    :param pot_distances: Distances between simple POT beams in mm.
    :param section_height: Total height of the section in mm.
    :param precast_concrete_strength_class: Strength class of the concrete of the POT beam.
    :param insitu_concrete_strength_class: Strength class of the concrete of the monolithic part.
    :param precast_steel_grade: Steel grade of reinforcement in the POT beam in MPa.
    :param concrete_cover: Cover of the reinforcement in mm.
    :param wwf_steel_grade: Steel grade of the welded wire fabric reinforcement.
    :param wwf_diameter: Diameter of the welded wire fabric reinforcement in mm.
    :param wwf_spacing: Spacing of the welded wire fabric reinforcement in mm.
    :param top_reinforcement_steel_grade: Steel grade of the top reinforcement.
    :param top_reinforcement_diameter: Diameter of the top reinforcement in mm.
    :param top_reinforcement_number: Number of top reinforcement bars.

    :return: A :class:`concreteproperties.section.ConcreteSection` object.
    """
    pot_label: str = "POT 400"
    n_pots: int = 1
    miako_height: float = 190.0
    pot_distances: tuple[float, float] = (625.0, 625.0)
    section_height: float = 250.0
    precast_concrete_strength_class: str = "C25/30"
    insitu_concrete_strength_class: str = "C20/25"
    precast_steel_grade: str = "B500B"
    concrete_cover: float = 20.0
    wwf_steel_grade: str = "B500A"
    wwf_diameter: float = 6.0
    wwf_spacing: float = 100.0
    top_reinforcement_steel_grade: str = "B500B"
    top_reinforcement_diameter: float = 10.0
    top_reinforcement_number: int = 0

    def __post_init__(self) -> None:
        """Initialise Miako section."""
        self.slab_height = self.section_height - self.miako_height

        self._validate_inputs()

        self.section_height -= 15.0  # account for ceramic cover
        self.ax_pot_dist = sum(self.pot_distances) / 2
        self.b_eff = self.ax_pot_dist + (self.n_pots - 1) * 160.0

        self._create_design_meshes()
        self._create_serviceability_meshes()

    def _validate_inputs(self):
        """Validate input parameters."""
        miako_heights = [80.0, 150.0, 190.0, 230.0, 250.0]
        miako_widths = [500.0, 625.0]

        if self.pot_label not in POT_BEAMS.keys():
            raise AttributeError(f"Invalid POT label. Choose from: {POT_BEAMS.keys()}.")

        if self.n_pots < 1:
            raise AttributeError("Number of POT beams must be at least 1.")

        if self.miako_height not in miako_heights:
            raise AttributeError(f"Invalid MIAKO height. Choose from: {miako_heights}.")

        if len(self.pot_distances) != 2:
            raise AttributeError("'pot_distances' must be a tuple of length 2.")

        if not all(w in miako_widths for w in self.pot_distances):
            raise AttributeError(f"Invalid  'pot_distances'. Choose from: {miako_widths}.")

        if self.concrete_cover < 0.0:
            raise AttributeError("Concrete cover cannot be negative.")

        if self.wwf_diameter < 0.0:
            raise AttributeError("WWF diameter cannot be negative.")

        if self.wwf_spacing < 0.0:
            raise AttributeError("WWF spacing cannot be negative.")

        if self.top_reinforcement_diameter < 0.0:
            raise AttributeError("Top reinforcement diameter cannot be negative.")

        if self.top_reinforcement_number < 0:
            raise AttributeError("Top reinforcement number cannot be negative.")

        if self.slab_height < 0.0:
            raise AttributeError("Slab height cannot be negative. Check 'miako_height' and 'section_height'.")

    def _create_serviceability_meshes(
        self,
        h_interval: float = 1.0,
        concrete_compression_stress_strain_profile: ConcreteCompressionSSP = ConcreteCompressionSSP.NONLINEAR,
        concrete_tension_stress_strain_profile: ConcreteTensionSSP = ConcreteTensionSSP.ELASTIC_PLASTIC_WITH_SOFTENING,
        reinforcement_stress_strain_profile: ReinforcementSSP = ReinforcementSSP.ELASTIC_PLASTIC_WITH_HARDENING,
        consider_top_pot_reinforcement: bool = False,
        consider_top_reinforcement: bool = True,
    ) -> None:
        self._serviceability_mesh: list[MiakoMesh] = [
            self._create_mesh(
                h_interval=h_interval,
                theta=0.0,
                concrete_compression_stress_strain_profile=concrete_compression_stress_strain_profile,
                concrete_tension_stress_strain_profile=concrete_tension_stress_strain_profile,
                reinforcement_stress_strain_profile=reinforcement_stress_strain_profile,
                consider_top_pot_reinforcement=consider_top_pot_reinforcement,
                consider_top_reinforcement=consider_top_reinforcement,
                limit_state="SLS",
            ),
            self._create_mesh(
                h_interval=h_interval,
                theta=np.pi,
                concrete_compression_stress_strain_profile=concrete_compression_stress_strain_profile,
                concrete_tension_stress_strain_profile=concrete_tension_stress_strain_profile,
                reinforcement_stress_strain_profile=reinforcement_stress_strain_profile,
                consider_top_pot_reinforcement=consider_top_pot_reinforcement,
                consider_top_reinforcement=consider_top_reinforcement,
                limit_state="SLS",
            )
        ]

    def _create_design_meshes(
        self,
        h_interval: float = 1.0,
        concrete_compression_stress_strain_profile: ConcreteCompressionSSP = ConcreteCompressionSSP.PARABOLIC_RECTANGULAR,
        concrete_tension_stress_strain_profile: ConcreteTensionSSP = ConcreteTensionSSP.NONE,
        reinforcement_stress_strain_profile: ReinforcementSSP = ReinforcementSSP.ELASTIC_PLASTIC_WITH_HARDENING,
    ) -> None:
        self._serviceability_mesh: list[MiakoMesh] = [
            self._create_mesh(
                h_interval=h_interval,
                theta=0.0,
                concrete_compression_stress_strain_profile=concrete_compression_stress_strain_profile,
                concrete_tension_stress_strain_profile=concrete_tension_stress_strain_profile,
                reinforcement_stress_strain_profile=reinforcement_stress_strain_profile,
                consider_top_pot_reinforcement=False,
                consider_top_reinforcement=False,
                limit_state="ULS",
            ),
            self._create_mesh(
                h_interval=h_interval,
                theta=np.pi,
                concrete_compression_stress_strain_profile=concrete_compression_stress_strain_profile,
                concrete_tension_stress_strain_profile=concrete_tension_stress_strain_profile,
                reinforcement_stress_strain_profile=reinforcement_stress_strain_profile,
                consider_top_pot_reinforcement=False,
                consider_top_reinforcement=True,
                limit_state="ULS",
            )
        ]

    def _create_mesh(
        self,
        h_interval: float = 1.0,
        theta: float = 0.0,
        concrete_compression_stress_strain_profile: ConcreteCompressionSSP = ConcreteCompressionSSP.NONLINEAR,
        concrete_tension_stress_strain_profile: ConcreteTensionSSP = ConcreteTensionSSP.ELASTIC_PLASTIC_WITH_SOFTENING,
        reinforcement_stress_strain_profile: ReinforcementSSP = ReinforcementSSP.ELASTIC_PLASTIC_WITH_HARDENING,
        consider_top_pot_reinforcement: bool = False,
        consider_top_reinforcement: bool = True,
        limit_state: str = "SLS",
    ) -> MiakoMesh:
        """
        Create mesh for the Miako section.

        :param h_interval: Height of intervals for the mesh in (mm).
        :param theta: Angle of the section. Choose from: [0.0, PI].
        :param concrete_compression_stress_strain_profile: Stress-strain profile for concrete in compression.
        :param concrete_tension_stress_strain_profile: Stress-strain profile for concrete in tension.
        :param reinforcement_stress_strain_profile: Stress-strain profile for reinforcement.
        :param consider_top_pot_reinforcement: Flag to consider top reinforcement in the POT beam.
        :param consider_top_reinforcement: Flag to consider top reinforcement in the monolithic part.
        """
        return MiakoMesh(
            miako_section=self,
            h_interval=h_interval,
            theta=theta,
            concrete_compression_stress_strain_profile=concrete_compression_stress_strain_profile,
            concrete_tension_stress_strain_profile=concrete_tension_stress_strain_profile,
            reinforcement_stress_strain_profile=reinforcement_stress_strain_profile,
            consider_top_pot_reinforcement=consider_top_pot_reinforcement,
            consider_top_reinforcement=consider_top_reinforcement,
            limit_state=limit_state,
        )


class MiakoMesh:
    def __init__(
        self,
        miako_section: MiakoSection,
        h_interval: float,
        concrete_compression_stress_strain_profile: ConcreteCompressionSSP,
        concrete_tension_stress_strain_profile: ConcreteTensionSSP,
        reinforcement_stress_strain_profile: ReinforcementSSP,
        theta: float,
        consider_top_pot_reinforcement: bool,
        consider_top_reinforcement: bool,
        limit_state: str
    ) -> None:
        """Initialise mesh for the Miako section."""
        if not (np.isclose(theta, 0.0) or np.isclose(theta, np.pi)):
            raise AttributeError("Invalid theta. Choose from: [0.0, PI].")
        self.miako_section = miako_section
        self.h_interval = h_interval
        self.theta = theta
        self.concrete_compressive_stress_strain_profile = concrete_compression_stress_strain_profile
        self.concrete_tension_stress_strain_profile = concrete_tension_stress_strain_profile
        self.reinforcement_stress_strain_profile = reinforcement_stress_strain_profile
        self.consider_top_pot_reinforcement = consider_top_pot_reinforcement
        self.consider_top_reinforcement = consider_top_reinforcement
        self.limit_state = limit_state

        self._create_mesh()
        self.c_y, self.c_z = self._centroid()

    def _centroid(self) -> tuple[float, float]:
        zae = 0.0
        ze = 0.0
        z = 0.0
        for mesh in self.meshes:
            if isinstance(mesh, ConcreteMesh):
                zae += np.sum(mesh.z_btm * mesh.area * mesh.concrete.e_cm)
                ze += np.sum(mesh.area * mesh.concrete.e_cm)
            elif isinstance(mesh, SteelMesh):
                zae += np.sum(mesh.z_btm * mesh.area * mesh.steel.e)
                ze += np.sum(mesh.area * mesh.steel.e)
                zae -= np.sum(mesh.z_btm * mesh.area * mesh.concrete.e_cm)
                ze -= np.sum(mesh.area * mesh.concrete.e_cm)
            else:
                raise AttributeError("Invalid mesh type.")

            z = zae / ze

        return self.miako_section.b_eff / 2, z

    def _create_mesh(self):
        """Create mesh for the Miako section."""
        self.meshes = [
            ConcreteMesh(miako_mesh=self, part="precast"),
            ConcreteMesh(miako_mesh=self, part="insitu"),
            SteelMesh(miako_mesh=self, part="precast_btm"),
        ]

        if self.consider_top_pot_reinforcement:
            mesh = SteelMesh(miako_mesh=self, part="precast_top")
            if mesh.created:
                self.meshes.append(mesh)

        if self.consider_top_reinforcement:
            top_mesh = SteelMesh(miako_mesh=self, part="top")
            if top_mesh.created:
                self.meshes.append(top_mesh)

            wwf_mesh = SteelMesh(miako_mesh=self, part="wwf")
            if wwf_mesh.created:
                self.meshes.append(wwf_mesh)

    def max_tension_force(self):
        """
        Calculate the maximum tension axial force that section can carry.
        """
        max_strain = 0.0
        for mesh in self.meshes:
            if isinstance(mesh, SteelMesh):
                max_strain = max(max_strain, mesh.ssp_steel.max_strain)

        n = 0.0
        for mesh in self.meshes:
            n += np.sum(
                mesh._normal_forces_from_strain(strains=np.full_like(mesh.z_btm, max_strain))
            )

        return n

    def min_compression_force(self):
        """
        Calculate the minimum compression axial force that section can carry.
        """
        min_strain = 0.0

        for mesh in self.meshes:
            if isinstance(mesh, ConcreteMesh):
                match mesh.ssp.compression_profile:
                    case ConcreteCompressionSSP.BILINEAR | ConcreteCompressionSSP.STRESS_BLOCK:
                        min_strain = min(min_strain, -mesh.concrete.eps_c3)
                    case ConcreteCompressionSSP.PARABOLIC_RECTANGULAR:
                        min_strain = min(min_strain, -mesh.concrete.eps_c2)
                    case ConcreteCompressionSSP.NONLINEAR:
                        min_strain = min(min_strain, -mesh.concrete.eps_c1)
                    case _:
                        raise AttributeError("Invalid concrete compression stress-strain profile.")
        n = 0.0
        for mesh in self.meshes:
            n += np.sum(
                mesh._normal_forces_from_strain(strains=np.full_like(mesh.z_btm, min_strain))
            )
        return n

    def moment_curvature_analysis(
        self,
        axial_force: float,
        kappa_0: float = 0.0,
        kappa_inc: float = 1.0e-7,
        adaptive_step: bool = True,
        kappa_mult: float = 2.0,
        kappa_inc_max: float = 5e-6,
        delta_m_min: float = 0.1,
        delta_m_max: float = 0.15,
        max_iter=100,
    ) -> MomentCurvatureResults:
        # initialize result container
        n_min = self.min_compression_force()
        n_max = self.max_tension_force()

        if not (n_min <= axial_force <= n_max):
            raise ValueError(f"Axial force is greater than the capacity"
                             f"of cross-section. Choose from: "
                             f"({n_min/1E3:.2f} kN, {n_max/1E3:.2f} kN).")

        mc_results = MomentCurvatureResults(theta=self.theta, n_target=axial_force)

        # function, that performs the moment curvature analysis
        def _calculate_moment_curvature() -> bool:
            """
            Calculate moment curvature analysis for the Miako section.
            """
            if kappa == 0.0:
                mc_results.moments = np.append(mc_results.moments, 0.0)
                mc_results.curvature = np.append(mc_results.curvature, kappa)
                return False  # TODO: handle this

            ax_force_precision = abs(10_000 * abs(kappa) ** 0.5)

            ax_force_convergence = ax_force_precision + 1.0

            # initialize strain at the top of the section
            strain_top_i = 0.075
            strain_top_j = 0.0
            for mesh in self.meshes:
                if isinstance(mesh, ConcreteMesh):
                    min_strain_mesh = mesh.ssp.min_strain
                    if min_strain_mesh < strain_top_j:
                        strain_top_j = 1.01 * min_strain_mesh

            # iterate until the axial force converges or failure happen
            c = 0
            while abs(ax_force_convergence) > ax_force_precision:

                if c > max_iter:
                    return False

                strain_top = (strain_top_i + strain_top_j) / 2
                n = 0.0
                failures = []
                mc_results._failure_mesh = []

                for mesh in self.meshes:
                    ax_forces, failure = mesh._normal_forces(eps_0=strain_top, kappa=kappa)

                    if failure:
                        mc_results._failure_mesh.append(mesh)

                    failures.append(failure)

                    n += np.sum(ax_forces)

                if n - axial_force < 0:
                    strain_top_j = strain_top
                else:
                    strain_top_i = strain_top

                ax_force_convergence = n - axial_force

                c += 1

            if any(failures):
                print(f"Failure at strain: {strain_top}")
                print(f"Failure at curvature: {kappa}")
                mc_results._failure = True
                kappa_i = mc_results.curvature[-1]
                kappa_j = kappa

                ax_force_precision = abs(1_000 * kappa ** 0.5)
                ax_force_convergence = ax_force_precision + 1.0
                c = 0
                while abs(ax_force_convergence) > ax_force_precision:
                    if c > max_iter:
                        return False

                    kappa_fail = (kappa_i + kappa_j) / 2

                    for failed_mesh in mc_results._failure_mesh:
                        if isinstance(failed_mesh, SteelMesh):
                            z_fail = failed_mesh.z_btm[0]
                            fail_strain = failed_mesh._failure_strain
                            if not np.isclose(abs(fail_strain), failed_mesh.ssp_steel.max_strain, rtol=0.1):
                                continue
                            fail_strain = np.sign(fail_strain) * failed_mesh.ssp_steel.max_strain
                            strain_top_fail = fail_strain + kappa_fail * (self.miako_section.section_height - z_fail)
                        elif isinstance(failed_mesh, ConcreteMesh):
                            strain_top_fail = failed_mesh.ssp.min_strain
                        else:
                            raise AttributeError("Invalid mesh type.")

                    n = 0.0

                    for mesh in self.meshes:
                        ax_forces, _ = mesh._normal_forces(eps_0=strain_top_fail, kappa=kappa_fail)
                        n += sum(ax_forces)

                    if n - axial_force < 0.0:
                        kappa_j = kappa_fail
                    else:
                        kappa_i = kappa_fail

                    ax_force_convergence = n - axial_force

                    c += 1

                m = 0.0
                for mesh in self.meshes:
                    ax_forces, failure = mesh._normal_forces(eps_0=strain_top_fail, kappa=kappa_fail)
                    m += np.sum(ax_forces * mesh.z_btm)

                if np.isclose(m, 0.0):
                    return False

                mc_results.curvature = np.append(mc_results.curvature, kappa_fail)
                mc_results.moments = np.append(mc_results.moments, m)
                mc_results.axial_force_error = np.append(mc_results.axial_force_error, ax_force_convergence)

                return True

            m = 0.0
            for mesh in self.meshes:
                ax_forces, failure = mesh._normal_forces(eps_0=strain_top, kappa=kappa)
                m += np.sum(ax_forces * mesh.z_btm)

            mc_results.curvature = np.append(mc_results.curvature, kappa)
            mc_results.moments = np.append(mc_results.moments, m)
            mc_results.axial_force_error = np.append(mc_results.axial_force_error, ax_force_convergence)

            return True

        kappa = kappa_0

        i = 0
        no_add_loop = 0
        while not mc_results._failure:
            result_added = _calculate_moment_curvature()

            print(f"{i=}")

            if adaptive_step and not mc_results._failure:
                if i > 2:
                    m1 = mc_results.moments[-1]
                    m2 = mc_results.moments[-2]
                    m_diff = abs(m1 - m2) / m1

                    if m_diff <= delta_m_min:
                        kappa_inc *= kappa_mult
                    elif m_diff >= delta_m_max:
                        kappa_inc *= 1 / kappa_mult

                    if kappa_inc > kappa_inc_max:
                        kappa_inc = kappa_inc_max

            kappa -= kappa_inc

            if not result_added:
                no_add_loop += 1
            else:
                no_add_loop = 0

            if no_add_loop > 10:
                return mc_results

            i += 1

        return mc_results


class ConcreteMesh:
    """
    Create concrete mesh for the Miako section.

    :param miako_mesh: Miako mesh object.
    :param part: Part of the concrete section. Choose from: ["precast", "insitu"].
    """
    def __init__(
        self,
        miako_mesh: MiakoMesh,
        part: str,
    ) -> None:
        """Initialise concrete mesh for the Miako section."""
        if part not in ["precast", "insitu"]:
            raise AttributeError(f"Invalid part: {part}. Choose from: ['precast', 'insitu'].")
        self.miako_mesh = miako_mesh
        self.part = part
        self._create_concrete_mesh()

    def _create_concrete_mesh(self):
        """Create concrete mesh for the Miako section."""
        if self.part == "precast":
            self._create_precast_mesh()
        elif self.part == "insitu":
            self._create_insitu_mesh()
        else:
            raise AttributeError(f"Invalid part: {self.part}. Choose from: ['precast', 'insitu'].")

        self.ssp = ConcreteStressStrainProfile(
            concrete=self.concrete,
            compression_profile=self.miako_mesh.concrete_compressive_stress_strain_profile,
            tension_profile=self.miako_mesh.concrete_tension_stress_strain_profile,
            limit_state=self.miako_mesh.limit_state
        )

        # Reverse the "z" array if the section is rotated by 180 degrees.
        if np.isclose(self.miako_mesh.theta, np.pi):
            self.z_btm = self.miako_mesh.miako_section.section_height - self.z_btm

    def _create_precast_mesh(self):
        """Create concrete mesh of the POT beam."""
        dh = self.miako_mesh.h_interval
        n_intervals = int(45.0 / dh)
        self.z_btm = np.linspace(0.0, 45.0 - dh, n_intervals) + dh / 2
        self.b = np.full_like(self.z_btm, self.miako_mesh.miako_section.n_pots * 130.0)
        self.area = self.b * self.miako_mesh.h_interval

        self.concrete = Concrete(
            strength_class=self.miako_mesh.miako_section.precast_concrete_strength_class,
        )

    def _create_insitu_mesh(self):
        """Create concrete mesh of the monolithic part."""
        dh = self.miako_mesh.h_interval
        n_intervals = int((self.miako_mesh.miako_section.section_height - 45.0) / dh)
        self.z_btm = np.linspace(45.0, self.miako_mesh.miako_section.section_height - dh, n_intervals) + dh / 2

        self.b = np.zeros_like(self.z_btm)

        n = self.miako_mesh.miako_section.n_pots

        match self.miako_mesh.miako_section.miako_height:
            case 80.0:
                h1 = 45.0  # 60 - 15
                b1 = n * 130.0  # n * (160 - 2 * 15) for cover

                # from POT beam to the top of MIAKO block
                h2 = h1 + 20.0
                b2 = 100.0 + (n - 1) * 160.0

                # from top of MIAKO block to the top of the section
                h3 = self.miako_mesh.miako_section.section_height - 15.0
                b3 = self.miako_mesh.miako_section.b_eff

                self.b[self.z_btm <= h3] = b3
                self.b[self.z_btm <= h2] = b2
                self.b[self.z_btm <= h1] = b1

            case 150.0 | 190.0 | 230.0:
                # from bottom to the top of the POT beam
                h1 = 45.0
                b1 = n * 130.0

                # from top of the POT beam to the bottom of the triangle
                h2 = h1 + self.miako_mesh.miako_section.miako_height - 112.0  # 60+52
                b2 = 100.0 + (n - 1) * 160.0

                # from the bottom of the triangle to the bottom of the block grip
                h3 = h2 + 40.0
                b3 = 100.0 + (n - 1) * 160.0 + 2 * (15 / 40) * (self.z_btm[self.z_btm <= h3] - h2)

                # from the bottom of the block grip to the top of the block grip
                h4 = h3 + 12.0
                b4 = 100.0 + (n - 1) * 160.0

                # from the top of the block grip to the top of the section
                h5 = self.miako_mesh.miako_section.section_height
                b5 = self.miako_mesh.miako_section.b_eff

                self.b[self.z_btm <= h5] = b5
                self.b[self.z_btm <= h4] = b4
                self.b[self.z_btm <= h3] = b3
                self.b[self.z_btm <= h2] = b2
                self.b[self.z_btm <= h1] = b1

            case 250.0:
                # from bottom to the top of the POT beam
                h1 = 45.0
                b1 = n * 130.0

                # from top of the POT beam to the bottom of the triangle
                h2 = h1 + 115.0
                b2 = 100.0 + (n - 1) * 160.0

                # from the bottom of the triangle to the top of the first triangle
                h3 = h2 + 33.0
                b3 = 100.0 + (n - 1) * 160.0 + 2 * (99 / 33) * (self.z_btm[self.z_btm <= h3] - h2)

                # from the top of the first triangle to the bottom of the top grip
                h4 = h3 + 26.0
                b4 = 100.0 + (n - 1) * 160.0 + 2 * 99.0

                # from the bottom of the top grip to the top of the top grip
                h5 = h4 + 16.0
                b5 = 100.0 + (n - 1) * 160.0 + 2 * 84.0

                self.b[self.z_btm <= h5] = b5
                self.b[self.z_btm <= h4] = b4
                self.b[self.z_btm <= h3] = b3
                self.b[self.z_btm <= h2] = b2
                self.b[self.z_btm <= h1] = b1

            case _:
                raise AttributeError(f"Invalid MIAKO height: {self.miako_mesh.miako_section.miako_height}.")

        self.area = self.b * self.miako_mesh.h_interval

        self.concrete = Concrete(
            strength_class=self.miako_mesh.miako_section.insitu_concrete_strength_class,
        )

    def _normal_forces(
        self,
        eps_0: float,
        kappa: float,
    ) -> tuple[npt.NDArray[np.float64], bool]:
        x = eps_0 / kappa

        # distance from the centroid of each
        z_c = - self.miako_mesh.miako_section.section_height + x + self.z_btm
        strains = - np.abs(kappa) * z_c
        stresses = self.ssp.get_stresses(strains)

        normal_forces = stresses * self.area

        within_limits = self.ssp.min_strain <= strains
        failure = not np.all(within_limits)

        return normal_forces, failure

    def _normal_forces_from_strain(
        self,
        strains: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        stresses = self.ssp.get_stresses(strains)
        normal_forces = stresses * self.area

        return normal_forces

    def _min_compression_strain(self, x: float) -> float:
        """
        Return minimum (compression is negative) strain in the concrete.

        According to EN 1992-1-1:2004, 6.1(5), Fig. 6.1.

        In parts of cross-sections which are subjected to approximately concentric
        loading (ed/h <= 0,1), such as compression flanges of box girders, the mean
        compressive strain in that part of the section should be limited to epsilon_c2
        (or epsilon_c3 if the bilinear relation of Figure 3.4 is used)

        :param x: Depth of the neutral axis in mm.
        """
        match self.ssp.compression_profile:
            case ConcreteCompressionSSP.BILINEAR:
                eps_c = self.concrete.eps_c3
                eps_cu = self.concrete.eps_cu3
            case ConcreteCompressionSSP.PARABOLIC_RECTANGULAR:
                eps_c = self.concrete.eps_c2
                eps_cu = self.concrete.eps_cu2
            case _:
                raise AttributeError("Invalid concrete compression stress-strain profile.")

        h = self.miako_mesh.miako_section.section_height

        if x <= h:
            return - eps_cu
        else:
            return -min(x * abs(eps_c) / (x - (1 - abs(eps_c) / abs(eps_cu)) * h), abs(eps_cu))


class SteelMesh:
    """
    Create concrete mesh for the Miako section.

    :param miako_mesh: Miako mesh object.
    :param part: Part of the steel section. Choose from: ["precast_btm", "precast_top", "wwf", "top"].
    """
    def __init__(
        self,
        miako_mesh: MiakoMesh,
        part: str,
    ) -> None:
        """Initialise steel mesh for the Miako section."""
        self.miako_mesh = miako_mesh
        self.part = part
        self._create_steel_mesh()
        self._failure_strain: float | None = None

    def _create_steel_mesh(self) -> None:
        """
        Create steel mesh for the Miako section.

        Also create concrete material, so the overlapping geometries can be avoided.
        """
        if self.part == "precast_btm":
            self._create_bottom_precast_mesh()
        elif self.part == "precast_top":
            self._create_top_precast_mesh()
        elif self.part == "wwf":
            self._create_wwf_mesh()
        elif self.part == "top":
            self._create_top_reinforcement_mesh()
        else:
            raise AttributeError(f"Invalid part: {self.part}. Choose from: "
                                 f"['precast_btm', 'precast_top', 'wwf', 'top'].")

        if not self.created:
            return

        self.area = self.number * np.pi * self.diameter ** 2 / 4

        self.ssp_steel = ReinforcementStressStrainProfile(
            steel=self.steel,
            profile=self.miako_mesh.reinforcement_stress_strain_profile,
            limit_state=self.miako_mesh.limit_state
        )
        self.ssp_concrete = ConcreteStressStrainProfile(
            concrete=self.concrete,
            compression_profile=self.miako_mesh.concrete_compressive_stress_strain_profile,
            tension_profile=self.miako_mesh.concrete_tension_stress_strain_profile,
            limit_state=self.miako_mesh.limit_state,
        )

        if np.isclose(self.miako_mesh.theta, np.pi):
            self.z_btm = self.miako_mesh.miako_section.section_height - self.z_btm

    def _create_bottom_precast_mesh(self) -> None:
        self.created = True

        n = self.miako_mesh.miako_section.n_pots

        pot_dict = POT_BEAMS[self.miako_mesh.miako_section.pot_label]

        d_s = pot_dict["d_bottom_sides"]
        self.diameter = np.array([d_s])
        self.number = np.array([n * 2], dtype=np.int32)
        self.z_btm = np.array([15.0 + d_s / 2])

        if pot_dict["d_bottom_middle"]:
            d_m = pot_dict["d_bottom_middle"]
            self.diameter = np.append(self.diameter, d_m)
            self.number = np.append(self.number, n * 1)
            self.z_btm = np.append(self.z_btm, 15.0 + d_m / 2)

        self.steel = Steel(steel_grade=self.miako_mesh.miako_section.precast_steel_grade)
        self.concrete = Concrete(strength_class=self.miako_mesh.miako_section.precast_concrete_strength_class)

    def _create_top_precast_mesh(self) -> None:
        self.created = False

        n = self.miako_mesh.miako_section.n_pots

        pot_dict = POT_BEAMS[self.miako_mesh.miako_section.pot_label]

        if self.miako_mesh.consider_top_pot_reinforcement:
            d_t = pot_dict["d_top"]
            self.diameter = np.array([d_t])
            self.number = np.array([n * 1])
            self.z_btm = np.array(
                [
                    pot_dict["height_with_rebar"] - 15 - d_t / 2
                ]
            )
            self.created = True

        self.steel = Steel(steel_grade=self.miako_mesh.miako_section.precast_steel_grade)
        self.concrete = Concrete(strength_class=self.miako_mesh.miako_section.insitu_concrete_strength_class)

    def _create_top_reinforcement_mesh(self) -> None:
        self.created = False
        c = self.miako_mesh.miako_section.concrete_cover
        h = self.miako_mesh.miako_section.section_height

        if self.miako_mesh.consider_top_reinforcement:
            if self.miako_mesh.miako_section.top_reinforcement_number:
                d_top = self.miako_mesh.miako_section.top_reinforcement_diameter
                self.diameter = np.array([d_top])
                self.number = np.array([self.miako_mesh.miako_section.top_reinforcement_number])
                self.z_btm = np.array([
                        h - c - self.miako_mesh.miako_section.top_reinforcement_diameter / 2
                    ]
                )
                self.steel = Steel(steel_grade=self.miako_mesh.miako_section.top_reinforcement_steel_grade)
                self.concrete = Concrete(strength_class=self.miako_mesh.miako_section.insitu_concrete_strength_class)
                self.created = True

    def _create_wwf_mesh(self) -> None:
        self.created = False

        c = self.miako_mesh.miako_section.concrete_cover
        h = self.miako_mesh.miako_section.section_height

        d_top = 0.0
        if self.miako_mesh.consider_top_reinforcement:
            if self.miako_mesh.miako_section.top_reinforcement_number:
                d_top = self.miako_mesh.miako_section.top_reinforcement_diameter

        if not np.isclose(self.miako_mesh.miako_section.slab_height, 0.0):
            d_wwf = self.miako_mesh.miako_section.wwf_diameter
            s_wwf = self.miako_mesh.miako_section.wwf_spacing
            n_wwf = int(np.floor(self.miako_mesh.miako_section.b_eff / s_wwf))
            self.diameter = np.array([d_wwf])
            self.number = np.array([n_wwf])
            self.z_btm = np.array([
                    max(h - c - d_top + d_wwf / 2, h - c - d_wwf / 2)
                ]
            )
            self.steel = Steel(steel_grade=self.miako_mesh.miako_section.wwf_steel_grade)
            self.concrete = Concrete(strength_class=self.miako_mesh.miako_section.insitu_concrete_strength_class)
            self.created = True

    def _normal_forces(
        self,
        eps_0: float,
        kappa: float,
    ) -> tuple[npt.NDArray[np.float64], bool]:
        x = eps_0 / kappa

        # distance from the centroid of each
        z_c = - self.miako_mesh.miako_section.section_height + x + self.z_btm
        strains = - np.abs(kappa) * z_c
        steel_stresses = self.ssp_steel.get_stresses(strains)
        conc_stresses = self.ssp_concrete.get_stresses(strains)

        normal_forces = (steel_stresses - conc_stresses) * self.area

        within_limits = (self.ssp_steel.min_strain <= strains) & (strains <= self.ssp_steel.max_strain)
        failure = not np.all(within_limits)

        if failure:
            self._failure_strain = strains[~within_limits][0]

        return normal_forces, failure

    def _normal_forces_from_strain(
        self,
        strains: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        steel_stresses = self.ssp_steel.get_stresses(strains)
        conc_stresses = self.ssp_concrete.get_stresses(strains)

        normal_forces = (steel_stresses - conc_stresses) * self.area

        return normal_forces


def main():

    import matplotlib.pyplot as plt

    miako = MiakoSection(n_pots=1, wwf_steel_grade="B500B")

    comp = [ConcreteCompressionSSP.NONLINEAR, ConcreteCompressionSSP.PARABOLIC_RECTANGULAR]
    tens = [ConcreteTensionSSP.ELASTIC_PLASTIC_WITH_SOFTENING, ConcreteTensionSSP.ELASTIC_BRITTLE]
    steel = [ReinforcementSSP.ELASTIC_PLASTIC_WITH_HARDENING, ReinforcementSSP.ELASTIC_PLASTIC]

    axial = [0e3]

    labels = []
    mc_res = []

    for cssp in comp:
        for tssp in tens:
            for st in steel:
                for ax in axial:
                    labels.append(
                        f"{cssp.name} - {tssp.name} - {st.name} - {ax/1000:.0f} kN"
                    )

                    mesh_pos = miako._create_mesh(
                        theta=np.pi,
                        concrete_compression_stress_strain_profile=cssp,
                        concrete_tension_stress_strain_profile=tssp,
                        reinforcement_stress_strain_profile=st,
                        limit_state="SLS"
                    )
                    mc_res.append(mesh_pos.moment_curvature_analysis(axial_force=ax, adaptive_step=True))

                    fig, axs = plt.subplots()
                    mc_res[-1].plot_results(ax=axs, fmt="-")
                    plt.title(labels[-1])
                    plt.show()

    # # res_pos.plot_results()
    #
    # mesh_neg = miako.create_mesh(
    #     theta=np.pi,
    #     concrete_compression_stress_strain_profile=ConcreteCompressionSSP.PARABOLIC_RECTANGULAR,
    #     concrete_tension_stress_strain_profile=ConcreteTensionSSP.ELASTIC_PLASTIC_WITH_SOFTENING
    # )
    # res_neg = mesh_neg.moment_curvature_analysis(axial_force=0.0, adaptive_step=True)

    # mesh_pos.meshes[0].ssp.plot_profile()
    # mesh_pos.meshes[-1].ssp_steel.plot_profile()


if __name__ == "__main__":
    main()
