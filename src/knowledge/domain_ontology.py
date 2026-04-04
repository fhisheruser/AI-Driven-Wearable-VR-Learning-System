"""Domain Ontology for structured educational content.

Maps educational concepts to structured knowledge aligned with
standard curricula as described in the paper.
"""

from dataclasses import dataclass, field


@dataclass
class Concept:
    """A single educational concept within the ontology."""
    id: str
    name: str
    domain: str
    description: str
    difficulty_level: int  # 1-5
    prerequisites: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)
    visualization_hint: str = "3d_model"
    curriculum_level: str = "undergraduate"
    keywords: list[str] = field(default_factory=list)
    explanation: str = ""
    steps: list[str] = field(default_factory=list)


class DomainOntology:
    """Educational domain ontology with structured concept relationships."""

    def __init__(self):
        self.concepts: dict[str, Concept] = {}
        self._build_ontology()

    def _build_ontology(self):
        """Build the complete educational ontology across all domains."""
        self._build_chemistry_ontology()
        self._build_biology_ontology()
        self._build_physics_ontology()
        self._build_mathematics_ontology()
        self._build_anatomy_ontology()
        self._build_engineering_ontology()

    def _build_chemistry_ontology(self):
        concepts = [
            Concept(
                id="chem_atom", name="Atom", domain="chemistry",
                description="The basic unit of matter consisting of a nucleus surrounded by electrons",
                difficulty_level=1, related_concepts=["chem_electron", "chem_proton", "chem_neutron"],
                visualization_hint="3d_model", curriculum_level="K12",
                keywords=["atom", "atomic", "matter", "particle"],
                explanation="An atom is the smallest unit of ordinary matter that forms a chemical element. "
                "Every solid, liquid, gas, and plasma is composed of atoms. An atom consists of a nucleus "
                "made of protons and neutrons, surrounded by a cloud of electrons.",
                steps=["Identify the nucleus at the center", "Count protons to determine element",
                       "Observe electron shells orbiting the nucleus"],
            ),
            Concept(
                id="chem_molecule", name="Molecule", domain="chemistry",
                description="A group of atoms bonded together representing the smallest unit of a chemical compound",
                difficulty_level=1, prerequisites=["chem_atom", "chem_bond"],
                related_concepts=["chem_bond", "chem_compound"],
                visualization_hint="3d_model", curriculum_level="K12",
                keywords=["molecule", "molecular", "compound"],
                explanation="A molecule is an electrically neutral group of two or more atoms held together by "
                "chemical bonds. Molecules are distinguished from ions by their lack of electrical charge.",
            ),
            Concept(
                id="chem_bond", name="Chemical Bond", domain="chemistry",
                description="An attraction between atoms that allows formation of chemical substances",
                difficulty_level=2, prerequisites=["chem_atom", "chem_electron"],
                related_concepts=["chem_covalent", "chem_ionic"],
                visualization_hint="animation", curriculum_level="K12",
                keywords=["bond", "bonding", "covalent", "ionic", "metallic"],
                explanation="A chemical bond is a lasting attraction between atoms, ions or molecules that "
                "enables the formation of chemical compounds. The three main types are ionic bonds, "
                "covalent bonds, and metallic bonds.",
                steps=["Understand electron sharing (covalent)", "Understand electron transfer (ionic)",
                       "Compare bond strengths and properties"],
            ),
            Concept(
                id="chem_electron", name="Electron", domain="chemistry",
                description="A subatomic particle with a negative electric charge",
                difficulty_level=1, related_concepts=["chem_atom", "chem_orbital"],
                visualization_hint="3d_model", curriculum_level="K12",
                keywords=["electron", "charge", "orbital", "shell"],
                explanation="An electron is a subatomic particle with a negative elementary electric charge. "
                "Electrons belong to the first generation of the lepton particle family. They orbit the "
                "nucleus in specific energy levels called shells.",
            ),
            Concept(
                id="chem_water", name="Water Molecule", domain="chemistry",
                description="H2O molecule with bent geometry and polar properties",
                difficulty_level=1, prerequisites=["chem_molecule", "chem_bond"],
                visualization_hint="3d_model", curriculum_level="K12",
                keywords=["water", "h2o", "polar", "hydrogen"],
                explanation="Water (H2O) is a polar molecule consisting of two hydrogen atoms covalently "
                "bonded to one oxygen atom. The molecule has a bent geometry with a bond angle of 104.5 degrees.",
            ),
            Concept(
                id="chem_periodic_table", name="Periodic Table", domain="chemistry",
                description="Tabular arrangement of chemical elements organized by atomic number",
                difficulty_level=2, prerequisites=["chem_atom"],
                visualization_hint="contextual_overlay", curriculum_level="K12",
                keywords=["periodic table", "element", "group", "period"],
                explanation="The periodic table is a tabular display of the chemical elements organized by "
                "atomic number, electron configuration, and recurring chemical properties.",
            ),
            Concept(
                id="chem_reaction", name="Chemical Reaction", domain="chemistry",
                description="Process involving rearrangement of atoms to form new substances",
                difficulty_level=2, prerequisites=["chem_molecule", "chem_bond"],
                visualization_hint="animation", curriculum_level="undergraduate",
                keywords=["reaction", "reactant", "product", "equation", "balance"],
                explanation="A chemical reaction is a process that leads to the chemical transformation of "
                "one set of chemical substances (reactants) to another (products).",
                steps=["Identify reactants", "Write unbalanced equation", "Balance the equation",
                       "Identify products", "Calculate stoichiometry"],
            ),
        ]
        for c in concepts:
            self.concepts[c.id] = c

    def _build_biology_ontology(self):
        concepts = [
            Concept(
                id="bio_cell", name="Cell", domain="biology",
                description="The basic structural and functional unit of life",
                difficulty_level=1, related_concepts=["bio_nucleus", "bio_mitochondria"],
                visualization_hint="3d_model", curriculum_level="K12",
                keywords=["cell", "cellular", "organelle"],
                explanation="The cell is the basic structural, functional, and biological unit of all known "
                "organisms. Cells are the smallest unit of life that can replicate independently. "
                "They consist of cytoplasm enclosed within a membrane with many biomolecules.",
                steps=["Observe the cell membrane boundary", "Identify the nucleus containing DNA",
                       "Locate key organelles: mitochondria, ER, Golgi apparatus"],
            ),
            Concept(
                id="bio_dna", name="DNA", domain="biology",
                description="Double helix molecule carrying genetic instructions for life",
                difficulty_level=2, prerequisites=["bio_cell", "bio_nucleus"],
                related_concepts=["bio_rna", "bio_gene", "bio_chromosome"],
                visualization_hint="3d_model", curriculum_level="undergraduate",
                keywords=["dna", "double helix", "genetic", "nucleotide", "base pair"],
                explanation="DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains "
                "that coil around each other to form a double helix. It carries genetic instructions for "
                "development, functioning, growth, and reproduction of all known organisms.",
            ),
            Concept(
                id="bio_mitochondria", name="Mitochondria", domain="biology",
                description="Powerhouse of the cell responsible for energy production",
                difficulty_level=2, prerequisites=["bio_cell"],
                visualization_hint="3d_model", curriculum_level="undergraduate",
                keywords=["mitochondria", "atp", "energy", "respiration", "powerhouse"],
                explanation="Mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic "
                "cells. They generate most of the cell's supply of ATP, used as a source of chemical energy.",
            ),
            Concept(
                id="bio_photosynthesis", name="Photosynthesis", domain="biology",
                description="Process by which plants convert light energy into chemical energy",
                difficulty_level=2, prerequisites=["bio_cell"],
                visualization_hint="animation", curriculum_level="K12",
                keywords=["photosynthesis", "chlorophyll", "light", "glucose", "carbon dioxide", "oxygen"],
                explanation="Photosynthesis is the process used by plants, algae and certain bacteria to "
                "convert light energy into chemical energy stored in glucose. It occurs primarily in "
                "chloroplasts using chlorophyll.",
                steps=["Light absorption by chlorophyll", "Water splitting (photolysis)",
                       "Carbon fixation (Calvin cycle)", "Glucose synthesis"],
            ),
            Concept(
                id="bio_nucleus", name="Cell Nucleus", domain="biology",
                description="Membrane-bound organelle containing the cell's genetic material",
                difficulty_level=1, prerequisites=["bio_cell"],
                visualization_hint="3d_model", curriculum_level="K12",
                keywords=["nucleus", "nuclear", "chromatin", "nucleolus"],
                explanation="The cell nucleus is a membrane-bound organelle found in eukaryotic cells. "
                "It contains most of the cell's genetic material organized as DNA molecules.",
            ),
        ]
        for c in concepts:
            self.concepts[c.id] = c

    def _build_physics_ontology(self):
        concepts = [
            Concept(
                id="phys_solar_system", name="Solar System", domain="physics",
                description="The Sun and all objects gravitationally bound to it",
                difficulty_level=1, visualization_hint="3d_model", curriculum_level="K12",
                keywords=["solar system", "planet", "sun", "orbit", "gravity"],
                explanation="The Solar System consists of the Sun and everything gravitationally bound to it — "
                "eight planets, dwarf planets, moons, asteroids, comets, and meteoroids. The planets orbit "
                "the Sun in elliptical paths.",
                steps=["Start with the Sun at the center", "Observe inner rocky planets: Mercury, Venus, Earth, Mars",
                       "Note the asteroid belt", "Observe outer gas giants: Jupiter, Saturn, Uranus, Neptune"],
            ),
            Concept(
                id="phys_gravity", name="Gravity", domain="physics",
                description="Fundamental force of attraction between objects with mass",
                difficulty_level=2, visualization_hint="animation", curriculum_level="K12",
                keywords=["gravity", "gravitational", "force", "mass", "weight", "newton"],
                explanation="Gravity is a fundamental force by which all things with mass or energy are "
                "attracted to one another. On Earth, gravity gives weight to physical objects.",
            ),
            Concept(
                id="phys_electromagnetic", name="Electromagnetic Spectrum", domain="physics",
                description="The range of frequencies of electromagnetic radiation",
                difficulty_level=3, prerequisites=["phys_wave"],
                visualization_hint="animation", curriculum_level="undergraduate",
                keywords=["electromagnetic", "spectrum", "radiation", "wavelength", "frequency",
                          "radio", "microwave", "infrared", "visible", "ultraviolet", "xray", "gamma"],
                explanation="The electromagnetic spectrum is the range of frequencies of electromagnetic "
                "radiation and their respective wavelengths and photon energies.",
            ),
            Concept(
                id="phys_wave", name="Wave", domain="physics",
                description="A disturbance that transfers energy through matter or space",
                difficulty_level=2, visualization_hint="animation", curriculum_level="K12",
                keywords=["wave", "amplitude", "frequency", "wavelength", "period", "oscillation"],
                explanation="A wave is a disturbance in a field that propagates energy and momentum without "
                "the transport of matter. Waves have amplitude, wavelength, frequency, and period.",
            ),
            Concept(
                id="phys_newton_laws", name="Newton's Laws of Motion", domain="physics",
                description="Three fundamental laws describing the relationship between body motion and forces",
                difficulty_level=2, visualization_hint="animation", curriculum_level="K12",
                keywords=["newton", "law", "motion", "force", "inertia", "action", "reaction"],
                explanation="Newton's three laws of motion describe the relationship between a body and "
                "the forces acting upon it: inertia, F=ma, and action-reaction.",
                steps=["First Law: Object at rest stays at rest (inertia)",
                       "Second Law: Force equals mass times acceleration (F=ma)",
                       "Third Law: Every action has an equal and opposite reaction"],
            ),
        ]
        for c in concepts:
            self.concepts[c.id] = c

    def _build_mathematics_ontology(self):
        concepts = [
            Concept(
                id="math_pythagorean", name="Pythagorean Theorem", domain="mathematics",
                description="a^2 + b^2 = c^2 for right triangles",
                difficulty_level=1, visualization_hint="animation", curriculum_level="K12",
                keywords=["pythagorean", "theorem", "triangle", "hypotenuse", "right angle"],
                explanation="The Pythagorean theorem states that in a right triangle, the square of the "
                "hypotenuse equals the sum of the squares of the other two sides: a^2 + b^2 = c^2.",
                steps=["Identify the right angle", "Label sides a, b (legs) and c (hypotenuse)",
                       "Apply a^2 + b^2 = c^2", "Solve for the unknown side"],
            ),
            Concept(
                id="math_derivative", name="Derivative", domain="mathematics",
                description="Rate of change of a function at a given point",
                difficulty_level=3, prerequisites=["math_function", "math_limit"],
                visualization_hint="animation", curriculum_level="undergraduate",
                keywords=["derivative", "differentiation", "rate", "slope", "tangent", "calculus"],
                explanation="A derivative measures how a function changes as its input changes. "
                "Geometrically, it represents the slope of the tangent line to the function's graph.",
            ),
            Concept(
                id="math_function", name="Function", domain="mathematics",
                description="A relation that assigns exactly one output to each input",
                difficulty_level=1, visualization_hint="animation", curriculum_level="K12",
                keywords=["function", "input", "output", "domain", "range", "mapping"],
                explanation="A function is a relation between sets that associates every element of the "
                "first set (domain) to exactly one element of the second set (range).",
            ),
            Concept(
                id="math_integral", name="Integral", domain="mathematics",
                description="Accumulation of quantities and area under curves",
                difficulty_level=3, prerequisites=["math_derivative"],
                visualization_hint="animation", curriculum_level="undergraduate",
                keywords=["integral", "integration", "area", "accumulation", "antiderivative"],
                explanation="Integration is the reverse of differentiation. It computes the accumulation "
                "of quantities and can be interpreted as the area under a curve.",
            ),
        ]
        for c in concepts:
            self.concepts[c.id] = c

    def _build_anatomy_ontology(self):
        concepts = [
            Concept(
                id="anat_heart", name="Human Heart", domain="anatomy",
                description="Muscular organ that pumps blood through the circulatory system",
                difficulty_level=2, visualization_hint="3d_model", curriculum_level="undergraduate",
                keywords=["heart", "cardiac", "atrium", "ventricle", "valve", "cardiovascular"],
                explanation="The human heart is a muscular organ about the size of a fist. It has four "
                "chambers: two atria (upper) and two ventricles (lower). It pumps blood through the "
                "circulatory system via a network of arteries and veins.",
                steps=["Identify the four chambers", "Trace blood flow from right atrium to lungs",
                       "Follow oxygenated blood from lungs to left atrium", "Observe pumping to the body"],
            ),
            Concept(
                id="anat_brain", name="Human Brain", domain="anatomy",
                description="Central organ of the nervous system controlling all body functions",
                difficulty_level=3, visualization_hint="3d_model", curriculum_level="undergraduate",
                keywords=["brain", "cerebrum", "cerebellum", "brainstem", "cortex", "neuron", "lobe"],
                explanation="The human brain is the central organ of the nervous system. It consists of "
                "the cerebrum, cerebellum, and brainstem, containing approximately 86 billion neurons.",
            ),
            Concept(
                id="anat_skeleton", name="Human Skeleton", domain="anatomy",
                description="Internal framework of 206 bones providing structure and protection",
                difficulty_level=1, visualization_hint="3d_model", curriculum_level="K12",
                keywords=["skeleton", "bone", "skeletal", "skull", "spine", "rib", "femur"],
                explanation="The adult human skeleton consists of 206 bones. It provides structural support, "
                "protects organs, enables movement, produces blood cells, and stores minerals.",
            ),
            Concept(
                id="anat_lungs", name="Human Lungs", domain="anatomy",
                description="Primary organs of the respiratory system for gas exchange",
                difficulty_level=2, visualization_hint="3d_model", curriculum_level="undergraduate",
                keywords=["lung", "respiratory", "alveoli", "bronchi", "breathing", "oxygen", "diaphragm"],
                explanation="The lungs are the primary organs of respiration. They facilitate gas exchange "
                "between the air and the bloodstream through tiny air sacs called alveoli.",
            ),
        ]
        for c in concepts:
            self.concepts[c.id] = c

    def _build_engineering_ontology(self):
        concepts = [
            Concept(
                id="eng_gear", name="Gear System", domain="engineering",
                description="Mechanical device using toothed wheels to transmit torque",
                difficulty_level=2, visualization_hint="animation", curriculum_level="undergraduate",
                keywords=["gear", "torque", "transmission", "ratio", "mechanical"],
                explanation="A gear system uses toothed wheels that mesh together to transmit rotational "
                "motion and torque between shafts. Gear ratios determine speed and force relationships.",
                steps=["Understand gear teeth meshing", "Calculate gear ratio",
                       "Determine speed and torque relationships", "Analyze power transmission"],
            ),
            Concept(
                id="eng_circuit", name="Electric Circuit", domain="engineering",
                description="A closed path through which electric current flows",
                difficulty_level=2, visualization_hint="animation", curriculum_level="K12",
                keywords=["circuit", "current", "voltage", "resistance", "ohm", "series", "parallel"],
                explanation="An electric circuit is a closed loop through which current flows from a power "
                "source through conductors and components back to the source.",
                steps=["Identify the power source", "Trace the current path",
                       "Apply Ohm's Law (V=IR)", "Calculate total resistance"],
            ),
            Concept(
                id="eng_bridge", name="Bridge Structure", domain="engineering",
                description="Structure spanning physical obstacles for passage",
                difficulty_level=2, visualization_hint="3d_model", curriculum_level="undergraduate",
                keywords=["bridge", "truss", "beam", "arch", "suspension", "stress", "load"],
                explanation="A bridge is a structure built to span a physical obstacle. Different types "
                "include beam, arch, truss, and suspension bridges, each with unique load-bearing properties.",
            ),
        ]
        for c in concepts:
            self.concepts[c.id] = c

    def find_concept(self, query: str, domain: str | None = None) -> Concept | None:
        """Find the most relevant concept for a query."""
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for concept in self.concepts.values():
            if domain and concept.domain != domain:
                continue

            score = 0
            # Check name match
            if concept.name.lower() in query_lower:
                score += 10
            # Check keyword matches
            for kw in concept.keywords:
                if kw in query_lower:
                    score += 3

            if score > best_score:
                best_score = score
                best_match = concept

        return best_match

    def get_prerequisites(self, concept_id: str) -> list[Concept]:
        """Get prerequisite concepts for a given concept."""
        concept = self.concepts.get(concept_id)
        if not concept:
            return []
        return [self.concepts[pid] for pid in concept.prerequisites if pid in self.concepts]

    def get_related(self, concept_id: str) -> list[Concept]:
        """Get related concepts."""
        concept = self.concepts.get(concept_id)
        if not concept:
            return []
        return [self.concepts[rid] for rid in concept.related_concepts if rid in self.concepts]

    def get_by_domain(self, domain: str) -> list[Concept]:
        """Get all concepts in a domain."""
        return [c for c in self.concepts.values() if c.domain == domain]

    def get_by_difficulty(self, level: int, domain: str | None = None) -> list[Concept]:
        """Get concepts at a specific difficulty level."""
        concepts = self.concepts.values()
        if domain:
            concepts = [c for c in concepts if c.domain == domain]
        return [c for c in concepts if c.difficulty_level == level]
