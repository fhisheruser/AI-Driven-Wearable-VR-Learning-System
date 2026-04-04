"""VR/AR Visualization Engine for immersive educational content.

Generates 3D scene descriptions, animations, and interactive visualizations
optimized for real-time rendering on mixed-reality displays.
Target: >= 30 fps, <= 2 second generation latency.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SceneObject:
    """A 3D object in the visualization scene."""
    id: str
    type: str  # "sphere", "cylinder", "box", "custom", "text", "arrow"
    position: list[float] = field(default_factory=lambda: [0, 0, 0])
    rotation: list[float] = field(default_factory=lambda: [0, 0, 0])
    scale: list[float] = field(default_factory=lambda: [1, 1, 1])
    color: str = "#ffffff"
    opacity: float = 1.0
    label: str = ""
    interactive: bool = False
    animation: dict | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position,
            "rotation": self.rotation,
            "scale": self.scale,
            "color": self.color,
            "opacity": self.opacity,
            "label": self.label,
            "interactive": self.interactive,
            "animation": self.animation,
            "metadata": self.metadata,
        }


@dataclass
class VisualizationScene:
    """A complete 3D scene for rendering."""
    scene_id: str
    title: str
    objects: list[SceneObject] = field(default_factory=list)
    camera_position: list[float] = field(default_factory=lambda: [0, 2, 5])
    camera_target: list[float] = field(default_factory=lambda: [0, 0, 0])
    lighting: dict = field(default_factory=lambda: {
        "ambient": {"color": "#404040", "intensity": 0.5},
        "directional": {"color": "#ffffff", "intensity": 0.8, "position": [5, 10, 5]},
    })
    background_color: str = "#000011"
    annotations: list[dict] = field(default_factory=list)
    animation_timeline: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    generation_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "objects": [obj.to_dict() for obj in self.objects],
            "camera_position": self.camera_position,
            "camera_target": self.camera_target,
            "lighting": self.lighting,
            "background_color": self.background_color,
            "annotations": self.annotations,
            "animation_timeline": self.animation_timeline,
            "steps": self.steps,
            "generation_time_ms": self.generation_time_ms,
        }


class VisualizationEngine:
    """Generates immersive VR/AR content for educational visualization.

    Creates 3D scene descriptions that can be rendered by the frontend
    Three.js engine in the mixed-reality display.
    """

    # Element colors for chemistry
    ELEMENT_COLORS = {
        "H": "#ffffff", "C": "#333333", "N": "#3050F8", "O": "#FF0D0D",
        "S": "#FFFF30", "P": "#FF8000", "Cl": "#1FF01F", "Na": "#AB5CF2",
        "Fe": "#E06633", "Ca": "#3DFF00", "K": "#8F40D4",
    }

    def generate_scene(self, viz_params: dict, concept_data: dict | None = None) -> VisualizationScene:
        """Generate a visualization scene from parameters.

        Args:
            viz_params: Visualization parameters from the knowledge engine
            concept_data: Optional concept data for domain-specific generation

        Returns:
            VisualizationScene ready for frontend rendering
        """
        start_time = time.time()

        viz_type = viz_params.get("type", "3d_model")
        concept_id = viz_params.get("concept_id", "")
        title = viz_params.get("title", "Visualization")

        # Route to appropriate generator
        generators = {
            "3d_model": self._generate_3d_model,
            "animation": self._generate_animation,
            "step_by_step": self._generate_step_by_step,
            "contextual_overlay": self._generate_overlay,
        }

        generator = generators.get(viz_type, self._generate_3d_model)
        scene = generator(viz_params, concept_id, title, concept_data)
        scene.generation_time_ms = (time.time() - start_time) * 1000

        return scene

    def _generate_3d_model(self, params: dict, concept_id: str,
                            title: str, concept_data: dict | None) -> VisualizationScene:
        """Generate a 3D model scene based on the concept."""
        # Route to domain-specific generators
        domain = params.get("domain", "")

        if concept_id.startswith("chem_"):
            return self._generate_chemistry_model(params, concept_id, title)
        elif concept_id.startswith("bio_"):
            return self._generate_biology_model(params, concept_id, title)
        elif concept_id.startswith("phys_"):
            return self._generate_physics_model(params, concept_id, title)
        elif concept_id.startswith("anat_"):
            return self._generate_anatomy_model(params, concept_id, title)
        elif concept_id.startswith("eng_"):
            return self._generate_engineering_model(params, concept_id, title)
        elif concept_id.startswith("math_"):
            return self._generate_math_model(params, concept_id, title)

        return self._generate_generic_model(title)

    def _generate_chemistry_model(self, params: dict, concept_id: str,
                                   title: str) -> VisualizationScene:
        """Generate chemistry-specific 3D models."""
        scene = VisualizationScene(scene_id=f"scene_{concept_id}", title=title)

        if concept_id == "chem_water":
            # Water molecule H2O with bent geometry (104.5 degrees)
            scene.objects = [
                SceneObject(id="oxygen", type="sphere", position=[0, 0, 0],
                           scale=[0.6, 0.6, 0.6], color=self.ELEMENT_COLORS["O"],
                           label="Oxygen (O)", interactive=True),
                SceneObject(id="hydrogen1", type="sphere", position=[-0.96, 0.55, 0],
                           scale=[0.35, 0.35, 0.35], color=self.ELEMENT_COLORS["H"],
                           label="Hydrogen (H)", interactive=True),
                SceneObject(id="hydrogen2", type="sphere", position=[0.96, 0.55, 0],
                           scale=[0.35, 0.35, 0.35], color=self.ELEMENT_COLORS["H"],
                           label="Hydrogen (H)", interactive=True),
                SceneObject(id="bond1", type="cylinder", position=[-0.48, 0.275, 0],
                           rotation=[0, 0, 0.52], scale=[0.08, 0.55, 0.08],
                           color="#aaaaaa", label="Covalent Bond"),
                SceneObject(id="bond2", type="cylinder", position=[0.48, 0.275, 0],
                           rotation=[0, 0, -0.52], scale=[0.08, 0.55, 0.08],
                           color="#aaaaaa", label="Covalent Bond"),
            ]
            scene.annotations = [
                {"text": "Bond angle: 104.5°", "position": [0, 1.2, 0]},
                {"text": "H₂O - Water Molecule", "position": [0, -1, 0]},
            ]

        elif concept_id == "chem_atom":
            # Bohr model atom
            scene.objects = [
                SceneObject(id="nucleus", type="sphere", position=[0, 0, 0],
                           scale=[0.5, 0.5, 0.5], color="#FF4444", label="Nucleus",
                           interactive=True),
            ]
            # Add electron orbits
            for i, (radius, electrons) in enumerate([(1.5, 2), (2.5, 4), (3.5, 2)]):
                scene.objects.append(SceneObject(
                    id=f"orbit_{i}", type="ring", position=[0, 0, 0],
                    scale=[radius, radius, radius], color="#4444FF", opacity=0.3,
                    label=f"Shell {i+1}",
                ))
                for j in range(electrons):
                    import math
                    angle = (2 * math.pi * j) / electrons
                    x = radius * math.cos(angle)
                    z = radius * math.sin(angle)
                    scene.objects.append(SceneObject(
                        id=f"electron_{i}_{j}", type="sphere",
                        position=[x, 0, z], scale=[0.15, 0.15, 0.15],
                        color="#00AAFF", label="Electron",
                        animation={"type": "orbit", "radius": radius,
                                   "speed": 1.0 / (i + 1), "axis": "y"},
                    ))

        elif concept_id in ("chem_molecule", "chem_bond"):
            # Generic molecular structure
            scene.objects = [
                SceneObject(id="atom_c", type="sphere", position=[0, 0, 0],
                           scale=[0.5, 0.5, 0.5], color=self.ELEMENT_COLORS["C"],
                           label="Carbon", interactive=True),
                SceneObject(id="atom_h1", type="sphere", position=[1.2, 0.7, 0],
                           scale=[0.3, 0.3, 0.3], color=self.ELEMENT_COLORS["H"],
                           label="Hydrogen"),
                SceneObject(id="atom_h2", type="sphere", position=[-1.2, 0.7, 0],
                           scale=[0.3, 0.3, 0.3], color=self.ELEMENT_COLORS["H"],
                           label="Hydrogen"),
                SceneObject(id="atom_h3", type="sphere", position=[0, -0.8, 1.0],
                           scale=[0.3, 0.3, 0.3], color=self.ELEMENT_COLORS["H"],
                           label="Hydrogen"),
                SceneObject(id="atom_h4", type="sphere", position=[0, -0.8, -1.0],
                           scale=[0.3, 0.3, 0.3], color=self.ELEMENT_COLORS["H"],
                           label="Hydrogen"),
            ]
        else:
            return self._generate_generic_model(title)

        scene.camera_position = [0, 2, 6]
        return scene

    def _generate_biology_model(self, params: dict, concept_id: str,
                                 title: str) -> VisualizationScene:
        """Generate biology-specific 3D models."""
        scene = VisualizationScene(scene_id=f"scene_{concept_id}", title=title)

        if concept_id == "bio_cell":
            scene.objects = [
                SceneObject(id="membrane", type="sphere", position=[0, 0, 0],
                           scale=[3, 3, 3], color="#FFE4B5", opacity=0.3,
                           label="Cell Membrane", interactive=True),
                SceneObject(id="nucleus", type="sphere", position=[0, 0, 0],
                           scale=[1.0, 1.0, 1.0], color="#4169E1", opacity=0.7,
                           label="Nucleus", interactive=True),
                SceneObject(id="nucleolus", type="sphere", position=[0.2, 0.2, 0],
                           scale=[0.3, 0.3, 0.3], color="#1a1aff",
                           label="Nucleolus"),
                SceneObject(id="mito1", type="capsule", position=[1.5, 0.5, 0],
                           scale=[0.4, 0.2, 0.2], color="#FF6347",
                           label="Mitochondria", interactive=True),
                SceneObject(id="mito2", type="capsule", position=[-1.2, -0.8, 0.5],
                           scale=[0.35, 0.18, 0.18], color="#FF6347",
                           label="Mitochondria"),
                SceneObject(id="er", type="custom", position=[0.8, -0.5, 0],
                           scale=[1.0, 0.5, 0.5], color="#98FB98", opacity=0.5,
                           label="Endoplasmic Reticulum"),
                SceneObject(id="golgi", type="custom", position=[-1.0, 0.8, 0],
                           scale=[0.5, 0.3, 0.3], color="#DDA0DD",
                           label="Golgi Apparatus", interactive=True),
                SceneObject(id="ribosome1", type="sphere", position=[1.8, -0.3, 0.5],
                           scale=[0.08, 0.08, 0.08], color="#8B4513",
                           label="Ribosome"),
                SceneObject(id="ribosome2", type="sphere", position=[1.5, -0.5, -0.3],
                           scale=[0.08, 0.08, 0.08], color="#8B4513",
                           label="Ribosome"),
            ]
            scene.camera_position = [0, 3, 8]

        elif concept_id == "bio_dna":
            # DNA double helix
            import math
            objects = []
            for i in range(40):
                t = i * 0.3
                # Strand 1
                x1 = math.cos(t) * 0.8
                z1 = math.sin(t) * 0.8
                y1 = t * 0.3 - 3
                objects.append(SceneObject(
                    id=f"strand1_{i}", type="sphere",
                    position=[x1, y1, z1], scale=[0.12, 0.12, 0.12],
                    color="#FF4444",
                ))
                # Strand 2
                x2 = math.cos(t + math.pi) * 0.8
                z2 = math.sin(t + math.pi) * 0.8
                objects.append(SceneObject(
                    id=f"strand2_{i}", type="sphere",
                    position=[x2, y1, z2], scale=[0.12, 0.12, 0.12],
                    color="#4444FF",
                ))
                # Base pair connections
                if i % 3 == 0:
                    objects.append(SceneObject(
                        id=f"basepair_{i}", type="cylinder",
                        position=[(x1+x2)/2, y1, (z1+z2)/2],
                        scale=[0.05, 0.7, 0.05], color="#44FF44", opacity=0.6,
                        label="Base Pair" if i == 0 else "",
                    ))
            scene.objects = objects
            scene.camera_position = [0, 0, 6]
            scene.annotations = [{"text": "DNA Double Helix", "position": [0, 4, 0]}]

        elif concept_id == "bio_mitochondria":
            scene.objects = [
                SceneObject(id="outer_membrane", type="capsule", position=[0, 0, 0],
                           scale=[2.0, 1.0, 1.0], color="#FF6347", opacity=0.4,
                           label="Outer Membrane", interactive=True),
                SceneObject(id="inner_membrane", type="capsule", position=[0, 0, 0],
                           scale=[1.7, 0.8, 0.8], color="#FF4500", opacity=0.5,
                           label="Inner Membrane"),
                SceneObject(id="cristae1", type="custom", position=[0.3, 0, 0],
                           scale=[0.3, 0.6, 0.6], color="#FF6347",
                           label="Cristae"),
                SceneObject(id="matrix", type="capsule", position=[0, 0, 0],
                           scale=[1.4, 0.6, 0.6], color="#FFB6C1", opacity=0.3,
                           label="Matrix"),
            ]
            scene.camera_position = [0, 2, 5]
        else:
            return self._generate_generic_model(title)

        return scene

    def _generate_physics_model(self, params: dict, concept_id: str,
                                 title: str) -> VisualizationScene:
        """Generate physics-specific 3D models."""
        scene = VisualizationScene(scene_id=f"scene_{concept_id}", title=title)

        if concept_id == "phys_solar_system":
            import math
            # Sun
            scene.objects = [
                SceneObject(id="sun", type="sphere", position=[0, 0, 0],
                           scale=[1.5, 1.5, 1.5], color="#FFD700",
                           label="Sun", interactive=True,
                           metadata={"emissive": True}),
            ]
            planets = [
                ("Mercury", 2.5, 0.15, "#A0522D", 4.15),
                ("Venus", 3.5, 0.25, "#DEB887", 1.62),
                ("Earth", 4.5, 0.27, "#4169E1", 1.0),
                ("Mars", 5.5, 0.2, "#CD5C5C", 0.53),
                ("Jupiter", 7.5, 0.8, "#DAA520", 0.084),
                ("Saturn", 9.5, 0.7, "#F4A460", 0.034),
                ("Uranus", 11.5, 0.5, "#87CEEB", 0.012),
                ("Neptune", 13.5, 0.45, "#4682B4", 0.006),
            ]
            for name, dist, size, color, speed in planets:
                angle = hash(name) % 360 * math.pi / 180
                x = dist * math.cos(angle)
                z = dist * math.sin(angle)
                scene.objects.append(SceneObject(
                    id=name.lower(), type="sphere",
                    position=[x, 0, z], scale=[size, size, size],
                    color=color, label=name, interactive=True,
                    animation={"type": "orbit", "radius": dist,
                               "speed": speed, "axis": "y"},
                ))
                # Orbit ring
                scene.objects.append(SceneObject(
                    id=f"{name.lower()}_orbit", type="ring",
                    position=[0, 0, 0], scale=[dist, dist, dist],
                    color="#333333", opacity=0.2,
                ))
            scene.camera_position = [5, 15, 20]
            scene.camera_target = [0, 0, 0]

        elif concept_id == "phys_newton_laws":
            scene.objects = [
                SceneObject(id="block", type="box", position=[0, 0.5, 0],
                           scale=[1, 1, 1], color="#4169E1", label="Object",
                           interactive=True),
                SceneObject(id="surface", type="box", position=[0, -0.1, 0],
                           scale=[8, 0.2, 3], color="#8B8B8B", label="Surface"),
                SceneObject(id="force_arrow", type="arrow",
                           position=[1.0, 0.5, 0], rotation=[0, 0, -1.57],
                           scale=[0.1, 2, 0.1], color="#FF0000",
                           label="Applied Force (F)", interactive=True),
                SceneObject(id="friction_arrow", type="arrow",
                           position=[-1.0, 0.5, 0], rotation=[0, 0, 1.57],
                           scale=[0.1, 1.2, 0.1], color="#FFA500",
                           label="Friction"),
            ]
            scene.steps = [
                {"step": 1, "title": "First Law - Inertia",
                 "description": "Object at rest stays at rest unless acted upon by a force",
                 "highlight": ["block"]},
                {"step": 2, "title": "Second Law - F=ma",
                 "description": "Force equals mass times acceleration",
                 "highlight": ["force_arrow", "block"]},
                {"step": 3, "title": "Third Law - Action/Reaction",
                 "description": "Every action has an equal and opposite reaction",
                 "highlight": ["force_arrow", "friction_arrow"]},
            ]
            scene.camera_position = [3, 4, 6]
        else:
            return self._generate_generic_model(title)

        return scene

    def _generate_anatomy_model(self, params: dict, concept_id: str,
                                 title: str) -> VisualizationScene:
        """Generate anatomy-specific 3D models."""
        scene = VisualizationScene(scene_id=f"scene_{concept_id}", title=title)

        if concept_id == "anat_heart":
            scene.objects = [
                SceneObject(id="heart_body", type="custom", position=[0, 0, 0],
                           scale=[1.5, 1.5, 1.5], color="#CC0000", opacity=0.8,
                           label="Heart", interactive=True,
                           metadata={"model": "heart"}),
                SceneObject(id="right_atrium", type="sphere", position=[0.6, 0.8, 0],
                           scale=[0.5, 0.5, 0.4], color="#8B0000", opacity=0.7,
                           label="Right Atrium", interactive=True),
                SceneObject(id="left_atrium", type="sphere", position=[-0.6, 0.8, 0],
                           scale=[0.5, 0.5, 0.4], color="#FF0000", opacity=0.7,
                           label="Left Atrium", interactive=True),
                SceneObject(id="right_ventricle", type="sphere", position=[0.4, -0.3, 0],
                           scale=[0.6, 0.7, 0.5], color="#8B0000", opacity=0.7,
                           label="Right Ventricle", interactive=True),
                SceneObject(id="left_ventricle", type="sphere", position=[-0.4, -0.3, 0],
                           scale=[0.6, 0.8, 0.5], color="#FF0000", opacity=0.7,
                           label="Left Ventricle", interactive=True),
                SceneObject(id="aorta", type="cylinder", position=[-0.2, 1.5, 0],
                           rotation=[0.2, 0, 0.3], scale=[0.2, 1.0, 0.2],
                           color="#FF4444", label="Aorta"),
                SceneObject(id="pulmonary", type="cylinder", position=[0.3, 1.3, 0],
                           rotation=[-0.2, 0, -0.3], scale=[0.18, 0.8, 0.18],
                           color="#4444FF", label="Pulmonary Artery"),
            ]
            scene.annotations = [
                {"text": "Deoxygenated blood enters right atrium", "position": [1.5, 1.0, 0]},
                {"text": "Oxygenated blood exits via aorta", "position": [-1.5, 1.8, 0]},
            ]
            scene.camera_position = [0, 1, 5]

        elif concept_id == "anat_brain":
            scene.objects = [
                SceneObject(id="cerebrum", type="sphere", position=[0, 0.5, 0],
                           scale=[2.0, 1.5, 1.8], color="#FFB6C1", opacity=0.7,
                           label="Cerebrum", interactive=True),
                SceneObject(id="cerebellum", type="sphere", position=[0, -0.8, -0.5],
                           scale=[1.0, 0.7, 0.8], color="#DDA0DD", opacity=0.7,
                           label="Cerebellum", interactive=True),
                SceneObject(id="brainstem", type="cylinder", position=[0, -1.2, 0],
                           scale=[0.3, 1.0, 0.3], color="#CD853F",
                           label="Brain Stem", interactive=True),
                SceneObject(id="frontal_lobe", type="custom", position=[0, 0.8, 0.8],
                           scale=[0.8, 0.6, 0.5], color="#FF6347", opacity=0.4,
                           label="Frontal Lobe"),
                SceneObject(id="temporal_lobe", type="custom", position=[1.0, 0, 0],
                           scale=[0.5, 0.5, 0.7], color="#4169E1", opacity=0.4,
                           label="Temporal Lobe"),
            ]
            scene.camera_position = [0, 1, 6]

        elif concept_id == "anat_skeleton":
            scene.objects = [
                SceneObject(id="skull", type="sphere", position=[0, 4, 0],
                           scale=[0.6, 0.7, 0.6], color="#FFFFF0",
                           label="Skull", interactive=True),
                SceneObject(id="spine", type="cylinder", position=[0, 2, 0],
                           scale=[0.15, 2.5, 0.15], color="#FFFFF0",
                           label="Spine"),
                SceneObject(id="ribcage", type="custom", position=[0, 2.5, 0],
                           scale=[0.8, 1.0, 0.5], color="#FFFFF0", opacity=0.6,
                           label="Ribcage", interactive=True),
                SceneObject(id="pelvis", type="custom", position=[0, 0.8, 0],
                           scale=[0.7, 0.4, 0.4], color="#FFFFF0",
                           label="Pelvis"),
                SceneObject(id="femur_l", type="cylinder", position=[-0.3, -0.5, 0],
                           scale=[0.1, 1.5, 0.1], color="#FFFFF0",
                           label="Femur", interactive=True),
                SceneObject(id="femur_r", type="cylinder", position=[0.3, -0.5, 0],
                           scale=[0.1, 1.5, 0.1], color="#FFFFF0",
                           label="Femur"),
            ]
            scene.camera_position = [0, 2, 8]
        else:
            return self._generate_generic_model(title)

        return scene

    def _generate_engineering_model(self, params: dict, concept_id: str,
                                     title: str) -> VisualizationScene:
        """Generate engineering-specific 3D models."""
        scene = VisualizationScene(scene_id=f"scene_{concept_id}", title=title)

        if concept_id == "eng_gear":
            import math
            objects = []
            # Two meshing gears
            for gear_idx, (cx, cz, teeth, color) in enumerate(
                [(-1.2, 0, 12, "#4169E1"), (1.2, 0, 16, "#FF6347")]
            ):
                objects.append(SceneObject(
                    id=f"gear_{gear_idx}_hub", type="cylinder",
                    position=[cx, 0, cz], scale=[0.2, 0.3, 0.2],
                    color=color, label=f"Gear {gear_idx+1} ({teeth}T)",
                    interactive=True,
                    animation={"type": "rotate", "axis": "y",
                               "speed": 1.0 if gear_idx == 0 else -0.75},
                ))
                radius = teeth * 0.08
                objects.append(SceneObject(
                    id=f"gear_{gear_idx}_body", type="cylinder",
                    position=[cx, 0, cz], scale=[radius, 0.2, radius],
                    color=color, opacity=0.7,
                    animation={"type": "rotate", "axis": "y",
                               "speed": 1.0 if gear_idx == 0 else -0.75},
                ))
            scene.objects = objects
            scene.camera_position = [0, 4, 5]

        elif concept_id == "eng_circuit":
            scene.objects = [
                SceneObject(id="battery", type="box", position=[-3, 0, 0],
                           scale=[0.5, 1, 0.3], color="#FFD700",
                           label="Battery (V)", interactive=True),
                SceneObject(id="resistor", type="box", position=[0, 2, 0],
                           scale=[1.5, 0.3, 0.3], color="#8B4513",
                           label="Resistor (R)", interactive=True),
                SceneObject(id="led", type="sphere", position=[3, 0, 0],
                           scale=[0.3, 0.3, 0.3], color="#FF0000",
                           label="LED", interactive=True,
                           metadata={"emissive": True}),
                # Wires
                SceneObject(id="wire1", type="cylinder", position=[-3, 1, 0],
                           scale=[0.05, 1, 0.05], color="#FFD700"),
                SceneObject(id="wire2", type="cylinder", position=[-1.5, 2, 0],
                           rotation=[0, 0, 1.57], scale=[0.05, 1.5, 0.05],
                           color="#FFD700"),
                SceneObject(id="wire3", type="cylinder", position=[3, 1, 0],
                           scale=[0.05, 1, 0.05], color="#FFD700"),
                SceneObject(id="wire4", type="cylinder", position=[1.5, 2, 0],
                           rotation=[0, 0, 1.57], scale=[0.05, 1.5, 0.05],
                           color="#FFD700"),
            ]
            scene.annotations = [
                {"text": "V = IR (Ohm's Law)", "position": [0, 3, 0]},
                {"text": "Current flows from + to -", "position": [0, -1, 0]},
            ]
            scene.camera_position = [0, 1, 8]
        else:
            return self._generate_generic_model(title)

        return scene

    def _generate_math_model(self, params: dict, concept_id: str,
                              title: str) -> VisualizationScene:
        """Generate math-specific visualizations."""
        scene = VisualizationScene(scene_id=f"scene_{concept_id}", title=title)

        if concept_id == "math_pythagorean":
            scene.objects = [
                # Right triangle
                SceneObject(id="side_a", type="box", position=[0, 0, 0],
                           scale=[3, 0.1, 0.1], color="#FF4444", label="a = 3"),
                SceneObject(id="side_b", type="box", position=[1.5, 2, 0],
                           scale=[0.1, 4, 0.1], color="#4444FF", label="b = 4"),
                SceneObject(id="side_c", type="box", position=[0, 2, 0],
                           rotation=[0, 0, 0.93], scale=[5, 0.1, 0.1],
                           color="#44FF44", label="c = 5"),
                # Squares on each side
                SceneObject(id="sq_a", type="box", position=[0, -1.5, -0.1],
                           scale=[3, 3, 0.05], color="#FF4444", opacity=0.3,
                           label="a² = 9"),
                SceneObject(id="sq_b", type="box", position=[3.5, 2, -0.1],
                           scale=[4, 4, 0.05], color="#4444FF", opacity=0.3,
                           label="b² = 16"),
                SceneObject(id="right_angle", type="box", position=[1.35, 0.15, 0],
                           scale=[0.3, 0.3, 0.05], color="#ffffff", opacity=0.5),
            ]
            scene.annotations = [
                {"text": "a² + b² = c²", "position": [0, 5, 0]},
                {"text": "9 + 16 = 25", "position": [0, 4.5, 0]},
            ]
            scene.camera_position = [2, 2, 10]
        else:
            return self._generate_generic_model(title)

        return scene

    def _generate_animation(self, params: dict, concept_id: str,
                             title: str, concept_data: dict | None) -> VisualizationScene:
        """Generate animated visualization."""
        scene = self._generate_3d_model(params, concept_id, title, concept_data)
        # Add animation timeline
        scene.animation_timeline = [
            {"time": 0, "action": "fade_in", "targets": "all", "duration": 1000},
            {"time": 1000, "action": "highlight", "targets": "interactive", "duration": 2000},
            {"time": 3000, "action": "rotate", "targets": "all", "duration": 5000,
             "params": {"axis": "y", "angle": 360}},
        ]
        return scene

    def _generate_step_by_step(self, params: dict, concept_id: str,
                                title: str, concept_data: dict | None) -> VisualizationScene:
        """Generate step-by-step visual walkthrough."""
        scene = self._generate_3d_model(params, concept_id, title, concept_data)
        if not scene.steps:
            scene.steps = [
                {"step": 1, "title": "Overview", "description": f"Introduction to {title}",
                 "highlight": []},
                {"step": 2, "title": "Key Components", "description": "Identifying main parts",
                 "highlight": []},
                {"step": 3, "title": "Interactions", "description": "How components work together",
                 "highlight": []},
                {"step": 4, "title": "Summary", "description": "Key takeaways",
                 "highlight": []},
            ]
        return scene

    def _generate_overlay(self, params: dict, concept_id: str,
                           title: str, concept_data: dict | None) -> VisualizationScene:
        """Generate contextual overlay visualization."""
        scene = VisualizationScene(scene_id=f"scene_{concept_id}", title=title)
        scene.objects = [
            SceneObject(id="info_panel", type="box", position=[0, 1.5, 0],
                       scale=[4, 2.5, 0.05], color="#000033", opacity=0.8,
                       label=title, metadata={"type": "info_panel"}),
        ]
        scene.annotations = [
            {"text": title, "position": [0, 2.5, 0.1], "size": "large"},
            {"text": "Tap for more details", "position": [0, 0.5, 0.1], "size": "small"},
        ]
        return scene

    def _generate_generic_model(self, title: str) -> VisualizationScene:
        """Generate a generic placeholder model."""
        scene = VisualizationScene(scene_id=f"scene_generic", title=title)
        scene.objects = [
            SceneObject(id="placeholder", type="sphere", position=[0, 0, 0],
                       scale=[1, 1, 1], color="#4169E1", label=title,
                       interactive=True,
                       animation={"type": "rotate", "axis": "y", "speed": 0.5}),
        ]
        scene.annotations = [
            {"text": title, "position": [0, 2, 0]},
        ]
        scene.camera_position = [0, 2, 5]
        return scene
