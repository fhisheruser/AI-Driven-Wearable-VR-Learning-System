"""Tests for the visualization engine."""

import pytest
from src.visualization.visualization_engine import (
    VisualizationEngine, VisualizationScene, SceneObject,
)


@pytest.fixture
def engine():
    return VisualizationEngine()


class TestVisualizationEngine:

    def test_generate_chemistry_scene(self, engine):
        params = {"type": "3d_model", "concept_id": "chem_water", "title": "Water",
                  "domain": "chemistry"}
        scene = engine.generate_scene(params)
        assert isinstance(scene, VisualizationScene)
        assert len(scene.objects) > 0
        assert scene.title == "Water"

    def test_generate_biology_scene(self, engine):
        params = {"type": "3d_model", "concept_id": "bio_cell", "title": "Cell",
                  "domain": "biology"}
        scene = engine.generate_scene(params)
        assert len(scene.objects) > 0

    def test_generate_physics_scene(self, engine):
        params = {"type": "3d_model", "concept_id": "phys_solar_system",
                  "title": "Solar System", "domain": "physics"}
        scene = engine.generate_scene(params)
        # Sun + 8 planets + 8 orbits
        assert len(scene.objects) >= 9

    def test_generate_anatomy_scene(self, engine):
        params = {"type": "3d_model", "concept_id": "anat_heart",
                  "title": "Heart", "domain": "anatomy"}
        scene = engine.generate_scene(params)
        assert len(scene.objects) > 0

    def test_generate_math_scene(self, engine):
        params = {"type": "3d_model", "concept_id": "math_pythagorean",
                  "title": "Pythagorean Theorem", "domain": "mathematics"}
        scene = engine.generate_scene(params)
        assert len(scene.objects) > 0
        assert any("a" in obj.label or "b" in obj.label for obj in scene.objects)

    def test_generate_animation(self, engine):
        params = {"type": "animation", "concept_id": "chem_atom",
                  "title": "Atom", "domain": "chemistry"}
        scene = engine.generate_scene(params)
        assert len(scene.animation_timeline) > 0

    def test_generate_step_by_step(self, engine):
        params = {"type": "step_by_step", "concept_id": "phys_newton_laws",
                  "title": "Newton's Laws", "domain": "physics"}
        scene = engine.generate_scene(params)
        assert len(scene.steps) > 0

    def test_generate_overlay(self, engine):
        params = {"type": "contextual_overlay", "concept_id": "unknown",
                  "title": "Info"}
        scene = engine.generate_scene(params)
        assert scene.title == "Info"

    def test_generic_fallback(self, engine):
        params = {"type": "3d_model", "concept_id": "unknown_concept",
                  "title": "Unknown"}
        scene = engine.generate_scene(params)
        assert len(scene.objects) >= 1

    def test_scene_to_dict(self, engine):
        params = {"type": "3d_model", "concept_id": "bio_dna",
                  "title": "DNA", "domain": "biology"}
        scene = engine.generate_scene(params)
        d = scene.to_dict()
        assert "objects" in d
        assert "camera_position" in d
        assert "scene_id" in d

    def test_latency_tracking(self, engine):
        params = {"type": "3d_model", "concept_id": "chem_water",
                  "title": "Water", "domain": "chemistry"}
        scene = engine.generate_scene(params)
        assert scene.generation_time_ms >= 0
