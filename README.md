# AI-Driven Wearable Virtual Reality Learning System

An AI-powered smart educational glasses platform that integrates **conversational AI**, **adaptive learning**, and **immersive 3D visualization** for personalized education. The system enables students to interact via natural language and receive real-time, interactive 3D visualizations of educational content projected into their field of view.

> **Research Paper:** *"An AI-Driven Wearable Virtual Reality Learning System for Enhanced Visualization and Personalized Education"*
> **Author:** Leena Rajan Katkar, Department of Computer Science, University of Southern California

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Modules in Detail](#modules-in-detail)
  - [ASR Module](#1-asr-module---speech-recognition)
  - [NLU Module](#2-nlu-module---natural-language-understanding)
  - [Knowledge Engine](#3-knowledge-engine)
  - [Personalization Engine](#4-personalization-engine-ppo-based-adaptive-learning)
  - [Visualization Engine](#5-visualization-engine)
  - [Feedback Module](#6-feedback-module)
  - [Evaluation Module](#7-evaluation-module)
  - [API Server](#8-api-server)
  - [Frontend](#9-frontend---threejs-3d-visualization)
- [Supported Educational Domains](#supported-educational-domains)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Configuration](#configuration)
- [System Workflow](#system-workflow)
- [Evaluation Framework](#evaluation-framework)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

Traditional learning tools such as textbooks, slides, and 2D videos are limited in their ability to facilitate visualization and real-time interaction. This system addresses that gap by combining:

- **Speech Recognition (ASR):** Whisper-based deep learning ASR with noise reduction and educational vocabulary optimization
- **Natural Language Understanding (NLU):** Transformer-based intent classification and semantic parsing of learner queries
- **Knowledge Reasoning:** Domain ontology with 40+ educational concepts aligned to standard curricula
- **Reinforcement Learning Personalization:** PPO-based adaptive engine that dynamically adjusts content depth, pacing, and complexity
- **Immersive Visualization:** Real-time 3D scene generation for mixed-reality display on smart glasses
- **Continuous Feedback Loop:** Implicit and explicit feedback collection driving iterative learning improvement

The system targets an end-to-end response latency of **under 2 seconds** and ASR accuracy of **>= 95%** for seamless immersive learning.

---

## Key Features

- **Natural Language Interaction** - Students ask questions using natural speech; the system understands intent, domain, and context
- **6 Educational Domains** - Chemistry, Biology, Physics, Mathematics, Anatomy, and Engineering with 40+ fully modeled concepts
- **Adaptive Personalization** - PPO reinforcement learning adjusts difficulty, content depth, visualization complexity, and pacing per learner
- **3D Visualization Generation** - Domain-specific 3D scenes (molecules, organs, solar system, circuits, etc.) with animations and step-by-step walkthroughs
- **Real-Time WebSocket Streaming** - Audio streaming and real-time response via WebSocket for smart glasses integration
- **Learner Profiles** - Persistent profiles tracking comprehension, engagement, domain progress, and interaction history
- **Evaluation Framework** - Built-in metrics for ASR accuracy (WER), latency, learning effectiveness (paired t-test), and cognitive load (NASA-TLX)
- **Interactive Web Frontend** - Three.js-based 3D visualization with drag-to-rotate, zoom, hover labels, and step-by-step navigation

---

## System Architecture

```
                          +---------------------------+
                          |     AI SMART GLASSES       |
                          |  Microphone  |  Camera     |
                          |  Display Lenses            |
                          +------------++--------------+
                                       ||
                                       vv
+------------------------------------------------------------------------------+
|                           AI PROCESSING PIPELINE                             |
|                                                                              |
|  +----------+    +-----------+    +------------+    +----------------+       |
|  |   ASR    |--->|    NLU    |--->| Knowledge  |--->| Visualization  |       |
|  | (Whisper)|    | (Semantic |    |  Reasoning |    |    Engine      |       |
|  |          |    |  Analysis)|    |  Engine    |    | (3D Scenes)    |       |
|  +----------+    +-----------+    +------------+    +----------------+       |
|       |                                                      |               |
|       |          +----------------+                          |               |
|       +--------->| Personalization|<-------------------------+               |
|                  | (PPO Adaptive  |                                          |
|                  |  Engine)       |                                          |
|                  +-------+--------+                                          |
|                          |                                                   |
|                  +-------v--------+                                          |
|                  |   Feedback     |                                          |
|                  |  Collector     |                                          |
|                  +----------------+                                          |
+------------------------------------------------------------------------------+
                                       ||
                                       vv
                          +---------------------------+
                          |   FastAPI Server           |
                          |   REST + WebSocket         |
                          |   Three.js Frontend        |
                          +---------------------------+
```

---

## Project Structure

```
AI-Driven-Wearable-VR-Learning-System/
|
|-- main.py                          # Entry point (server + demo mode)
|-- setup.py                         # Package configuration
|-- requirements.txt                 # Python dependencies
|-- .env.example                     # Environment variable template
|
|-- configs/
|   +-- system_config.yaml           # Full system configuration (ASR, NLU, PPO, etc.)
|
|-- src/
|   |-- __init__.py
|   |-- pipeline.py                  # Main pipeline orchestrator
|   |
|   |-- asr/                         # Speech Recognition
|   |   |-- __init__.py
|   |   |-- speech_recognizer.py     # Whisper-based ASR with noise reduction
|   |   +-- voice_activity_detector.py  # Energy-based VAD with calibration
|   |
|   |-- nlu/                         # Natural Language Understanding
|   |   |-- __init__.py
|   |   |-- intent_classifier.py     # Pattern + embedding-based intent classification
|   |   +-- semantic_analyzer.py     # Full semantic analysis pipeline
|   |
|   |-- knowledge/                   # Knowledge Reasoning
|   |   |-- __init__.py
|   |   |-- domain_ontology.py       # 40+ concepts across 6 domains
|   |   +-- knowledge_engine.py      # Adaptive reasoning and content generation
|   |
|   |-- personalization/             # Adaptive Learning
|   |   |-- __init__.py
|   |   |-- learner_profile.py       # Profile management and persistence
|   |   +-- adaptive_engine.py       # PPO agent with RL environment
|   |
|   |-- visualization/               # VR/AR Content Generation
|   |   |-- __init__.py
|   |   +-- visualization_engine.py  # 3D scene generation (638 lines)
|   |
|   |-- feedback/                    # Feedback Collection
|   |   |-- __init__.py
|   |   +-- feedback_collector.py    # Implicit/explicit feedback processing
|   |
|   |-- evaluation/                  # Evaluation Metrics
|   |   |-- __init__.py
|   |   +-- metrics.py               # ASR WER, latency, NASA-TLX, paired t-test
|   |
|   |-- api/                         # Web Server
|   |   |-- __init__.py
|   |   +-- server.py                # FastAPI with REST + WebSocket endpoints
|   |
|   +-- utils/
|       |-- __init__.py
|       +-- config.py                # Pydantic config with YAML loader
|
|-- frontend/
|   |-- templates/
|   |   +-- index.html               # Main web UI
|   +-- static/
|       |-- css/
|       |   +-- main.css             # Dark theme responsive styles
|       +-- js/
|           |-- renderer.js          # Three.js 3D scene renderer
|           +-- app.js               # Frontend application logic
|
|-- tests/
|   |-- __init__.py
|   |-- test_pipeline.py             # End-to-end pipeline tests (9 tests)
|   |-- test_semantic_analyzer.py    # NLU module tests (14 tests)
|   |-- test_knowledge_engine.py     # Knowledge engine tests (9 tests)
|   |-- test_visualization.py        # Visualization engine tests (11 tests)
|   |-- test_feedback.py             # Feedback module tests (10 tests)
|   |-- test_evaluation.py           # Evaluation metrics tests (10 tests)
|   +-- test_adaptive_engine.py      # PPO/RL engine tests (7 tests)
|
+-- data/
    |-- knowledge_base/              # Domain knowledge data (populated at runtime)
    +-- learner_profiles/            # Learner profile JSON storage
```

---

## Modules in Detail

### 1. ASR Module - Speech Recognition

**Files:** `src/asr/speech_recognizer.py`, `src/asr/voice_activity_detector.py`

The ASR module converts spoken learner queries into text using OpenAI Whisper with educational optimizations:

- **Voice Activity Detection (VAD):** Energy-based detection with zero-crossing rate analysis and spectral flatness computation for noise/speech discrimination. Includes adaptive calibration for classroom environments.
- **Noise Reduction:** Spectral subtraction using STFT-based processing. Estimates noise profile from the first 0.5 seconds of audio and subtracts it from the speech signal.
- **Educational Vocabulary:** Domain-specific vocabulary for 6 fields (72 terms) provides context prompts to Whisper for improved recognition of scientific terminology.
- **Fallback Mode:** Gracefully degrades to demo mode when Whisper is not installed, allowing development without GPU dependencies.

**Target:** Word Error Rate (WER) < 5% in classroom environments.

### 2. NLU Module - Natural Language Understanding

**Files:** `src/nlu/intent_classifier.py`, `src/nlu/semantic_analyzer.py`

Two-stage NLU pipeline that understands what the learner is asking:

- **Intent Classification:** Dual-mode classifier supporting both regex pattern matching (50+ patterns) and sentence-transformer embeddings. Classifies queries into 6 intent categories:
  - Conceptual explanation ("What is...?", "Explain...")
  - Procedural question ("How do you...?", "Steps to...")
  - Visual representation ("Show me...", "Visualize...")
  - Elaboration ("Tell me more", "Go deeper...")
  - Quiz request ("Quiz me", "Test my knowledge")
  - Topic change ("Now tell me about...")

- **Semantic Analysis:** Extracts domain, topic, entities, visualization type, and complexity level. Maintains conversational context across interactions with topic depth tracking.

### 3. Knowledge Engine

**Files:** `src/knowledge/domain_ontology.py`, `src/knowledge/knowledge_engine.py`

Maps learner queries to structured educational content:

- **Domain Ontology:** 40+ concepts across 6 domains, each with:
  - Difficulty level (1-5)
  - Prerequisites and related concepts
  - Visualization hints
  - Curriculum level alignment
  - Keywords and detailed explanations
  - Procedural steps (where applicable)

- **Reasoning Engine:**
  - Determines content depth (overview / standard / detailed / expert) based on learner profile
  - Selects visualization type and generates domain-specific rendering parameters
  - Builds pedagogical sequences for structured content delivery
  - Generates quiz questions with multiple-choice options and difficulty scaling

### 4. Personalization Engine (PPO-Based Adaptive Learning)

**Files:** `src/personalization/adaptive_engine.py`, `src/personalization/learner_profile.py`

Reinforcement learning-based personalization using Proximal Policy Optimization (PPO):

- **RL Environment:**
  - 7-dimensional state space: comprehension, engagement, difficulty, time-on-task, quiz score, interaction count, follow-up rate
  - 5-dimensional continuous action space
  - Reward function combining comprehension improvement, engagement signals, and quiz performance

- **PPO Agent:**
  - 2-layer MLP policy and value networks (numpy-based, no PyTorch dependency)
  - Generalized Advantage Estimation (GAE) for variance reduction
  - Clipped surrogate objective with configurable clip range
  - Online training from learner interactions

- **Learner Profiles:**
  - Persistent JSON-based profiles tracking per-domain progress
  - Exponential moving average for comprehension updates
  - Dynamic skill level adjustment (1-5)
  - Interaction history and engagement metrics

### 5. Visualization Engine

**File:** `src/visualization/visualization_engine.py` (638 lines)

Generates 3D scene descriptions for the Three.js frontend renderer:

| Domain | Visualizations |
|--------|---------------|
| **Chemistry** | Water molecule (H2O bent geometry, 104.5 degrees), Bohr atom model (orbiting electrons), methane (CH4), generic molecules with ball-and-stick rendering |
| **Biology** | Cell cross-section (9 organelles), DNA double helix (40-point helix with base pairs), mitochondria (outer/inner membrane, cristae) |
| **Physics** | Solar system (Sun + 8 planets with orbital animation), Newton's laws (force/friction vectors with step-by-step walkthrough) |
| **Mathematics** | Pythagorean theorem (right triangle with squares on each side, a^2 + b^2 = c^2 annotation) |
| **Anatomy** | Heart (4 chambers, aorta, pulmonary artery with blood flow annotations), Brain (cerebrum, cerebellum, brainstem, lobes), Skeleton (skull, spine, ribcage, femur) |
| **Engineering** | Gear system (two meshing gears with rotation animation), Electric circuit (battery, resistor, LED with Ohm's Law overlay) |

Each scene includes: positioned 3D objects, material properties (color, opacity, emissive), animations (orbit, rotate, pulse), labels, annotations, camera positioning, and lighting configuration.

### 6. Feedback Module

**File:** `src/feedback/feedback_collector.py`

Closed-loop feedback system for continuous adaptation:

- **Implicit Feedback:** Inferred from interaction time, gaze duration, and follow-up query patterns
- **Explicit Feedback:** Star ratings (1-5), verbal feedback, quiz scores
- **Engagement Inference:** Combines multiple behavioral signals into a normalized engagement score
- **Session Summaries:** Aggregated statistics for monitoring learning effectiveness

### 7. Evaluation Module

**File:** `src/evaluation/metrics.py`

Implements the paper's evaluation framework:

| Metric | Implementation | Target |
|--------|---------------|--------|
| ASR Accuracy | Word Error Rate via edit distance alignment | >= 95% |
| Response Latency | Per-request timing with mean and P95 | <= 2 seconds |
| Visualization FPS | Frame rate monitoring | >= 30 fps |
| Learning Effectiveness | Pre/post test score comparison | Significant gain |
| Statistical Testing | Paired t-test at 95% confidence level | p < 0.05 |
| Cognitive Load | NASA Task Load Index (NASA-TLX) | Minimized |
| Engagement | Interaction duration, query frequency, gaze tracking | Maximized |

### 8. API Server

**File:** `src/api/server.py`

FastAPI server with full REST and WebSocket support:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend web UI |
| `/api/health` | GET | Health check and pipeline status |
| `/api/query` | POST | Process text query through full pipeline |
| `/api/feedback` | POST | Submit learner feedback |
| `/api/learner/{id}` | GET | Get learner profile and statistics |
| `/api/session/reset` | POST | Reset learning session |
| `/api/domains` | GET | List available domains and concepts |
| `/ws/{learner_id}` | WebSocket | Real-time audio streaming and interaction |

### 9. Frontend - Three.js 3D Visualization

**Files:** `frontend/templates/index.html`, `frontend/static/js/renderer.js`, `frontend/static/js/app.js`, `frontend/static/css/main.css`

Interactive web-based 3D visualization interface:

- **3D Renderer:** Three.js-based renderer with:
  - Multiple geometry types (sphere, box, cylinder, capsule, ring, arrow)
  - Material properties (color, opacity, emissive glow, transparency)
  - Orbit, rotation, and pulse animations
  - Mouse-drag camera rotation and scroll-to-zoom
  - Raycasting-based hover highlighting on interactive objects
  - Canvas-texture sprite labels
  - Auto-rotate mode

- **UI Features:**
  - Chat interface with query history
  - Quick-query chips for common questions
  - Real-time explanation panel
  - Learning path / pedagogical sequence display
  - Related topics navigation
  - Personalization metrics (difficulty, depth, pacing, latency)
  - Star-rating feedback system
  - Step-by-step navigation for procedural content
  - Fullscreen mode
  - Dark theme responsive design

---

## Supported Educational Domains

| Domain | Concepts | Example Queries |
|--------|----------|----------------|
| **Chemistry** | Atom, Molecule, Bond, Electron, Water, Periodic Table, Reaction | "Show me a water molecule", "Explain covalent bonds" |
| **Biology** | Cell, DNA, Mitochondria, Photosynthesis, Nucleus | "Show me a DNA double helix", "Explain the cell structure" |
| **Physics** | Solar System, Gravity, Waves, Electromagnetic, Newton's Laws | "How does the solar system work?", "Explain Newton's laws" |
| **Mathematics** | Pythagorean Theorem, Derivative, Function, Integral | "What is the Pythagorean theorem?", "Explain derivatives" |
| **Anatomy** | Heart, Brain, Skeleton, Lungs | "Explain the human heart", "Show me the brain structure" |
| **Engineering** | Gear System, Electric Circuit, Bridge | "How do gears work?", "Show me an electric circuit" |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/fhisheruser/AI-Driven-Wearable-VR-Learning-System.git
cd AI-Driven-Wearable-VR-Learning-System

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
```

### Minimal Installation (without heavy ML models)

The system works in fallback mode without Whisper and sentence-transformers, using pattern-based classification:

```bash
pip install fastapi uvicorn pydantic pyyaml numpy
```

---

## Usage

### Start the Web Server

```bash
python main.py
```

The server starts at `http://localhost:8000`. Open it in your browser to use the interactive 3D learning interface.

### Custom Host and Port

```bash
python main.py --host 0.0.0.0 --port 8080
```

### Run Pipeline Demo

```bash
python main.py --demo
```

This runs 5 sample queries through the full pipeline and displays the results:

```
============================================================
  AI-Driven VR Learning System - Demo
============================================================

--------------------------------------------------
  Student: "How does the solar system work?"
--------------------------------------------------
  Intent:      conceptual_explanation
  Domain:      physics
  Topic:       solar system
  Depth:       standard
  Viz Type:    3d_model
  Difficulty:  1/5
  Scene:       Solar System (17 objects)
  Latency:     6.5 ms
```

### Python API Usage

```python
from src.pipeline import LearningPipeline

pipeline = LearningPipeline()
pipeline.initialize()

# Process a text query
response = pipeline.process_text("How does the solar system work?")

print(response.semantic_result.intent)        # conceptual_explanation
print(response.semantic_result.domain)         # physics
print(response.reasoned_output.explanation)    # "The Solar System consists of..."
print(len(response.scene.objects))             # 17 (sun + planets + orbits)
print(response.total_latency_ms)               # ~6ms

# Submit feedback
pipeline.process_feedback("default", {
    "rating": 5,
    "comprehension_change": 0.3,
    "engagement": 0.9,
})

# Get learner stats
stats = pipeline.get_learner_stats("default")
```

---

## API Reference

### POST `/api/query`

Process a natural language query through the full pipeline.

**Request:**
```json
{
  "text": "Show me a water molecule",
  "learner_id": "student_001"
}
```

**Response:**
```json
{
  "query_text": "Show me a water molecule",
  "semantic": {
    "intent": "visual_representation",
    "intent_confidence": 0.85,
    "domain": "chemistry",
    "topic": "water molecule",
    "entities": ["water molecule"],
    "visualization_type": "3d_model",
    "complexity": "intermediate"
  },
  "knowledge": {
    "explanation": "A molecule is an electrically neutral group...",
    "content_depth": "standard",
    "visualization_type": "3d_model",
    "difficulty_level": 1,
    "pedagogical_sequence": ["Introduction to Molecule", "Core explanation", "..."],
    "related_topics": ["Chemical Bond"],
    "steps": []
  },
  "adaptation": {
    "difficulty_adjustment": 0,
    "content_depth": "standard",
    "visualization_complexity": "moderate",
    "pacing": "moderate"
  },
  "scene": {
    "scene_id": "scene_chem_molecule",
    "title": "Molecule",
    "objects": [
      {"id": "atom_c", "type": "sphere", "position": [0,0,0], "color": "#333333", "label": "Carbon"},
      "..."
    ],
    "camera_position": [0, 2, 6],
    "annotations": []
  },
  "latency": {
    "total_ms": 2.5,
    "asr_ms": 0
  }
}
```

### POST `/api/feedback`

```json
{
  "learner_id": "student_001",
  "rating": 4,
  "comprehension_change": 0.2,
  "engagement": 0.8,
  "quiz_score": 0.9,
  "domain": "chemistry"
}
```

### WebSocket `/ws/{learner_id}`

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/student_001");

// Send text query
ws.send(JSON.stringify({ type: "text", query: "Explain DNA" }));

// Send feedback
ws.send(JSON.stringify({ type: "feedback", rating: 5 }));

// Send audio bytes (binary)
ws.send(audioBlob);
```

---

## Testing

The project includes 70 tests covering all modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_visualization.py -v
python -m pytest tests/test_semantic_analyzer.py -v
```

**Test Coverage:**

| Test File | Tests | Module |
|-----------|-------|--------|
| `test_pipeline.py` | 9 | End-to-end pipeline integration |
| `test_semantic_analyzer.py` | 14 | NLU intent/domain/entity detection |
| `test_knowledge_engine.py` | 9 | Knowledge reasoning and content generation |
| `test_visualization.py` | 11 | 3D scene generation across all domains |
| `test_feedback.py` | 10 | Feedback collection and signal processing |
| `test_evaluation.py` | 10 | ASR metrics, latency, t-test, NASA-TLX |
| `test_adaptive_engine.py` | 7 | PPO agent, RL environment, adaptation |

---

## Configuration

System configuration is managed via `configs/system_config.yaml`:

```yaml
asr:
  model_size: base              # Whisper model: tiny, base, small, medium, large
  language: en
  beam_size: 5
  word_error_rate_target: 0.05

nlu:
  model_name: distilbert-base-uncased
  intent_confidence_threshold: 0.6
  supported_intents:
    - conceptual_explanation
    - procedural_question
    - visual_representation
    - elaboration
    - quiz_request
    - topic_change

personalization:
  algorithm: PPO
  learning_rate: 0.0003
  gamma: 0.99
  clip_range: 0.2
  difficulty_levels: 5

visualization:
  target_fps: 30
  max_latency_ms: 2000
  render_quality: medium

evaluation:
  participant_count: 60
  asr_accuracy_target: 0.95
  latency_target_seconds: 2.0
  fps_target: 30
  confidence_level: 0.95
```

---

## System Workflow

The system follows the algorithm described in the research paper:

```
1. Student asks a question using natural language
   |
2. Speech Recognition (ASR) converts audio to text
   |  - Voice Activity Detection
   |  - Noise reduction (spectral subtraction)
   |  - Whisper transcription with educational vocabulary
   |
3. Semantic Analysis (NLU) processes the text
   |  - Intent classification (6 categories)
   |  - Domain detection (6 domains)
   |  - Entity extraction
   |  - Complexity assessment
   |
4. Knowledge Reasoning generates educational content
   |  - Concept lookup in domain ontology
   |  - Content depth determination
   |  - Pedagogical sequence planning
   |  - Visualization parameter generation
   |
5. Adaptive Engine personalizes the response
   |  - PPO policy maps learner state to adaptation actions
   |  - Adjusts difficulty, depth, pacing, and complexity
   |
6. Visualization Engine renders 3D content
   |  - Domain-specific scene generation
   |  - Animation timeline creation
   |  - Step-by-step walkthrough generation
   |
7. Content is projected into the student's field of view
   |  - Three.js renders the 3D scene in the browser
   |  - Mixed-reality display on smart glasses (hardware target)
   |
8. Feedback loop enables continuous adaptation
      - Implicit: gaze tracking, interaction duration, follow-up queries
      - Explicit: verbal feedback, star ratings, quiz results
      - PPO agent updates from feedback signals
```

---

## Evaluation Framework

The proposed evaluation methodology (for future prototype validation):

- **Participants:** 60 undergraduate STEM students (30 experimental, 30 control)
- **Experimental Group:** Uses the AI glasses system
- **Control Group:** Uses conventional 2D e-learning tools
- **Measurements:**
  - Pre/post test scores (standardized subject assessments)
  - Interaction duration and query frequency
  - Gaze tracking data
  - ASR accuracy (target >= 95%)
  - End-to-end latency (target <= 2 seconds)
  - Visualization frame rate (target >= 30 fps)
  - NASA Task Load Index (NASA-TLX) for cognitive load
- **Statistical Analysis:** Paired t-test with 95% confidence level

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| **Speech Recognition** | OpenAI Whisper, NumPy (spectral subtraction) |
| **NLU** | Sentence-Transformers, BERT/DistilBERT, Regex patterns |
| **Reinforcement Learning** | Custom PPO implementation (NumPy) |
| **Knowledge Base** | Domain ontology with 40+ structured concepts |
| **Web Framework** | FastAPI, Uvicorn, WebSockets |
| **3D Visualization** | Three.js (WebGL) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Configuration** | Pydantic, PyYAML |
| **Testing** | pytest (70 tests) |
| **Language** | Python 3.10+ |

---

## License

This project is developed as part of academic research at the University of Southern California.
