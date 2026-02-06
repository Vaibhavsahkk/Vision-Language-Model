# Cascadia Board Game Scorer

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](tests/)

A Vision-Language Model (VLM) based scoring system for the Cascadia board game. This system automatically analyzes game board images, detects animal tokens, and calculates scores using deterministic rule-based algorithms.

## Overview

This project combines Vision-Language Models with deterministic scoring logic to accurately score Cascadia board game final states from images. The system:

- Detects animal tokens (bear, elk, salmon, hawk, fox) from game board images
- Constructs hexagonal graph representations of player boards
- Applies official Cascadia scoring rules using graph algorithms
- Determines the winner and generates detailed explanations

## Architecture

The system follows a hybrid approach with clear separation of concerns:

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| Vision | VLM (GPT-4V / Gemini) | Token detection and position identification |
| Logic | Pure Python + NetworkX | Deterministic scoring calculations |
| Output | JSON + Markdown | Results, explanations, and verification |

This separation ensures that scoring is deterministic and verifiable. The VLM handles perception while Python implements the exact scoring rules.

### Design Principles

- Images are the only source of truth
- Scoring rules are fixed and deterministic
- VLMs assist vision, not scoring calculations
- Every step is verifiable and reproducible

## Project Structure

```
cascadia_scorer/
├── config/
│   └── scoring_rules.json        # Official scoring rules reference
├── src/
│   ├── models.py                 # Data classes (Tile, Board, Score)
│   ├── phase1_preprocessing.py   # Image loading and segmentation
│   ├── phase2_detection.py       # VLM-based token detection
│   ├── phase3_graph.py           # Hexagonal graph construction
│   ├── phase4_scoring.py         # Rule-based scoring engine
│   ├── phase5_winner.py          # Winner determination
│   ├── phase6_explanation.py     # Report generation
│   └── utils/
│       └── helpers.py            # Utility functions
├── tests/
│   └── test_scoring.py           # Unit tests for scoring rules
├── data/
│   ├── input/                    # Input images
│   ├── output/                   # Results and debug images
│   └── sample_detection.json     # Sample data for testing
├── main.py                       # Main pipeline entry point
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- OpenAI API key (for GPT-4 Vision) **OR** Google API key (for Gemini)

### Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/Vaibhavsahkk/Vision-Language-Model.git
cd Vision-Language-Model
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API keys**

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

**Security Note:** Never commit your `.env` file to version control.

## Usage

### Running the Full Pipeline

Process a single game board image:

```bash
python main.py --input data/input/game_board.png
```

Process separate player board images:

```bash
python main.py --input player1.png player2.png player3.png
```

Use Google Gemini instead of OpenAI:

```bash
python main.py --input game_board.png --vlm google
```

### Testing Without VLM

Use pre-defined detection data to test the scoring pipeline:

```bash
python main.py --input game_board.png --manual-data data/sample_detection.json
```

### Running Unit Tests

```bash
pytest tests/ -v
```

## Scoring Rules

The scoring rules implemented match the specific cards shown in the challenge images:

### Bear (Card B - Mother and Cubs)

Only groups of EXACTLY 3 adjacent bears score. Singles, pairs, and groups of 4+ score 0 points.

| Group Size | Points |
|------------|--------|
| 1 | 0 |
| 2 | 0 |
| 3 | 10 |
| 4+ | 0 |

### Elk (Card B)

Contiguous groups of elk score based on group size.

| Group Size | Points |
|------------|--------|
| 1 | 2 |
| 2 | 5 |
| 3 | 9 |
| 4+ | 13 |

### Salmon (Card C - Runs)

Non-branching chains of salmon score based on run length. Minimum 3 salmon to score.

| Run Length | Points |
|------------|--------|
| 1-2 | 0 |
| 3 | 10 |
| 4 | 12 |
| 5+ | 15 |

### Hawk (Card B - Isolated)

Only isolated hawks (not adjacent to other hawks) score points. Minimum 2 isolated hawks to score.

| Isolated Hawks | Points |
|----------------|--------|
| 1 | 0 |
| 2 | 5 |
| 3 | 9 |
| 4 | 12 |
| 5 | 16 |
| 6 | 20 |
| 7+ | 24 |

### Fox (Diversity)

Each fox scores points equal to the number of unique adjacent animal types (0-5 points per fox).

## Output

The pipeline generates the following outputs in `data/output/`:

| File | Description |
|------|-------------|
| score_breakdown.json | Detailed scores per player and animal type |
| game_result.json | Complete game results with winner determination |
| verification_report.md | Human-readable explanation of scoring |

## Testing

The test suite validates:

- Hexagonal grid utilities (neighbor calculation, distance)
- Individual animal scoring algorithms
- Integration of scoring components
- Edge cases and boundary conditions

Run the full test suite:

```bash
pytest tests/ -v
```

**Result:** All 25 tests pass.

## Technical Implementation

The implementation follows a 6-phase pipeline:

1. **Image Preprocessing** - Segment player boards
2. **Token Detection** - Use VLM to identify animals and positions
3. **Graph Construction** - Build hexagonal graph with NetworkX
4. **Rule-Based Scoring** - Apply deterministic scoring algorithms
5. **Winner Determination** - Calculate winner with tie-breaker logic
6. **Report Generation** - Create explanations and verification reports

**Key Insight:** VLMs excel at perception but are unreliable for complex rule-based reasoning. By extracting structured data from images and processing it with deterministic algorithms, the system achieves both accuracy and verifiability.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

Please review [SECURITY.md](SECURITY.md) for security best practices and reporting vulnerabilities.

## About Cascadia

[Cascadia](https://www.alderac.com/cascadia/) is a tile-laying and token-drafting board game published by AEG. This project is a fan-made tool for educational purposes and is not affiliated with the game's publishers.

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{cascadia_scorer,
  title = {Cascadia Board Game Scorer},
  author = {Vaibhav Sahkk},
  year = {2026},
  url = {https://github.com/Vaibhavsahkk/Vision-Language-Model}
}
```
