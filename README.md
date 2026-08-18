# Stocki

> AI-powered stock direction prediction using intraday market data.

Stocki is a supervised binary classification system that learns from recent 5-minute OHLCV (Open, High, Low, Close, Volume) market bars to predict whether a stock's next movement will be **UP** or **DOWN**. Built by a student machine-learning team, Stocki experiments with CNN-based architectures (subject to iteration) trained on 20 days of intraday data across a curated universe of 10 tickers. The project emphasizes reproducible pipelines, version-controlled data, and rigorous evaluation against a simple baseline.

---

## Team Members

| Name | Role | GitHub |
|------|------|--------|
| Jason Kwiatkowski | Data Pipeline / Backend | [@TheRealLuro](https://github.com/TheRealLuro) |
| Nathaniel Cruz | Frontend / Dashboard | — |
| Cason Terry | Model Development / Training / Evaluation | — |

---

## Tech Stack

- **Model**: CNN (Convolutional Neural Network) — architecture subject to change
- **Language**: Python 3.11+
- **IDEs**: PyCharm (core Python), VS Code (notebooks / scripting), WebStorm (frontend / dashboard)
- **Data**: 5-minute intraday OHLCV bars
- **Version Control**: Git + GitHub

---

## Repository Structure

```
stocki/
├── backend/                # Backend API and data pipeline
├── data/                   # Dataset files
├── frontend/               # Dashboard / web interface
├── model/                  # Model definitions, training scripts, and checkpoints
├── tests/                  # Unit and integration tests
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## How to Run

> **TBD** — Setup instructions will be added once the data pipeline and model training scripts are finalized.

### Prerequisites

- Python 3.11+
- pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/TheRealLuro/stocki.git
cd stocki

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# TBD — data collection, training, and inference commands coming soon
```

---

## Deployment

- **Staging URL**: `TBD`
- **Production URL**: `TBD`

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Universe** | 10 stocks (tickers selected by team) |
| **Time Range** | Past 20 trading days |
| **Session Window** | 6 hours (regular market session) |
| **Interval** | 5-minute bars |
| **Bars per Stock** | 72 per day × 20 days = 1,440 |
| **Total Size** | ~14,400 rows across all 10 stocks |
| **Fields** | timestamp, ticker, open, high, low, close, volume |

Data collection is **complete**. Raw data, collection scripts, and cleaned versions are committed to the repository.

---

## Model

- **Task**: Supervised binary classification (UP / DOWN)
- **Input**: Window of recent 5-minute OHLCV bars
- **Output**: Direction prediction for next movement
- **Architecture**: CNN (subject to iteration)
- **Metrics**: Accuracy, Precision, Recall, F1 Score (vs. majority-class baseline)

---

## Contributing

Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for branch conventions, PR review requirements, and commit guidelines.

---

## Project Board

Track our progress on the [Stocki Project Board](https://github.com/TheRealLuro/stocki/projects) (GitHub Projects).

### Milestones

| Milestone | Status | Target |
|-----------|--------|--------|
| Data Collection | ✅ Complete | Week 1 |
| Exploration & Labeling | 🔄 In Progress | Week 2 |
| Baseline Model | ⏳ Pending | Week 3 |
| Model Iteration | ⏳ Pending | Week 4 |
| Evaluation & Report | ⏳ Pending | Week 5 |

---

*Learn from the market. Predict the move.*
