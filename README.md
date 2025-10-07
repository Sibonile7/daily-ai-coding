# 🧠 Daily AI Coding Challenges (Interview + Learning)

A structured, interview-style series of daily AI/ML coding challenges. Each day includes:
- a **prompt**
- **hints**
- a **guided solution**
- **tests** (pytest)
- a short **learning note** (+ rubric)

This repo is designed to be portfolio-ready and easy to extend.

## 🗂 Structure
```
/daily-ai-coding/
  ├── Day1_MinMaxNormalization/
  │   ├── problem.py
  │   ├── test_problem.py
  │   └── README.md
  ├── TEMPLATE_Day/                 # copy this for new days
  │   ├── problem.py
  │   ├── test_problem.py
  │   ├── README.md
  │   └── rubric.md
  ├── requirements.txt
  ├── pytest.ini
  └── .github/workflows/tests.yml   # CI
```

## 🚀 Getting Started
```bash
# (optional) create venv
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# run tests for all days
pytest -q

# run just Day 1
pytest Day1_MinMaxNormalization -q
```

## 🧩 How to add a new day
1. Copy `TEMPLATE_Day` → `DayN_YourTopic`
2. Update `README.md` with the prompt & learning notes
3. Implement `problem.py`
4. Add tests in `test_problem.py`
5. Run `pytest` locally; push — CI runs automatically

## 🏷 License
MIT — see `LICENSE`.
