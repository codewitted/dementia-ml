# Tests Directory

This directory contains test files for the dementia-ml project.

## Test Structure

- `test_preprocessing.py` - Tests for data preprocessing functions
- `test_models.py` - Tests for model training and prediction
- `test_data_loading.py` - Tests for data loading utilities

## Running Tests

To run all tests:

```bash
python -m pytest tests/
```

To run a specific test file:

```bash
python -m pytest tests/test_preprocessing.py
```

To run tests with coverage:

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Data

Test data fixtures are stored in `tests/fixtures/` and include small sample datasets for validation purposes.

## Note

Tests require the `pytest` package. Install it with:

```bash
pip install pytest pytest-cov
```
