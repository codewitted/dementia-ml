# Contributing to Dementia-ML

Thank you for your interest in contributing to this project! This guide will help you get started.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in [GitHub Issues](https://github.com/codewitted/dementia-ml/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment details (OS, Python version, etc.)

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/dementia-ml.git
   cd dementia-ml
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   conda env create -f environment.yml
   conda activate ad-ensemble
   ```

4. **Make your changes**
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

5. **Run tests**
   ```bash
   python -m pytest tests/ -v
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

7. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing

- Add unit tests for new functions
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

### Documentation

- Update README.md if adding new features
- Add docstrings following NumPy/Google style
- Update relevant notebooks if workflow changes
- Include comments for complex logic

### Commit Messages

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: what was updated`
- `Refactor: what was refactored`
- `Docs: documentation changes`

## Project Structure

When adding new code:
- **Source code**: Place in `src/`
- **Scripts**: Place in `scripts/`
- **Tests**: Place in `tests/`
- **Notebooks**: Place in `notebooks/`

## Questions?

Open an issue or contact the maintainers if you need help!

## Code of Conduct

Be respectful and constructive in all interactions. This project follows standard open-source community guidelines.

---

Thank you for contributing! 🎉
