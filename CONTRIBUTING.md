# Contributing to PyNumDiff

Thank you for your interest in contributing to PyNumDiff! This document provides guidelines and instructions for contributing to the project. Following these guidelines helps communicate that you respect the time of the developers managing and developing this open-source project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)
- [Commit Message Conventions](#commit-message-conventions)
- [Questions?](#questions)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please report it by [opening an issue](https://github.com/florisvb/PyNumDiff/issues/new). See the [Reporting Bugs](#reporting-bugs) section for details.

### Suggesting Enhancements

Have an idea for a new feature or improvement? Please [open an issue](https://github.com/florisvb/PyNumDiff/issues/new) to discuss it. See the [Proposing Features](#proposing-features) section for details.

### Contributing Code

1. Look for issues labeled `good first issue` if you're new to the project
2. Fork the repository
3. Create a branch for your changes
4. Make your changes following our guidelines
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Git
- (Optional) A virtual environment manager (venv, conda, etc.)

### Setting Up the Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PyNumDiff.git
   cd PyNumDiff
   ```

3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/florisvb/PyNumDiff.git
   ```

4. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

5. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

6. **Install development dependencies**:
   ```bash
   pip install pytest pylint
   ```

7. **Verify the installation**:
   ```bash
   pytest -s
   ```

### Project Structure

- `pynumdiff/` - Main source code
- `examples/` - Jupyter notebook examples
- `docs/` - Documentation source files
- `.github/workflows/` - GitHub Actions CI configuration
- `tests/` - Test files (if applicable)

## Code Style Guidelines

### Python Style Guide

PyNumDiff follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines. Here are the key points:

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 79 characters (soft limit) or 99 characters (hard limit)
- Use meaningful variable and function names
- Follow naming conventions:
  - Functions and variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Code Quality

The project uses `pylint` for code quality checks. Before submitting a PR:

1. **Run pylint** on your changes:
   ```bash
   pylint pynumdiff/
   ```

2. **Or use the project's linting script**:
   ```bash
   python linting.py
   ```

3. **Fix any issues** reported by pylint before submitting your PR

### Editor Configuration

The project includes an `.editorconfig` file that ensures consistent formatting. Most modern editors support EditorConfig automatically.

## Testing Guidelines

### Running Tests

PyNumDiff uses `pytest` for testing. To run tests:

```bash
# Run all tests
pytest -s

# Run tests with plots (to visualize method results)
pytest -s --plot

# Run tests with bounds (to print log error bounds)
pytest -s --bounds
```

### Writing Tests

- Write tests for new features and bug fixes
- Follow the existing test structure
- Ensure all tests pass before submitting a PR
- Tests should be deterministic and not depend on external resources

### Continuous Integration

The project uses GitHub Actions for continuous integration. All pull requests are automatically tested. Make sure your changes pass all CI checks before requesting a review.

## Pull Request Process

### Before Submitting

1. **Update your fork** with the latest changes from upstream:
   ```bash
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix-name
   ```

3. **Make your changes** following the code style guidelines

4. **Write or update tests** as needed

5. **Run tests** to ensure everything passes:
   ```bash
   pytest -s
   ```

6. **Run linting** to check code quality:
   ```bash
   python linting.py
   ```

7. **Commit your changes** with clear, descriptive commit messages (see [Commit Message Conventions](#commit-message-conventions))

### Submitting a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub:
   - Go to the [PyNumDiff repository](https://github.com/florisvb/PyNumDiff)
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template with:
     - A clear title and description
     - Reference to the related issue (e.g., "Fixes #169")
     - Description of changes
     - Any breaking changes

3. **Wait for CI** to run and ensure all checks pass

4. **Respond to feedback** from maintainers and reviewers

5. **Keep your PR up to date** by rebasing on master if needed:
   ```bash
   git fetch upstream
   git rebase upstream/master
   git push --force-with-lease origin feature/your-feature-name
   ```

### PR Guidelines

- Keep PRs focused on a single issue or feature
- Keep PRs reasonably sized (if large, consider breaking into smaller PRs)
- Ensure all CI checks pass
- Request review from maintainers when ready
- Be responsive to feedback

## Reporting Bugs

### Before Submitting a Bug Report

1. **Check existing issues** to see if the bug has already been reported
2. **Test with the latest version** to ensure the bug still exists
3. **Check the documentation** to ensure you're using the library correctly

### How to Report a Bug

When reporting a bug, please include:

1. **Clear and descriptive title**
2. **Steps to reproduce**:
   - What you were trying to do
   - What you expected to happen
   - What actually happened
3. **Minimal code example** that reproduces the issue
4. **Environment information**:
   - Python version
   - PyNumDiff version
   - Operating system
5. **Error messages** or stack traces (if applicable)
6. **Additional context** (screenshots, data files, etc.)

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. ...
2. ...

**Expected behavior**
A clear and concise description of what you expected to happen.

**Code example**
```python
# Minimal code that reproduces the issue
```

**Environment**
- Python version: 
- PyNumDiff version:
- OS:

**Additional context**
Add any other context about the problem here.
```

## Proposing Features

### Before Proposing a Feature

1. **Check existing issues** to see if the feature has been discussed
2. **Consider the scope** - is it within the project's goals?
3. **Think about implementation** - is it feasible?

### How to Propose a Feature

When proposing a feature, please include:

1. **Clear and descriptive title**
2. **Problem statement**: What problem does this feature solve?
3. **Proposed solution**: How would you implement it?
4. **Alternatives considered**: What other approaches did you consider?
5. **Additional context**: Examples, use cases, etc.

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or examples about the feature request here.
```

## Commit Message Conventions

### Commit Message Format

We follow a conventional commit message format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without changing functionality
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, build config, etc.)

### Examples

```
feat(optimize): add support for custom search spaces

Allow users to specify custom parameter search spaces in the
optimize function for better control over hyperparameter tuning.

Fixes #123
```

```
fix(differentiation): correct boundary condition handling

Fixed an issue where boundary conditions were not properly
handled for signals with variable step sizes.

Closes #456
```

```
docs(readme): update installation instructions

Updated README with clearer installation steps for Windows users.
```

### Guidelines

- Use the imperative mood ("add feature" not "added feature")
- Keep the subject line under 50 characters
- Capitalize the subject line
- Do not end the subject line with a period
- Use the body to explain what and why vs. how
- Reference issues and PRs in the footer

## Questions?

If you have questions about contributing:

1. Check the [documentation](https://pynumdiff.readthedocs.io/)
2. Look through [existing issues](https://github.com/florisvb/PyNumDiff/issues)
3. Open a new issue with the `question` label

## Additional Resources

- [PyNumDiff Documentation](https://pynumdiff.readthedocs.io/)
- [Project README](README.md)
- [GitHub Issues](https://github.com/florisvb/PyNumDiff/issues)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [pytest Documentation](https://docs.pytest.org/)

Thank you for contributing to PyNumDiff! 🎉

