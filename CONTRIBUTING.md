# Contributing to PyNumDiff

Thank you for your interest in contributing to PyNumDiff! This document provides guidelines and instructions for contributing to the project.

## Project Structure

- `pynumdiff/` - Main source code, tests in subfolder
- `notebooks/` - Jupyter notebook examples and experiments
- `docs/` - Documentation source files
- `.github/workflows/` - GitHub Actions CI configuration

## Opening Issues

If you discover a bug or have an improvement idea or question, the place to start is the [Issues page](https://github.com/florisvb/PyNumDiff/issues), which is really the beating heart of any project (even if you're just here to give us kudos). Take a look through the history to get a sense of what has been done and which ideas have been considered before. A lot of hard-won knowledge and tough decisions have been explored and documented in the Issues. Feel free to open new issues if we haven't covered something.

### Reporting Bugs

If reporting bugs, make sure you're on the latest version and that we haven't already taken care of something. Please include some or all of:

1. **Descriptive title**
2. **What happened**:
   - What you were trying to do
   - What you expected to happen
   - What actually happened
3. **Minimal code example** that reproduces the issue
4. **Environment information**: Python and library versions (`pynumdiff`, `numpy`, `scipy`, anything salient)
5. **Error messages** or stack traces
6. **Additional context** (screenshots, data files, etc.)

### Proposing Features

If we've got an ongoing or old discussion on a topic, and you can manage to find it, tack on discussion there. If your idea is otherwise within the scope of the project, start a new discussion. Let us know why you think something is necessary, and please feel free to suggest what would need to change to make it happen. The more investigation and thinking you do to show the feasibility and practicality of something, the more load that takes off other maintainers.

Here are some things you might include:

1. **Descriptive title**
2. **Problem statement**: What problem does this feature solve?
3. **Proposed solution**: How would you implement it?
4. **Alternatives considered**: What other approaches did you consider?
5. **Additional context**: Examples, use cases, etc.

## Addressing Issues

Look for issues labeled `good first issue` if you're new to the project. Talk to us, and we can suggest things that need to be donem, of varying levels of code and research difficulty.

### Research

Some issues will require going and digging into alternative methods of differentiation so they can be added to our collection, or comparing a new or modified method to other methods. This kind of work requires some mathematical chops, but if you're down, we're happy about it.

### Contributing Code

1. Fork the repository (button on the main repo page)
2. Clone down your version (`git clone https://github.com/YOUR_USERNAME/PyNumDiff.git`)
3. Set its upstream to point to this version so you can easily pull our changes (`git remote add upstream https://github.com/florisvb/PyNumDiff.git`)
4. Install the package in development mode (`pip install -e .`) as well as dependencies like `numpy`, `pytest`, `cvxpy`, etc.
5. Create a branch for your changes (`cd PyNumDiff`; `git checkout -b your-feature`)
6. Make your changes and commit (`git add file`; `git commit -m "descriptive commit message"`)
7. Update your fork with the latest changes from upstream (`git fetch upstream`; `git checkout master`; `git merge upstream/master`)
8. Run the tests to make sure they pass (`pytest -s`, with helpful `--plot` or `--bounds` flags for debugging), possibly adding new ones
9. Push your code up to the cloud (`git push`)
10. Submit a pull request ("PR") (via the pull requests page on the website)
11. We'll review, leave comments, kick around further change ideas, and merge.

No strict coding style is enforced, although we consider docstrings to be very important. The project uses `pylint` for code quality checks (`pylint pynumdiff`), because we're trying to meet a high bar so the JOSS (Journal of Open Source Software) likes us.

Bear in mind that smaller, focused PRs are generally easier to review. We encourage descriptive commit messages that explain what changed and why. Long, detailed commit messages are appreciated as they help others understand the project's history.

Once you push, GitHub Actions will kick off our continuous integration job, which runs the tests, including:
- `test_diff_methods`: Broadly tests for correctness and ability to actually differentiate
- `test_utils`: Contains tests of supporting and miscellaneous functionality like simulations and evaluation metrics
- `test_optimize`: Tests the hyperparameter optimization code
