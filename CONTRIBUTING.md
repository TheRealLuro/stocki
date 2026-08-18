# Contributing to Stocki

Thank you for contributing to Stocki! This document outlines our team's workflow for branching, reviewing, and merging code.

---

## Branching Strategy

We use a **feature-branch workflow**:

1. The `main` branch is always deployable.
2. Create a new branch for every feature, bugfix, or experiment.
3. Branch names should be descriptive:
   - `feature/data-pipeline-setup`
   - `bugfix/missing-volume-values`
   - `experiment/cnn-2d-conv`
   - `docs/update-readme`

### Creating a Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

---

## Commits

- **Commit early and often** with clear, descriptive messages.
- Use the present tense (e.g., "Add data cleaning script" not "Added data cleaning script").
- **Commit and push from your own GitHub account** — contribution is graded weekly from repository activity.

---

## Pull Requests

1. Push your branch to GitHub from your IDE or terminal:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a **Pull Request (PR)** against `main`.
3. Fill out the PR template (if applicable) with a summary of changes.
4. **Request review from at least one teammate** before merging.
5. Address review feedback and push updates.
6. Once approved, **squash and merge** or **merge** into `main`.

### PR Review Checklist

- [ ] Code runs without errors
- [ ] New code is documented (docstrings / comments)
- [ ] Tests pass (if applicable)
- [ ] No sensitive data (API keys, credentials) committed
- [ ] Branch is up to date with `main`

---

## Code Style

- Follow PEP 8 for Python code.
- Use shared IDE settings (formatting, linting) stored in the repository.

---

## Questions?

Open an issue or reach out in your team chat. Happy coding!
