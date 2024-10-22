<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the 25th of November.


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```shell
uv venv -p 3.11.0 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```
---

# Updates, Issues, Workarounds, Notes

---
## 15/10/2024 - Workaround for environment setup
> **Note**
- loosened python-version in pyproject.toml to `requires-python = ">=3.11, <3.12"`

- ```shell
    uv venv -p 3.11
    source .venv/bin/activate
    uv pip install -r pyproject.toml --all-extras
    uv lock
    ```

- installs and uses `python-3.11.10`
> **Issue**
- UV could not find managed or system version download for 3.11.0
  - ```shell
    error: No download found for request: cpython-3.11.0-macos-aarch64-none
    ```
> **Workaround**
- workaround was to run pyenv local 3.11.0 and then `uv venv -p 3.11.0`, but this lead to other issues:
    - Had issues with dependency resolution when using 3.11.0 (`pywin32` was being referenced for some reason? And it was not compatible with `mlflow >= 2.16.0`)
---

---
## 21/10/2024 - CI pipeline edits
> **Note**
- added `uv pip install pre-commit` to the CI pipeline
- added `uv run pre-commit run --all-files` to the CI pipeline
- added `uv pip install pytest pytest-mock pytest-cov` to the CI pipeline
- added `uv run pytest --cov=power_consumption tests/` to the CI pipeline

- ```yaml
    name: CI

    on:
    pull_request:
        branches:
        - main

    jobs:
    build_and_test:
        runs-on: ubuntu-latest
        steps:
        - uses: actions/checkout@v4

        - name: Install uv
            uses: astral-sh/setup-uv@v3

        - name: Set up Python
            run: uv python install 3.11

        - name: Install the dependencies
            run: uv sync

        - name: Run pre-commit checks
            run: |
            uv pip install pre-commit
            uv run pre-commit run --all-files

        - name: Run tests with coverage
            run: |
            uv pip install pytest pytest-mock pytest-cov
            uv run pytest --cov=power_consumption tests/
  ```

## Dataset
> **Note**

- The dataset is not included in the repository to avoid large file size. It should first attempt to get the data from the UCI ML Repository.
- If that fails, the dataset is expected to be in `data/Tetuan City power consumption.csv`. You can download it from [here](https://www.kaggle.com/datasets/gmkeshav/tetuan-city-power-consumption).
---
