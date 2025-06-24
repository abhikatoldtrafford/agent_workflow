"""Setup file for the agent-workflow package."""

from setuptools import setup

# This setup.py exists for pip compatibility purposes.
# Actual build configuration is in pyproject.toml

if __name__ == "__main__":
    setup(
        # The actual configuration is loaded from pyproject.toml
        name="agent-workflow",
        use_scm_version=True,
    )