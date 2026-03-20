# Project-level Skills Configuration

This project uses the following skills from the global rules:

- **Python**: `~/.claude/rules/python/` - Python-specific coding standards, testing, patterns
- **Common**: `~/.claude/rules/common/` - Universal principles for all languages

## Available Agents for This Project

Based on the project type (Python), use these agents proactively:

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| python-reviewer | Python code review | After writing Python code |
| tdd-guide | Test-driven development | New features, bug fixes |
| code-reviewer | General code review | After writing code |
| build-error-resolver | Fix build errors | When build fails |

## Project-specific Notes

- This is a **Python 3.12** project with pandas, pytest, PyBroker
- Virtual environment: `.venv312`
- Use `source myanalyser/.venv312/bin/activate` before running Python code
- All code changes must pass `bash tools/v2/verify.sh` for regression
