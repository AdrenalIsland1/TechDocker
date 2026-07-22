# TechDocker Technical Summary

## Product Purpose

TechDocker keeps repository summaries synchronized with reviewed code changes.
It exists so that a project's written overview never drifts away from what the
code actually does, and so reviewers can trust the summary during a pull
request.

## Repository Automation

TechDocker detects changed files from Git history and assembles them into a
review package for each push.

In `change_summary_generator.py`, `create_change_package` records only changed
file paths in `latest_change_summary.json`.

The section router in `section_candidate_scorer.py` uses a fixed list of
expected heading names to choose where an update belongs.

## Quality and Tests

The test suite exercises deterministic routing and updater behavior offline,
using fixtures instead of contacting any network service or language model.

## CI/CD Review Flow

GitHub Actions opens a pull request containing the proposed summary update, so
a human reviewer approves every documentation change before it merges.
