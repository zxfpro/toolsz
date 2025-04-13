#!/bin/bash

#pytest-html

github_repo=$(basename "$(pwd)")
uv run pytest --html=$test_html_path/$github_repo.html
