#!/bin/bash

#pytest-html

project="tools"
uv run pytest --html=$test_html_path/$project.html
