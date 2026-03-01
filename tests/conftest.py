"""
Shared test fixtures for MAUDE test suite.
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temp directory for collab JSON storage."""
    d = tmp_path / "collab"
    d.mkdir()
    (d / "projects").mkdir()
    (d / "tasks").mkdir()
    return d


@pytest.fixture
def tmp_file(tmp_path):
    """Create a temp file with known content."""
    f = tmp_path / "test.txt"
    f.write_text("line one\nline two\nline three\n")
    return f


@pytest.fixture
def tmp_dir_with_files(tmp_path):
    """Create a temp directory with several files."""
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.py").write_text("print('hi')")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.txt").write_text("nested")
    return tmp_path


@pytest.fixture
def sample_messages():
    """Test messages with known keyword matches."""
    return {
        "email": "check my email inbox",
        "drive": "list my google drive files",
        "calendar": "what's on my calendar today",
        "image": "generate an image of a cat",
        "collab": "who's online on the mesh",
        "plain": "tell me a joke",
        "shared": "list shared folder",
        "server": "run a command on the server",
    }
