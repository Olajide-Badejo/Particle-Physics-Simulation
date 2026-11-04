"""The one error the readers raise."""

from __future__ import annotations


class FormatError(ValueError):
    """A file is not what its reader expects. The message names the file."""
