"""Single-point-of-change shim for the affinity-rescoring dependency.

``boltz_lora`` depends on one function from the affinity-rescoring package:
:func:`featurize_complex`. In this fork it ships as part of the Boltz namespace
(``boltz.affinity_rescoring``). When the suite migrates to the standalone
``maomlab/TRIAGED`` repository the package will be renamed (see
``MIGRATION.md`` at the repo root). Centralising the import here means the
migration is a one-line edit.

To migrate: replace the body of :func:`get_featurize_complex` with::

    from triaged_rescoring import featurize_complex
    return featurize_complex

(or whatever the final package name is).
"""

from __future__ import annotations

from typing import Any, Callable


def get_featurize_complex() -> Callable[..., Any]:
    """Return the ``featurize_complex`` callable from the rescoring package.

    Deferred so that ``boltz_lora`` can be imported in environments that
    don't have the full Boltz data pipeline installed (e.g. unit tests
    that monkeypatch the featurizer).
    """
    from boltz.affinity_rescoring import featurize_complex

    return featurize_complex
