"""Importing this package registers all built-in EOS backends.

Each module calls :func:`xeos.registry.register` at import time, so importing
the package populates the global registry.  New schemes are added by dropping a
module here and importing it below.
"""

from . import _linear  # noqa: F401
from . import _wright  # noqa: F401
from . import _jmd95  # noqa: F401
from . import _unesco  # noqa: F401
from . import _mdjwf  # noqa: F401
from . import _roquet  # noqa: F401
from . import _roquet_spv  # noqa: F401
from . import _roquet_idealized  # noqa: F401
from . import _teos10  # noqa: F401
