"""smishlens: calibrated, explainable SMS spam and smishing detection.

Tested on sources it has never seen, hardened against disguised text, and able
to say "not sure" instead of guessing.
"""

__version__ = "0.1.0"

# Keep imports lazy-free: these modules are cheap and everything else depends on them.
from smishlens.normalise import normalise, normalise_with_flags  # noqa: F401
