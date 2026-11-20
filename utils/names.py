"""Match speaker names the model wrote back to the cast names.

Claude sometimes drops or adds a title ("Priya Raman" for "Dr. Priya Raman"),
which used to crash filming with "Missing private brief".
"""
import re

TITLES = {"dr", "prof", "professor", "mr", "mrs", "ms", "mx"}


def _key(name):
    words = re.findall(r"[a-z0-9'-]+", str(name).lower())
    return [w for w in words if w not in TITLES]


def match_speaker(name, speakers):
    """Cast name for `name`, or None. Exact first, then ignoring titles, then a unique surname."""
    names = [s.name for s in speakers]
    if name in names:
        return name
    key = _key(name)
    if not key:
        return None
    same = [n for n in names if _key(n) == key]
    if len(same) == 1:
        return same[0]
    surname = [n for n in names if _key(n) and _key(n)[-1] == key[-1]]
    return surname[0] if len(surname) == 1 else None


def remap(mapping, speakers):
    """Re-key a {name: value} dict to cast names; keys that match nobody are dropped."""
    out = {}
    for name, value in (mapping or {}).items():
        cast = match_speaker(name, speakers)
        if cast and cast not in out:
            out[cast] = value
    return out
