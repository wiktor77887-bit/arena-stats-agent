"""Name matching for champions, augments and items, including typos."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

KINDS = ("champion", "augment", "item")


def normalize(text: str) -> str:
    """"Kai'Sa" -> "kaisa", "Szpon Ciemnego Typa" -> "szponciemnegotypa", "Łuk" -> "luk"."""
    text = text.lower().replace("ł", "l")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text)


@dataclass(frozen=True)
class Match:
    name: str
    kind: str
    score: float

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "score": round(self.score, 3)}


class NameResolver:
    """Keeps a normalized-alias -> canonical-name lookup for every kind of object."""

    def __init__(self, aliases: dict[str, dict[str, str]]):
        self._aliases = {k: dict(aliases.get(k, {})) for k in KINDS}

    @classmethod
    def from_names(cls, names: dict[str, list[tuple[str, str]]]) -> "NameResolver":
        """names[kind] = list of (alias, canonical_name) pairs."""
        aliases: dict[str, dict[str, str]] = {k: {} for k in KINDS}
        for kind, pairs in names.items():
            for alias, canonical in pairs:
                key = normalize(alias)
                if key:
                    aliases[kind].setdefault(key, canonical)
        return cls(aliases)

    def exact(self, text: str, kind: str) -> str | None:
        """Match ignoring case, spaces, apostrophes and Polish diacritics, but not typos."""
        return self._aliases[kind].get(normalize(text))

    def search(self, text: str, kind: str | None = None, limit: int = 3) -> list[Match]:
        """Fuzzy search, best matches first."""
        query = normalize(text)
        if not query:
            return []
        kinds = KINDS if kind in (None, "any") else (kind,)
        best: dict[tuple[str, str], float] = {}
        for k in kinds:
            for alias, canonical in self._aliases[k].items():
                score = difflib.SequenceMatcher(None, query, alias).ratio()
                if alias == query:
                    score = 1.0
                elif len(query) >= 3 and alias.startswith(query):
                    score = max(score, 0.9)
                key = (canonical, k)
                if score > best.get(key, 0.0):
                    best[key] = score
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0][0]))
        return [Match(name, k, s) for (name, k), s in ranked[:limit]]
