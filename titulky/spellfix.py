"""Deterministická spell-korekcia commitnutých slov cez obrovský
SK slovník (hunspell-sk unmunch, 3.0M tvarov, marisa-trie 6MB).

Pravidlo (konzervatívne, merané 26.10.):
- slovo ≥5 znakov, čisto alfabetické (vrátane diakritiky),
- nie je v slovníku (lowercase),
- existuje PRÁVE JEDEN kandidát vo vzdialenosti 1 (delete/replace/insert)
  → nahradí sa (zachová kapitalizáciu prvého písmena).
Viac kandidátov = nejednoznačné = nechaj tak. ~1-3 ms na slovo.

Zapína sa TITULKY_SPELLFIX=1 (default zapnuté, 0 vypne).
"""
import os

SK_ALPHA = "aáäbcčdďdzdžeéfghiíjklĺľmnňoóôpqrsštťuúvwxyýzž"
_TRIE_PATH = os.environ.get(
    "TITULKY_SK_TRIE",
    "/Volumes/GRINDCASTER/AI-MODELS/hunspell-sk/sk_words.marisa")


class SpellFix:
    def __init__(self, path: str | None = None):
        self.trie = None
        p = path or _TRIE_PATH
        if os.path.exists(p):
            import marisa_trie
            self.trie = marisa_trie.Trie()
            self.trie.load(p)

    def _cands_d1(self, w: str) -> set[str]:
        tr = self.trie
        out = set()
        n = len(w)
        for i in range(n):
            c = w[:i] + w[i + 1:]              # delete
            if c in tr:
                out.add(c)
            pre, suf = w[:i], w[i + 1:]
            for ch in SK_ALPHA:                 # replace
                if ch != w[i]:
                    c = pre + ch + suf
                    if c in tr:
                        out.add(c)
        for i in range(n + 1):                  # insert
            pre, suf = w[:i], w[i:]
            for ch in SK_ALPHA:
                c = pre + ch + suf
                if c in tr:
                    out.add(c)
        return out

    def apply(self, words: list[str]) -> tuple[list[str],
                                               list[tuple[str, str]]]:
        """Rovnaký kontrakt ako Lexicon.apply: (nové slová, edity)."""
        if self.trie is None:
            return words, []
        out, edits = [], []
        for w in words:
            # odlep interpunkciu na okrajoch
            i0, i1 = 0, len(w)
            while i0 < i1 and not w[i0].isalpha():
                i0 += 1
            while i1 > i0 and not w[i1 - 1].isalpha():
                i1 -= 1
            core = w[i0:i1]
            low = core.lower()
            if (len(low) < 5 or not all(c.isalpha() for c in low)
                    or low in self.trie):
                out.append(w)
                continue
            cands = self._cands_d1(low)
            if len(cands) != 1:
                out.append(w)
                continue
            fix = cands.pop()
            if core[:1].isupper():
                fix = fix[:1].upper() + fix[1:]
            nw = w[:i0] + fix + w[i1:]
            edits.append((w, nw))
            out.append(nw)
        return out, edits
