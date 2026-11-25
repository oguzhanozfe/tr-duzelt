"""Karşılaştırılan yöntemler. Hepsi aynı arayüzle çalışır: küçük harfli kelime listesi alır, kelime listesi verir."""
import importlib.util
import math
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
from veri import sade      # noqa: E402


class HicbirSey:
    """Karşılaştırma tabanı: metne hiç dokunma."""
    ad = 'hiçbir şey yapma'

    def __call__(self, kelimeler):
        return list(kelimeler)


class TrDuzelt:
    """Bu depodaki yazım düzeltici (noktalama ve büyük harf aşaması hariç). kod: başka bir sürümün klasörü."""

    def __init__(self, kod=None, ad='tr-duzelt'):
        self.ad = ad
        yol = os.path.join(kod or KOK, 'tr_duzelt.py')
        spec = importlib.util.spec_from_file_location('tr_duzelt_' + str(abs(hash(yol))), yol)
        self.modul = importlib.util.module_from_spec(spec)
        sys.path.insert(0, os.path.dirname(yol))
        spec.loader.exec_module(self.modul)
        self.lm = self.modul.DilModeli()
        self.yazim = self.modul.YazimDuzeltici(self.lm)

    def gecerli(self, w):
        return self.yazim.gecerli(w)

    def __call__(self, kelimeler):
        return [sade(t) for c, _ in self.yazim.duzelt(list(kelimeler)) for t in c]


class ZemberekNormalizer:
    """Zemberek'in hazır cümle normalleştiricisi (kural tabloları + dil modeli)."""
    ad = 'zemberek-normalizer'

    def __init__(self):
        from zemberek import TurkishMorphology, TurkishSentenceNormalizer
        self.n = TurkishSentenceNormalizer(TurkishMorphology.create_with_defaults())

    def __call__(self, kelimeler):
        try:
            return [t for t in (sade(x) for x in self.n.normalize(' '.join(kelimeler)).split()) if t]
        except Exception:
            return list(kelimeler)


class ZemberekYazim:
    """Zemberek yazım denetleyicisi, bağlamsız: tanımadığı kelimeye ilk önerisini yazar."""
    ad = 'zemberek-yazim'

    def __init__(self):
        from zemberek import TurkishMorphology, TurkishSpellChecker
        self.m = TurkishMorphology.create_with_defaults()
        self.sc = TurkishSpellChecker(self.m)
        self.onbellek = {}

    def __call__(self, kelimeler):
        out = []
        for w in kelimeler:
            if w not in self.onbellek:
                y = w
                if w.isalpha() and not any(not a.is_unknown() for a in self.m.analyze(w)):
                    oneri = self.sc.suggest_for_word(w)
                    if oneri:
                        y = sade(oneri[0])
                self.onbellek[w] = y
            out.append(self.onbellek[w])
        return out


class SymSpell:
    """SymSpell: sıklık sözlüğü + düzenleme mesafesi (en çok 2). Bölünmüş/bitişik kelimeler için lookup_compound.
    Sözlük: Zemberek dil modelinin dağarcığındaki, morfolojik olarak geçerli kelimeler ve sıklıkları."""
    ad = 'symspell'

    def __init__(self, tr=None):
        from symspellpy import SymSpell as _S
        tr = tr or TrDuzelt()
        self.s = _S(max_dictionary_edit_distance=2, prefix_length=7)
        lm = tr.lm
        for w in lm.V.vocabulary:
            if w.isalpha() and len(w) < 30 and tr.yazim.duzgun(w):
                self.s.create_dictionary_entry(w, max(1, int(math.exp(lm.tekli(w)) * 1e9)))

    def __call__(self, kelimeler):
        r = self.s.lookup_compound(' '.join(kelimeler), max_edit_distance=2)
        return r[0].term.split() if r else list(kelimeler)


def kur(ad, kod=None):
    if ad == 'hicbir-sey':
        return HicbirSey()
    if ad == 'tr-duzelt':
        return TrDuzelt(kod, ad if not kod else 'tr-duzelt@' + os.path.basename(os.path.normpath(kod)))
    if ad == 'zemberek-normalizer':
        return ZemberekNormalizer()
    if ad == 'zemberek-yazim':
        return ZemberekYazim()
    if ad == 'symspell':
        return SymSpell()
    raise SystemExit('bilinmeyen sistem: %s' % ad)
