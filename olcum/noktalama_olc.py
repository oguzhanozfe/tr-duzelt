#!/usr/bin/env python3
"""Noktalama ölçümü: noktalaması ve büyük harfleri silinmiş metne yöntem işaretleri geri koyabiliyor mu?

  python olcum/noktalama_olc.py                                   # depodaki CRF (noktalama.crfsuite)
  python olcum/noktalama_olc.py --sistem bert-hazir               # uygarkurt/bert-restore-punctuation-turkish
  python olcum/noktalama_olc.py --sistem bert:yol/klasor          # kendi eğittiğiniz BERT etiketleyici

Veri: UD Turkish-IMST'nin geliştirme ve test bölümleri. CRF bu veriyle eğitilmedi (eğitim listesi egit_noktalama.py
başlığında). Ardışık cümleler sabit tohumla 1-10 cümlelik parçalara bölünür; her parça küçük harfli, noktalamasız
tek bir kelime dizisi olarak verilir. Sonuç: her işaret için isabet (P), yakalama (R), F1.
"""
import argparse
import os
import random
import sys
from collections import Counter

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.join(KOK, 'olcum'))
import veri                                                                  # noqa: E402
import tr_duzelt as T                                                        # noqa: E402
from egit_noktalama import read_ud, sentence_items, docs_to_sequences       # noqa: E402

ISARETLER = (('P', 'nokta'), ('C', 'virgül'), ('Q', 'soru'))


def diziler(set_adi, tohum=2026):
    """[(kelimeler, etiketler)]; etiket = kelimeden sonra gelen işaret: O, C, P, Q."""
    sents = read_ud(veri.dosya(set_adi + '.conllu'))[0]
    belge = [it for it in (sentence_items(t) for t in sents) if it]
    seqs = docs_to_sequences([belge], Counter(), Counter(), Counter(), random.Random(tohum))
    return [(w, [e[1] for e in etk]) for w, etk in seqs]


class Crf:
    ad = 'crf (noktalama.crfsuite)'

    def __init__(self):
        self.n = T.Noktalayici(os.path.join(KOK, 'noktalama.crfsuite'), T.DilModeli(), None)

    def __call__(self, kelimeler):
        return [e[1] for e in self.n.etiketle(kelimeler)[0]]


class BertEtiketleyici(T.BertNoktalayici):
    def __init__(self, model, virgul_esik=0.0, soru_esik=0.0):
        super().__init__(model, virgul_esik, soru_esik)
        self.ad = 'bert:' + model


def kur(ad, virgul_esik=0.0, soru_esik=0.0):
    if ad == 'crf':
        return Crf()
    if ad == 'bert-hazir':
        return BertEtiketleyici('uygarkurt/bert-restore-punctuation-turkish', virgul_esik, soru_esik)
    if ad.startswith('bert:'):
        return BertEtiketleyici(ad[5:], virgul_esik, soru_esik)
    raise SystemExit('bilinmeyen sistem: %s' % ad)


def degerlendir(dizi, tahminler):
    tp, fp, fn = Counter(), Counter(), Counter()
    for (_, altin), tahmin in zip(dizi, tahminler):
        assert len(altin) == len(tahmin)
        for g, p in zip(altin, tahmin):
            if g == p:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1
    out = {}
    for e, ad in ISARETLER:
        p = tp[e] / (tp[e] + fp[e]) if tp[e] + fp[e] else 0.0
        r = tp[e] / (tp[e] + fn[e]) if tp[e] + fn[e] else 0.0
        out[ad] = (p, r, 2 * p * r / (p + r) if p + r else 0.0, tp[e] + fn[e])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sistem', default='crf', help='crf | bert-hazir | bert:<model klasörü ya da adı>')
    ap.add_argument('--set', nargs='*', default=['imst-dev', 'imst-test'])
    ap.add_argument('--virgul-esik', type=float, default=0.0, help='BERT: virgül için en düşük olasılık')
    ap.add_argument('--soru-esik', type=float, default=0.0, help='BERT: soru işareti için en düşük olasılık')
    a = ap.parse_args()
    s = kur(a.sistem, a.virgul_esik, a.soru_esik)
    print('sistem: %s' % s.ad)
    for ad in a.set:
        d = diziler(ad)
        r = degerlendir(d, [s(w) for w, _ in d])
        print('%-10s %5d kelime | ' % (ad, sum(len(w) for w, _ in d)) + ' | '.join(
            '%s P=%.3f R=%.3f F1=%.3f (n=%d)' % ((isaret,) + r[isaret]) for _, isaret in ISARETLER), flush=True)


if __name__ == '__main__':
    main()
