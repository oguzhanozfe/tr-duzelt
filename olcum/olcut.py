"""Ölçüt: bir düzeltici neyi düzeltti, neyi bıraktı, neyi BOZDU?

Önce girdi ile doğru metin kelime kelime hizalanır ve aradaki farklar "düzenleme" olarak çıkarılır
(örn. 3. kelime 'herkez' -> 'herkes'; 5-6. kelimeler 'orma na' -> 'ormana'). Aynısı girdi ile sistemin
çıktısı için yapılır. İki düzenleme kümesi karşılaştırılır.
"""
import difflib
from functools import lru_cache


@lru_cache(maxsize=200000)
def _uzaklik(a, b):
    """Harf düzeyinde düzenleme mesafesi / uzun olanın boyu (0 = aynı, 1 = bambaşka)."""
    if a == b:
        return 0.0
    onceki = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        simdiki = [i]
        for j, cb in enumerate(b, 1):
            simdiki.append(min(onceki[j] + 1, simdiki[j - 1] + 1, onceki[j - 1] + (ca != cb)))
        onceki = simdiki
    return onceki[-1] / max(len(a), len(b))


def _blok_hizala(S, T, s0):
    """Farklı olan bir bloğu kelime kelime eşle; 1-1, 1-2 (ayırma) ve 2-1 (birleştirme) eşlemelerine izin ver."""
    n, m = len(S), len(T)
    INF = 1e9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    G = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            d = D[i][j]
            if d >= INF:
                continue
            secenek = []
            if i < n and j < m:
                secenek.append((i + 1, j + 1, 0.2 + _uzaklik(S[i], T[j])))
            if i < n and j + 1 < m:
                secenek.append((i + 1, j + 2, 0.3 + _uzaklik(S[i], T[j] + T[j + 1])))
            if i + 1 < n and j < m:
                secenek.append((i + 2, j + 1, 0.3 + _uzaklik(S[i] + S[i + 1], T[j])))
            if i < n:
                secenek.append((i + 1, j, 1.0))
            if j < m:
                secenek.append((i, j + 1, 1.0))
            for ni, nj, c in secenek:
                if d + c < D[ni][nj]:
                    D[ni][nj] = d + c
                    G[ni][nj] = (i, j)
    out = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = G[i][j]
        if tuple(S[pi:i]) != tuple(T[pj:j]):
            out.append((s0 + pi, s0 + i, tuple(T[pj:j])))
        i, j = pi, pj
    return out


def duzenlemeler(kaynak, hedef):
    """{(başlangıç, bitiş, yerine gelen kelimeler)} — kaynak[başlangıç:bitiş] yerine ne yazılmış."""
    out = set()
    for tur, i1, i2, j1, j2 in difflib.SequenceMatcher(None, kaynak, hedef, autojunk=False).get_opcodes():
        if tur != 'equal':
            out.update(_blok_hizala(kaynak[i1:i2], hedef[j1:j2], i1))
    return out


def degerlendir(ciftler, ciktilar):
    """ciftler: [(girdi, doğru)], ciktilar: [sistem çıktısı]. Sayımları ve oranları döndürür."""
    s = dict(hata=0, duzeltildi=0, birakildi=0, yanlis_duzeltildi=0, dogru_kelime=0, bozuldu=0, sistem=0, tam_cumle=0)
    for (girdi, dogru), cikti in zip(ciftler, ciktilar):
        altin = duzenlemeler(girdi, dogru)
        sistem = duzenlemeler(girdi, cikti)
        hatali_yer = set()
        for a, b, _ in altin:
            hatali_yer.update(range(a, b) if b > a else [a - 0.5])
        s['hata'] += len(altin)
        s['sistem'] += len(sistem)
        s['dogru_kelime'] += len(girdi) - sum(1 for x in hatali_yer if isinstance(x, int))
        s['duzeltildi'] += len(altin & sistem)
        for a, b, yeni in sistem - altin:
            if any(i in hatali_yer for i in range(a, b)) or (b == a and (a - 0.5) in hatali_yer):
                s['yanlis_duzeltildi'] += 1          # hatalı yere dokundu ama yanlış düzeltti
            else:
                s['bozuldu'] += max(1, b - a)        # doğru olan kelimeyi değiştirdi
        s['tam_cumle'] += int(list(cikti) == list(dogru))
    s['birakildi'] = s['hata'] - s['duzeltildi'] - s['yanlis_duzeltildi']
    if s['birakildi'] < 0:                            # bir hataya iki ayrı yanlış düzenleme denk geldiyse
        s['birakildi'] = 0
    yuzde = lambda a, b: 100.0 * a / b if b else 0.0
    s['isabet'] = yuzde(s['duzeltildi'], s['sistem'])                 # precision: yaptığı değişikliklerin doğru olanı
    s['yakalama'] = yuzde(s['duzeltildi'], s['hata'])                 # recall: hataların düzeltileni
    p, r = s['isabet'] / 100, s['yakalama'] / 100
    s['f05'] = 100 * (1.25 * p * r / (0.25 * p + r)) if p + r else 0.0   # isabete iki kat ağırlık veren F
    s['bozma'] = yuzde(s['bozuldu'], s['dogru_kelime'])               # doğru kelimelerin bozulanı
    s['tam_cumle_yuzde'] = yuzde(s['tam_cumle'], len(ciftler))
    return s
