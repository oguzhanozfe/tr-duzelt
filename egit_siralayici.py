#!/usr/bin/env python3
"""Aday sıralayıcıyı eğitir: "bu aday, yazılmak istenen kelime mi?" sorusuna olasılık veren model.

Nasıl çalışır:
  1. Temiz cümleler alınır (UD Turkish-BOUN eğitim bölümü; ilk çalıştırmada indirilir).
  2. Her cümle, insanların yaptığı hatalara benzeyen biçimde bozulur: yazım hatası, Türkçe karaktersiz yazım,
     i/ı karışıklığı, uzatılmış harf, konuşma dili, ünlüsüz kısaltma, bitişik yazılmış de/ki/mi, araya giren
     İngilizce. Doğrusunu bildiğimiz için her adayın doğru olup olmadığını da biliriz.
  3. Bozuk cümle düzelticinin aday üreticisinden geçirilir; her aday bir satır olur (48 öznitelik + doğru/yanlış).
  4. Gradyan artırmalı karar ağaçları (LightGBM) bu tabloyu öğrenir; model siralayici.json olarak yazılır.
     tr_duzelt.py onu saf Python ile çalıştırır.

Eğitim için ek kütüphane:  pip install lightgbm
Kullanım:                  python egit_siralayici.py [--cumle 6000] [--tur 2] [--cikti siralayici.json]
"""
import argparse
import json
import os
import random
import re
import sys
import time

KOK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.join(KOK, 'olcum'))
import tr_duzelt as T                     # noqa: E402
from veri import Bozucu, temiz_cumleler   # noqa: E402

UNLULER = 'aeıioöuü'
# ayrı yazılması gerekirken bitişik yazılanlar: de/da/ki bağlaçları ve soru eki (geldin mi -> geldinmi)
AYRI_YAZILAN = re.compile(r'^(?:de|da|ki|m[ıiuü](?:y[ıiuü]m|s[ıiuü]n|y[ıiuü]z|s[ıiuü]n[ıiuü]z|d[ıiuü]r|yd[ıiuü][mnk]?)?)$')
KONUSMA = [('yorsunuz', 'yosunuz'), ('yorsun', 'yosun'), ('yorsun', 'yon'), ('yorum', 'yom'), ('yoruz', 'yoz'),
           ('yorlar', 'yolar'), ('yordu', 'yodu'), ('yor', 'yo'),
           ('acağım', 'acam'), ('eceğim', 'ecem'), ('acağım', 'ıcam'), ('eceğim', 'icem'),
           ('acağız', 'acaz'), ('eceğiz', 'ecez'), ('acağız', 'ıcaz'), ('eceğiz', 'icez'),
           ('acaksın', 'acan'), ('eceksin', 'ecen'), ('acak', 'ıcak'), ('ecek', 'icek')]
KONUSMA_KELIME = {'burada': 'burda', 'şurada': 'şurda', 'orada': 'orda', 'nerede': 'nerde', 'buradan': 'burdan',
                  'oradan': 'ordan', 'bir': 'bi', 'değil': 'diil', 'herkes': 'herkez', 'yalnız': 'yanlız',
                  'şey': 'şi', 'sonra': 'sora', 'öyle': 'öle', 'böyle': 'böle'}


class EgitimBozucu(Bozucu):
    """olcum/veri.py'deki bozucunun genişletilmişi: gerçek tweetlerde görülen hata türlerini de üretir.
    Kural tabloları (yanlış -> doğru) burada TERS çevrilerek de kullanılır: temiz metinde tablonun "doğru" tarafı
    geçiyorsa yerine "yanlış" biçimi yazılır. Böylece sıralayıcı, bir kural adayının ne zaman doğru olduğunu öğrenir."""

    def __init__(self, gecerli, tohum, yazim=None):
        super().__init__(gecerli, tohum)
        self.tek, self.cift, self.govde, self.govde_cift = {}, {}, [], []
        tablolar = [T.TAM_DUZELT, T.KONUSMA_GECERLI] + ([yazim.tam] if yazim is not None else [])
        for tablo in tablolar:
            for yanlis, dogru in tablo.items():
                d = dogru.split()
                if len(d) == 1:
                    self.tek.setdefault(d[0], []).append(yanlis)
                elif len(d) == 2:
                    self.cift.setdefault(tuple(d), []).append(yanlis)
        for yanlis, dogru in T.GOVDE_DUZELT.items():
            d = dogru.split()
            if len(d) == 1:
                self.govde.append((d[0], yanlis))
            elif len(d) == 2:
                self.govde_cift.append((d[0], d[1], yanlis))

    def kural_tersi(self, ws, i):
        """ws[i] (ve gerekirse ws[i+1]) bir kuralın doğru tarafıysa: (yanlış biçim, tüketilen kelime sayısı) ya da None."""
        w = ws[i]
        if i + 1 < len(ws):
            v = ws[i + 1]
            if (w, v) in self.cift:
                return self.rng.choice(self.cift[(w, v)]), 2
            for a, b, yanlis in self.govde_cift:               # fark ettim -> farkettim
                if w == a and v.startswith(b):
                    return yanlis + v[len(b):], 2
        if w in self.tek:
            return self.rng.choice(self.tek[w]), 1
        for dogru, yanlis in self.govde:                       # herkese -> herkeze
            if w.startswith(dogru):
                return yanlis + w[len(dogru):], 1
        return None

    def gurultu(self, w):
        rng = self.rng
        r = rng.random()
        if r < 0.20:                                              # Türkçe karaktersiz
            a = w.translate(T._ASCII)
            if a != w:
                return a
        elif r < 0.27 and 'i' in w:                               # i yerine ı
            return w.replace('i', 'ı') if rng.random() < 0.6 else w.replace('i', 'ı', 1)
        elif r < 0.40:                                            # uzatma: tatlımmm, çoook
            if rng.random() < 0.5:
                return w + w[-1] * rng.randint(1, 3)
            yer = [i for i, ch in enumerate(w) if ch in UNLULER]
            if yer:
                i = rng.choice(yer)
                return w[:i] + w[i] * rng.randint(2, 4) + w[i + 1:]
        elif r < 0.58:                                            # konuşma dili
            if w in KONUSMA_KELIME:
                return KONUSMA_KELIME[w]
            uygun = [(a, b) for a, b in KONUSMA if w.endswith(a)]
            if uygun:
                a, b = rng.choice(uygun)
                return w[:-len(a)] + b
        elif r < 0.62 and 3 <= len(w) <= 6:                       # ünlüsüz: snr, tmm
            k = ''.join(ch for ch in w if ch not in UNLULER)
            if 2 <= len(k) < len(w):
                return k
        return self.yazim_hatasi(w)

    def cumle(self, ws):
        """-> (girdi kelimeleri, her girdi kelimesinin doğrusu: kelime demeti ya da None = etiket yok)"""
        rng = self.rng
        girdi, dogru = [], []
        ascii_modu = rng.random() < 0.25
        p_boz = rng.choice([0.08, 0.2, 0.35])
        n = len(ws)
        bolunecek = rng.randrange(n) if rng.random() < 0.15 else -1
        i = 0
        while i < n:
            w = ws[i]
            ters = self.kural_tersi(ws, i) if rng.random() < 0.5 else None
            if ters and i != bolunecek:                                         # kural tablosunun tersi
                y, tuketilen = ters
                girdi.append(y.translate(T._ASCII) if ascii_modu else y)
                dogru.append(tuple(ws[i:i + tuketilen]))
                i += tuketilen
                continue
            if i + 1 < n and AYRI_YAZILAN.match(ws[i + 1]) and rng.random() < 0.35:   # ben de -> bende
                j = w + ws[i + 1]
                girdi.append(j.translate(T._ASCII) if ascii_modu else j)
                dogru.append((w, ws[i + 1]))
                i += 2
                continue
            if i == bolunecek and len(w) >= 5:                                  # orma na: birleştirme ayrı ele alınıyor
                girdi += list(self.bol(w))
                dogru += [None, None]
                i += 1
                continue
            y = w
            if len(w) >= 3 and self.gecerli(w) and rng.random() < p_boz:
                y = self.gurultu(w) or w
            elif len(w) == 2 and rng.random() < 0.5 * p_boz:                    # ki -> kii, ya -> yaaa
                y = w + w[-1] * rng.randint(1, 3)
            if ascii_modu:
                y = y.translate(T._ASCII)
            girdi.append(y)
            dogru.append((w,))
            i += 1
        if rng.random() < 0.3:                                                  # araya giren İngilizce: dokunma
            ing = self.ingilizce()
            k = rng.randrange(0, len(girdi) + 1)
            girdi[k:k] = ing
            dogru[k:k] = [(t,) for t in ing]
        return girdi, dogru


def tablo_uret(yazim, cumleler, tohum, tur):
    X, y, grup = [], [], []
    b = EgitimBozucu(yazim.gecerli, tohum, yazim)
    t0 = time.time()
    for g, ws in enumerate(cumleler * tur):
        girdi, dogru = b.cumle(ws)
        if not girdi:
            continue
        # ilk geçiş: sıralayıcısız (elle ayarlı) düzeltme; komşu kelimeler oradan okunur
        sonuc = yazim.duzelt(girdi)
        for i, satirlar in yazim.aday_satirlari(girdi, sonuc):
            if dogru[i] is None:
                continue
            for c, v in satirlar:
                X.append(v)
                y.append(int(tuple(t.replace("'", '') for t in c) == dogru[i]))
                grup.append(g)
        if g % 500 == 499:
            print('  %d cümle, %d satır, %.0f sn' % (g + 1, len(X), time.time() - t0), flush=True)
    return X, y, grup


def agac_duzlestir(kok):
    """LightGBM'in iç içe ağaç yapısını düz listelere çevir (f: öznitelik no ya da yaprakta -1)."""
    f, t, sol, sag, deger = [], [], [], [], []

    def gez(d):
        k = len(f)
        f.append(-1); t.append(0.0); sol.append(-1); sag.append(-1); deger.append(0.0)
        if 'leaf_value' in d:
            deger[k] = d['leaf_value']
        else:
            assert d['decision_type'] == '<='
            f[k] = d['split_feature']
            t[k] = d['threshold']
            sol[k] = gez(d['left_child'])
            sag[k] = gez(d['right_child'])
        return k
    gez(kok)
    return {'f': f, 't': t, 'l': sol, 'r': sag, 'v': deger}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cumle', type=int, default=6000, help='kullanılacak temiz cümle sayısı')
    ap.add_argument('--tur', type=int, default=2, help='her cümle kaç farklı biçimde bozulsun')
    ap.add_argument('--cikti')
    ap.add_argument('--tablo', nargs='*', default=[],
                    help='öznitelik tablosu dosyaları: varsa oradan okunur (birden çoksa birleştirilir), yoksa ilkine yazılır')
    ap.add_argument('--atla', type=int, default=0, help='baştan bu kadar cümleyi atla (işi birkaç sürece bölmek için)')
    ap.add_argument('--tohum', type=int, default=2026)
    ap.add_argument('--bert', action='store_true',
                    help='BERTurk özniteliklerini de ekle (torch + transformers); varsayılan çıktı siralayici_bert.json')
    ap.add_argument('--yalniz-tablo', action='store_true', help='tabloyu üretip çık (eğitme)')
    a = ap.parse_args()
    a.cikti = a.cikti or os.path.join(KOK, 'siralayici_bert.json' if a.bert else 'siralayici.json')
    adlar = T.OZNITELIK_ADLARI + (T.BERT_OZNITELIKLER if a.bert else [])

    if a.tablo and all(os.path.exists(t) for t in a.tablo):
        X, y, grup = [], [], []
        for k, t in enumerate(a.tablo):
            with open(t) as fh:
                Xk, yk, gk = json.load(fh)
            X += [x[:len(adlar)] for x in Xk]      # BERT'li tablodan BERT'siz model de eğitilebilir (aynı veri)
            y += yk
            grup += [g * len(a.tablo) + k for g in gk]
    else:
        import sistemler
        yazim = sistemler.TrDuzelt(siralayici='').yazim   # sıralayıcısız: ilk geçiş + aday üretimi (Zemberek önbelleğiyle)
        if a.bert:
            yazim.bert = T.BertPuan()
        yazim._tablo_yukle(__import__('pkg_resources').resource_filename)
        cumleler = temiz_cumleler('boun-train.conllu', en_az=4, en_cok=40)
        random.Random(7).shuffle(cumleler)
        cumleler = cumleler[a.atla:a.atla + a.cumle]
        print('%d temiz cümle, her biri %d kez bozulacak' % (len(cumleler), a.tur), flush=True)
        X, y, grup = tablo_uret(yazim, cumleler, a.tohum, a.tur)
        if a.tablo:
            with open(a.tablo[0], 'w') as fh:
                json.dump([X, y, grup], fh)
    if a.yalniz_tablo:
        return
    import lightgbm as lgb
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    grup = np.asarray(grup)
    print('%d satır, %d öznitelik, doğru aday oranı %%%.1f' % (len(X), X.shape[1], 100.0 * y.mean()))
    dogrulama = (grup % 10 == 0)             # cümlelerin onda biri erken durdurma için
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                               verbose=-1, n_jobs=2)
    model.fit(X[~dogrulama], y[~dogrulama], eval_set=[(X[dogrulama], y[dogrulama])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    p = model.predict_proba(X[dogrulama])[:, 1]
    yd = y[dogrulama]
    for esik in (0.3, 0.5, 0.7, 0.9):
        sec = p >= esik
        print('  eşik %.1f: seçilen satırların %%%.1f\'i doğru, doğru satırların %%%.1f\'i seçildi' % (
            esik, 100.0 * yd[sec].mean() if sec.any() else 0.0, 100.0 * sec[yd == 1].mean()))
    dok = model.booster_.dump_model(num_iteration=model.best_iteration_)
    agaclar = [agac_duzlestir(a_['tree_structure']) for a_ in dok['tree_info']]
    onem = sorted(zip(model.booster_.feature_importance('gain'), adlar), reverse=True)
    print('en etkili öznitelikler:', ', '.join(ad for _, ad in onem[:12]))
    with open(a.cikti, 'w', encoding='utf-8') as fh:
        json.dump({'oznitelikler': adlar, 'esik': 0.5, 'agaclar': agaclar}, fh)
    # dökülen modelin saf Python ile aynı sonucu verdiğini doğrula
    s = T.Siralayici(a.cikti)
    ornek = X[dogrulama][:200].tolist()
    fark = max(abs(q - r) for q, r in zip(s.olasilik(ornek), p[:200]))
    print('%d ağaç -> %s (%.1f MB); saf Python ile en büyük fark %.2e' % (
        len(agaclar), a.cikti, os.path.getsize(a.cikti) / 1e6, fark))


if __name__ == '__main__':
    main()
