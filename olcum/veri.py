#!/usr/bin/env python3
"""Sabit ölçüm setleri: her yöntem AYNI veride ölçülsün diye.

  gercek-gelistirme, gercek-test : #Turki$hTweets (Koksal vd., 2020). Gerçek kullanıcıların yazdığı, elle düzeltilmiş
                                   2.000 tweet. İlk 1.000'i ayar yapmak, son 1.000'i sonuç bildirmek için.
  yapay-*                        : UD Turkish-IMST'nin test bölümündeki temiz cümlelerin sabit tohumla bozulmuş hâlleri.

Her set (girdi kelimeleri, doğru kelimeler) çiftlerinden oluşur; hepsi küçük harfli ve noktalamasızdır, çünkü burada
yalnızca yazım/normalleştirme aşaması ölçülür. Veriler depoya konmaz (lisansları ayrı); ilk çalıştırmada GitHub'dan
olcum/veri/ klasörüne indirilir.
"""
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tr_duzelt import tr_lower, _KLAVYE, _ASCII      # noqa: E402

VERI = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'veri')
KAYNAK = {
    'tweetler.txt': 'https://raw.githubusercontent.com/atubakoksal/annotated_tweets/HEAD/data.txt',
    'imst-test.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/HEAD/tr_imst-ud-test.conllu',
    'imst-dev.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/HEAD/tr_imst-ud-dev.conllu',
    'boun-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/HEAD/tr_boun-ud-train.conllu',
}
_KELIME = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)


def dosya(ad):
    """Veri dosyasının yolunu döndür; yoksa indir."""
    yol = os.path.join(VERI, ad)
    if not os.path.exists(yol):
        os.makedirs(VERI, exist_ok=True)
        print('indiriliyor: %s' % KAYNAK[ad], file=sys.stderr)
        urllib.request.urlretrieve(KAYNAK[ad], yol)
    return yol


def sade(kelime):
    """Karşılaştırma biçimi: küçük harf, kesme işareti yok. (U+0307: 'İ' harfinin hatalı küçültülmesinden kalan nokta.)"""
    return tr_lower(kelime).replace("'", '').replace('’', '').replace('\u0307', '')


# ---------------------------------------------------------------- gerçek veri
def tweetler():
    """[(girdi kelimeleri, doğru kelimeler, etiketler)] — etiketler girdiyle aynı uzunluktaysa dolu, değilse None."""
    out = []
    with open(dosya('tweetler.txt'), encoding='utf-8-sig') as f:
        next(f)                                    # başlık
        for satir in f:
            alan = satir.rstrip('\n').split('\t')
            if len(alan) < 4:
                continue
            # Veri kümesi girdi ile düzeltilmişi kelime kelime eşli tutmak için '|' kullanıyor:
            # girdide 'bir|çok' = kullanıcı "bir çok" yazmış; düzeltilmişte 'tabii|ki' = doğrusu "tabii ki".
            ham = alan[1].split()
            etiket = alan[-1].split()
            if len(etiket) != len(ham):
                etiket = None
            girdi, girdi_etiket = [], []
            for k, t in enumerate(ham):
                for parca in t.split('|'):
                    parca = sade(parca)
                    if parca:
                        girdi.append(parca)
                        girdi_etiket.append(etiket[k] if etiket else None)
            dogru = [t for t in (sade(x) for x in alan[2].replace('|', ' ').split()) if t]
            if girdi and dogru:
                out.append((girdi, dogru, girdi_etiket if etiket else None))
    return out


# ---------------------------------------------------------------- yapay veri
def temiz_cumleler(ad, en_az=6, en_cok=25):
    out = []
    with open(ad if os.path.exists(ad) else dosya(ad), encoding='utf-8') as f:     # yol da verilebilir
        for satir in f:
            if satir.startswith('# text = '):
                ws = [sade(w) for w in _KELIME.findall(satir[9:])]
                if en_az <= len(ws) <= en_cok and all(w.isalpha() for w in ws):
                    out.append(ws)
    return out


HARFLER = 'abcçdefgğhıijklmnoöprsştuüvyz'
KOMSU = {}
for _p in sorted(tuple(sorted(p)) for p in _KLAVYE):
    KOMSU.setdefault(_p[0], []).append(_p[1])
    KOMSU.setdefault(_p[1], []).append(_p[0])
INGILIZCE = ["here we go again", "oh my god", "what the hell", "game over", "by the way", "no problem",
             "thank you", "good job", "lets go", "level up", "ow shit man", "never mind", "see you",
             "take it easy", "come on", "nice", "sorry", "okay", "bro", "update", "feedback", "meeting",
             "deadline", "online", "stream", "follow", "story", "check", "team", "random", "fail", "wow",
             "why not", "of course", "good luck", "well done", "who cares", "best friend", "happy birthday",
             "download", "upload", "screenshot", "playlist", "weekend", "crush", "mood", "cringe", "spoiler"]


class Bozucu:
    """Temiz cümleyi insan hatalarına benzeyen biçimde bozar. gecerli(w): kelime Türkçe mi (Zemberek)."""

    def __init__(self, gecerli, tohum):
        self.gecerli = gecerli
        self.rng = random.Random(tohum)

    def _boz(self, w):
        rng = self.rng
        r = rng.random()
        i = rng.randrange(len(w))
        if r < 0.30 and w[i] in KOMSU:                                  # komşu tuş
            return w[:i] + rng.choice(KOMSU[w[i]]) + w[i + 1:]
        if r < 0.55:                                                    # eksik harf
            return w[:i] + w[i + 1:]
        if r < 0.75 and i + 1 < len(w) and w[i] != w[i + 1]:            # yer değiştirme
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if r < 0.88:                                                    # fazla harf
            return w[:i] + rng.choice(HARFLER) + w[i:]
        a = w.translate(_ASCII)                                         # Türkçe karaktersiz yazım
        return a if a != w else w[:i] + w[i + 1:]

    def yazim_hatasi(self, w):
        """Geçerli bir kelime olmayan bozuk biçim; bulunamazsa None."""
        for _ in range(10):
            b = self._boz(w)
            if b != w and not self.gecerli(b):
                return b
        return None

    def bol(self, w):
        k = (self.rng.choice([len(w) - 2, len(w) - 3, self.rng.randrange(2, len(w) - 1)])
             if len(w) >= 6 else self.rng.randrange(2, len(w) - 1))
        return w[:k], w[k:]

    def ingilizce(self):
        return self.rng.choice(INGILIZCE).split()


def yapay_setler(cumleler, gecerli, tohum=123, n=600):
    """{set adı: [(girdi, doğru)]}"""
    b = Bozucu(gecerli, tohum)
    rng = b.rng
    cumleler = list(cumleler)
    rng.shuffle(cumleler)
    cumleler = cumleler[:n]
    setler = {k: [] for k in ('yapay-tek-hata', 'yapay-uc-hata', 'yapay-bolunmus', 'yapay-bolunmus-iki-kelime',
                              'yapay-ingilizce', 'yapay-temiz', 'yapay-yogun')}
    for ws in cumleler:
        uygun = [i for i, w in enumerate(ws) if len(w) >= 4 and gecerli(w)]
        if uygun:                                                       # tek yazım hatası
            i = rng.choice(uygun)
            h = b.yazim_hatasi(ws[i])
            if h:
                setler['yapay-tek-hata'].append((ws[:i] + [h] + ws[i + 1:], ws))
        if len(uygun) >= 3:                                             # aynı cümlede üç hata
            yeni = list(ws)
            for i in rng.sample(uygun, 3):
                h = b.yazim_hatasi(ws[i])
                if h:
                    yeni[i] = h
            if yeni != ws:
                setler['yapay-uc-hata'].append((yeni, ws))
        uzun = [i for i, w in enumerate(ws) if len(w) >= 5 and gecerli(w)]
        if uzun:                                                        # ortasından bölünmüş kelime
            i = rng.choice(uzun)
            a, c = b.bol(ws[i])
            ad = 'yapay-bolunmus-iki-kelime' if (gecerli(a) and gecerli(c)) else 'yapay-bolunmus'
            setler[ad].append((ws[:i] + [a, c] + ws[i + 1:], ws))
        ing = b.ingilizce()                                             # araya giren İngilizce: dokunulmamalı
        k = rng.randrange(0, len(ws) + 1)
        yeni = ws[:k] + ing + ws[k:]
        setler['yapay-ingilizce'].append((yeni, yeni))
        setler['yapay-temiz'].append((ws, ws))                          # temiz metin: dokunulmamalı
        girdi, dogru = [], []                                           # yoğun gürültü: hepsi bir arada (issue #2)
        bolunecek = rng.choice(uzun) if uzun and rng.random() < 0.5 else -1
        for i, w in enumerate(ws):
            if i == bolunecek:
                girdi += list(b.bol(w))
            elif i in uygun and rng.random() < 0.3:
                girdi.append(b.yazim_hatasi(w) or w)
            else:
                girdi.append(w)
            dogru.append(w)
        if rng.random() < 0.5:
            ing = b.ingilizce()
            k = rng.randrange(0, len(ws) + 1)
            # eklenen parça girdi ve doğruda aynı yere gelsin diye kelime sınırını doğru üzerinden bul
            gk = sum(2 if j == bolunecek else 1 for j in range(k))
            girdi = girdi[:gk] + ing + girdi[gk:]
            dogru = dogru[:k] + ing + dogru[k:]
        setler['yapay-yogun'].append((girdi, dogru))
    return setler


def butun_setler(gecerli):
    tw = tweetler()
    setler = {'gercek-gelistirme': [(g, d) for g, d, _ in tw[:1000]],
              'gercek-test': [(g, d) for g, d, _ in tw[1000:]]}
    setler.update(yapay_setler(temiz_cumleler('imst-test.conllu'), gecerli))
    return setler
