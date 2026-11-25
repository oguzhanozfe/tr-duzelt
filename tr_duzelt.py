#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tr_duzelt.py — LLM kullanmadan Türkçe metin düzeltme (klasik NLP + küçük ML)

Katmanlar (her biri ayrı, klasik bir teknik):
  1) Yazım düzeltme  : gürültülü kanal modeli. Adaylar Zemberek'in morfolojik çözümleyicisi ve
                       yazım denetleyicisinden, konuşma dili kurallarından (gidiyodum -> gidiyordum),
                       sık-yanlış tablosundan ve kelime ayırmadan (vebeni -> ve beni) gelir;
                       seçim, 2-gram dil modeli üzerinde Viterbi ile yapılır.
  2) Kurallar        : sayı ekleri (2e -> 2'ye), tarih (15 temmuz -> 15 Temmuz), ASCII metinde
                       Türkçe karakterleri geri koyma (deasciifier).
  3) Noktalama + büyük harf : CRF dizi etiketleyici (egit_noktalama.py ile eğitilir). Öznitelikler:
                       kelime/sonek + dil modelinin "bu kelimeden sonra nokta/virgül gelir mi"
                       olasılıkları ("gizli olay" dil modeli yaklaşımı).
  4) Özel isim       : Zemberek çözümlemesi -> büyük harf + kesme işareti (trabzona -> Trabzon'a)
  5) Paragraf        : TextTiling benzeri; DENEYSEL ve varsayılan kapalı (ölçümde işe yaramadı).

Ölçülen başarım (ayrıntı için egit_noktalama.py ve koddaki notlar):
  yazım     : tek harflik yapay hataların ~%92'si geri geliyor; temiz metinde kelimelerin ~%0,3'üne dokunuyor
  nokta     : F1 0,81 (gazete) / 0,76 (roman) / 0,66 (öğrenci yazısı)      virgül: F1 0,2-0,3 (zayıf)
  yapamadığı: yazımı doğru ama anlamca yanlış kelimeler (orda/orta, telefonumda/telefonum da), paragraf

Kurulum (Python 3.11 ve 3.12 ile denendi; zemberek-python NumPy 2 ve Python 3.13+ ile çalışmıyor):
  pip install -r requirements.txt          # Windows'ta yönetici hakkı olmadan kurulum: README.md

Kullanım:
  python tr_duzelt.py "düzeltilecek metin"                  # noktalama.crfsuite aynı klasörde olmalı
  python tr_duzelt.py --dosya girdi.txt --ayrinti           # yapılan kelime düzeltmelerini de göster
  python tr_duzelt.py --dosya girdi.txt --cikti cikti.txt   # sonucu UTF-8 dosyaya yaz
  python tr_duzelt.py --dosya girdi.txt --eval              # Gemini ile isteğe bağlı, uzaktan değerlendirme

  from tr_duzelt import Duzeltici
  d = Duzeltici('noktalama.crfsuite')       # yükleme ~15 sn (dil modeli + morfoloji)
  print(d.duzelt('herkez geldimi bilmiyorum'))

Not: zemberek-python saf Python olduğu için yavaş (kelime başına ~10-40 ms). Aynı mantık Java
Zemberek ile (özgün kütüphane) çok daha hızlı çalışır. Bu modül içe aktarılınca `logging`i susturur.
"""
import os
import re
import sys
import json
import math
import logging
import argparse
import warnings

from tr_eval import DEFAULT_MODEL, EvaluationError, evaluate_correction

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)   # zemberek-python her yüklemede bolca log basıyor

# ==========================================================================
# Ortak araçlar
# ==========================================================================

def tr_lower(s):
    return s.replace('İ', 'i').replace('I', 'ı').lower()


def tr_upper(s):
    return s.replace('i', 'İ').replace('ı', 'I').upper()


def tr_capitalize(s):
    return tr_upper(s[:1]) + s[1:] if s else s


WORD_RE = re.compile(
    r"\d+(?:[.,:/]\d+)*(?:['’]?[^\W\d_]+)?(?![^\W_])"   # sayı (+ ek): 2e, 2'ye, 48.7, 1994'te
    r"|[^\W_]+(?:['’][^\W\d_]+)*"                        # kelime (mp3, covid19 dahil), kesmeli olabilir
)
TOKEN_RE = re.compile(WORD_RE.pattern + r"|\.\.\.|…|[^\w\s]")
CUMLE_SONU = re.compile(r"[.!?…]")


def tokenize(text):
    """Kelimeler + noktalama (eğitim verisi hazırlarken kullanılır)."""
    return TOKEN_RE.findall(text)


def is_word(tok):
    return any(ch.isalnum() for ch in tok)


# ==========================================================================
# Dil modeli: Zemberek paketiyle gelen sıkıştırılmış 2-gram model
# (küçük harfli; '.', ',', '?' ve <s>, </s> belirteçlerini de içeriyor)
# ==========================================================================

_KESME_EKI = re.compile(r"^(?:n?[ıiuü]n|y?[ıiuü]|y?[ae]|[dt][ae]n?|[dt][ae]ki|y?l[ae]|l[ae]r[a-zçğıöşü]*|s[ıiuü][a-zçğıöşü]*)$")


class DilModeli:
    def __init__(self):
        import numpy as np
        from pkg_resources import resource_filename
        from zemberek.lm import SmoothLM
        yol = resource_filename('zemberek', os.path.join('resources', 'lm.2gram.slm'))
        self.lm = SmoothLM.builder(yol).log_base(np.e).build()
        self.V = self.lm.vocabulary
        self._i, self._p = {}, {}
        self._ozel = None

    def idx(self, w):
        i = self._i.get(w)
        if i is None:
            i = self._i[w] = self.V.index_of(w)
        return i

    def lp(self, a, b):
        """log P(b | a), doğal logaritma."""
        k = (a, b)
        v = self._p.get(k)
        if v is None:
            v = self._p[k] = float(self.lm.get_probability((self.idx(a), self.idx(b))))
        return v

    def var(self, a, b):
        """(a, b) ikilisi eğitim derleminde gerçekten görülmüş mü?"""
        ia, ib = self.idx(a), self.idx(b)
        return bool(ia > 0 and ib > 0 and self.lm.ngram_exists((ia, ib)))

    def tekli(self, w):
        """log P(w); dağarcıkta yoksa -inf."""
        i = self.idx(w)
        return float(self.lm.get_unigram_probability(i)) if i > 0 else float('-inf')

    @property
    def ozel_isimler(self):
        """Derlemde özel isim gibi kullanılmış gövdeler: dağarcıkta en az iki farklı kesmeli çekimi
        olanlar (trabzon'a, trabzon'da ...). Zemberek'in özel isim sözlüğü çok geniş (we, go, na, ra
        bile var); büyük harf kararını bu kümeyle doğruluyoruz."""
        if self._ozel is None:
            from collections import Counter
            say = Counter()
            for w in self.V.vocabulary:
                if "'" in w:
                    govde, _, ek = w.partition("'")
                    if len(govde) > 1 and govde.isalpha() and _KESME_EKI.match(ek):
                        say[govde] += 1
            self._ozel = frozenset(g for g, c in say.items() if c >= 2)
        return self._ozel


# ==========================================================================
# CRF öznitelikleri — eğitim ve çıkarımda AYNI fonksiyon kullanılmalı.
# Girdi: küçük harfli, noktalamasız, kesme işaretsiz kelime listesi.
# ==========================================================================

def _base_feats(w):
    n = len(w)
    f = ['w=' + w]
    if w[0].isdigit():
        f.append('num')
        f.append('numlen=%d' % min(n, 5) if w.isdigit() else 'nummix')
        return f
    f.append('s1=' + w[-1])
    for k in (2, 3, 4, 5, 6):
        if n > k:
            f.append('s%d=%s' % (k, w[-k:]))
    for k in (3, 4, 5):
        if n > k:
            f.append('p%d=%s' % (k, w[:k]))
    f.append('len=%d' % min(n, 12))
    return f


_ORAN_DILIM = (-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8)
_OLAS_DILIM = (-9, -7, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5)


def _dilim(x, sinirlar):
    for k, t in enumerate(sinirlar):
        if x < t:
            return k
    return len(sinirlar)


def _lm_bosluk_feats(lm, w, sonraki):
    """w ile sonraki kelime arasındaki boşluk için "gizli olay" dil modeli kanıtı:
       hiçbir şey (O) / virgül (C) / nokta (P) / soru işareti (Q) koymanın olasılıkları."""
    pP, pC, pQ = lm.lp(w, '.'), lm.lp(w, ','), lm.lp(w, '?')
    f = ['pP=%d' % _dilim(pP, _OLAS_DILIM), 'pC=%d' % _dilim(pC, _OLAS_DILIM), 'pQ=%d' % _dilim(pQ, _OLAS_DILIM)]
    if lm.idx(w) == 0:
        f.append('lm_oov')
    best = None
    if sonraki is not None:
        aO = lm.lp(w, sonraki)
        sN = lm.lp('<s>', sonraki)
        aP = pP + sN
        aQ = pQ + sN
        aC = pC + lm.lp(',', sonraki)
        f += ['dP=%d' % _dilim(aP - aO, _ORAN_DILIM), 'dC=%d' % _dilim(aC - aO, _ORAN_DILIM),
              'dQ=%d' % _dilim(aQ - aO, _ORAN_DILIM), 'dPC=%d' % _dilim(aP - aC, _ORAN_DILIM),
              'sN=%d' % _dilim(sN, _OLAS_DILIM)]
        best = max((aO, 'O'), (aC, 'C'), (aP, 'P'), (aQ, 'Q'))[1]
        f.append('lmbest=' + best)
        if lm.idx(sonraki) == 0:
            f.append('lm_oov+1')
    else:
        best = 'Q' if pQ > pP else 'P'
    return f, best


def lm_tahmin(words, lm):
    """Yalnızca dil modeliyle (CRF'siz) noktalama tahmini — karşılaştırma tabanı."""
    out = []
    for i, w in enumerate(words):
        _, best = _lm_bosluk_feats(lm, w, words[i + 1] if i + 1 < len(words) else None)
        out.append(best)
    return out


_UZAKLIK_DILIM = (1, 2, 3, 4, 5, 7, 10, 15)
UZAK_KELIME = False     # ±2 konumdaki kelime kimliği: ezberletiyor, düzenlileştirince katkısı kalmıyor
GUCLU_BITIS = -1.2      # log P('.' | kelime) bunun üstündeyse "güçlü cümle bitirici" (çekimli fiil vb.)


def crf_features(words, lm=None):
    n = len(words)
    base = [_base_feats(w) for w in words]
    lmf = [None] * n
    uzak = [None] * n
    if lm is not None:
        for i, w in enumerate(words):
            lmf[i] = _lm_bosluk_feats(lm, w, words[i + 1] if i + 1 < n else None)[0]
        # Zincir CRF uzak bağlamı göremez; "en son güçlü cümle bitiriciden beri kaç kelime geçti /
        # bir sonrakine kaç kelime var" bilgisini gözlemden türetip öznitelik olarak veriyoruz.
        pP = [lm.lp(w, '.') for w in words]
        guclu = [p > GUCLU_BITIS for p in pP]
        son = -1
        beri = []
        for i in range(n):
            beri.append(i - son)
            if guclu[i]:
                son = i
        sonraki = n
        kala = [0] * n
        for i in range(n - 1, -1, -1):
            if guclu[i]:
                sonraki = i
            kala[i] = sonraki - i
        for i in range(n):
            b, k = _dilim(beri[i], _UZAKLIK_DILIM), _dilim(kala[i], _UZAKLIK_DILIM)
            ileri = max(pP[i + 1:i + 4]) - pP[i] if i + 1 < n else -99.0
            uzak[i] = ['beri=%d' % b, 'kala=%d' % k, 'beri|kala=%d|%d' % (min(b, 4), min(k, 3)),
                       'ileri=%d' % _dilim(ileri, _ORAN_DILIM)]
    X = []
    for i, w in enumerate(words):
        f = ['bias'] + base[i]
        if i == 0:
            f.append('BOS')
        else:
            p = words[i - 1]
            f += ['-1:w=' + p, '-1:s3=' + p[-3:], '-1:s2=' + p[-2:], '-1|0=' + p + '|' + w]
        if i > 1 and UZAK_KELIME:
            f.append('-2:w=' + words[i - 2])
        if i == n - 1:
            f.append('EOS')
        else:
            x = words[i + 1]
            f += ['+1:w=' + x, '+1:s3=' + x[-3:], '+1:s2=' + x[-2:], '+1:p3=' + x[:3],
                  '0|+1=' + w + '|' + x,
                  's3|+1w=' + w[-3:] + '|' + x,
                  's3|+1s3=' + w[-3:] + '|' + x[-3:]]
        if i < n - 2:
            y = words[i + 2]
            f.append('+2:s3=' + y[-3:])
            if UZAK_KELIME:
                f.append('+2:w=' + y)
        if lm is not None:
            f += lmf[i] + uzak[i]
            best = [a for a in lmf[i] if a.startswith('lmbest')]
            if best:
                f.append(best[0] + '|' + uzak[i][0])
            if i > 0:
                f += ['-1:' + a for a in lmf[i - 1] if a[0] == 'd' or a.startswith('lmbest')]
            if i < n - 1:
                f += ['+1:' + a for a in lmf[i + 1] if a.startswith('dP') or a.startswith('lmbest')]
        X.append(f)
    return X


def crf_coz(tagger, oznitelikler, nokta_esik=0.5, virgul_esik=0.5):
    """Sonsal (posterior) çözümleme: her kelimede noktalama olasılıklarını topla, eşikle karar ver.
    Eşiği yükseltmek = daha az ama daha isabetli işaret (özellikle virgülde işe yarar).
    Döndürür: (etiketler, büyük harf olasılıkları)"""
    tagger.set(oznitelikler)
    etiketler = tagger.labels()
    out, buyuk = [], []
    for i in range(len(oznitelikler)):
        pn = {'O': 0.0, 'C': 0.0, 'P': 0.0, 'Q': 0.0}
        ph = {'l': 0.0, 'c': 0.0, 'u': 0.0}
        for e in etiketler:
            m = tagger.marginal(e, i)
            pn[e[1]] += m
            ph[e[0]] += m
        if pn['P'] + pn['Q'] >= nokta_esik:
            n = 'Q' if pn['Q'] > pn['P'] else 'P'
        elif pn['C'] >= virgul_esik:
            n = 'C'
        else:
            n = 'O'
        out.append(max(ph, key=ph.get) + n)
        buyuk.append(ph['c'] + ph['u'])
    return out, buyuk


# ==========================================================================
# Kural: sayılara gelen ekler  (2e -> 2'ye, 5de -> 5'te, 1994de -> 1994'te)
# ==========================================================================

_BIRLER = ['sıfır', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz']
_ONLAR = ['', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan']
_SAYI_EK = re.compile(r"^(\d+(?:[.,:/]\d+)*)['’]?([a-zçğıöşü]+)$")
_KALIN, _INCE = 'aıou', 'eiöü'


def _sayi_son_okunus(rakamlar):
    n = rakamlar.lstrip('0') or '0'
    if n == '0':
        return 'sıfır'
    z = len(n) - len(n.rstrip('0'))
    if z == 0:
        return _BIRLER[int(n[-1])]
    if z == 1:
        return _ONLAR[int(n[-2])]
    if z == 2:
        return 'yüz'
    if z <= 5:
        return 'bin'
    if z <= 8:
        return 'milyon'
    return 'milyar'


def sayi_eki_duzelt(tok):
    """'2e' -> "2'ye". Tanınmayan / ses uyumuna uymayan ek varsa None (dokunma)."""
    m = _SAYI_EK.match(tok)
    if not m:
        return None
    sayi, ek = m.group(1), m.group(2)
    okunus = _sayi_son_okunus(re.sub(r'\D', '', sayi))
    son_unlu = [c for c in okunus if c in _KALIN + _INCE][-1]
    unlu_biter = okunus[-1] in _KALIN + _INCE
    A = 'a' if son_unlu in _KALIN else 'e'
    I = {'a': 'ı', 'ı': 'ı', 'e': 'i', 'i': 'i', 'o': 'u', 'u': 'u', 'ö': 'ü', 'ü': 'ü'}[son_unlu]
    D = 't' if okunus[-1] in 'çfhkpsşt' else 'd'
    y = 'y' if unlu_biter else ''
    n = 'n' if unlu_biter else ''
    kaliplar = [
        (r'y?([ae])', y + A, A),                                   # yönelme  : 2'ye, 3'e, 6'ya
        (r'[dt]([ae])', D + A, A),                                 # bulunma  : 2'de, 3'te
        (r'[dt]([ae])n', D + A + 'n', A),                          # ayrılma  : 2'den, 3'ten
        (r'[dt]([ae])ki', D + A + 'ki', A),                        #           : 2'deki
        (r'y?([ıiuü])', y + I, I),                                 # belirtme : 2'yi, 3'ü
        (r'n?([ıiuü])n', n + I + 'n', I),                          # ilgi     : 2'nin, 3'ün
        (r'y?l([ae])', y + 'l' + A, A),                            # araç     : 2'yle
        (r'l([ae])r', 'l' + A + 'r', A),                           # çoğul    : 90'lar
        (r'l([ıiuü])', 'l' + I, I),                                # -li      : 2'li
        (r'l([ıiuü])k', 'l' + I + 'k', I),                         # -lik     : 100'lük
        (r'[ıiuü]?nc([ıiuü])', ('' if unlu_biter else I) + 'nc' + I, I),   # sıra: 2'nci, 3'üncü
    ]
    for kalip, dogru, beklenen_unlu in kaliplar:
        mm = re.fullmatch(kalip, ek)
        if mm and mm.group(1) == beklenen_unlu:      # ses uyumu tutuyorsa gerçekten ek
            return sayi + "'" + dogru
    return None


# ==========================================================================
# 1) YAZIM DÜZELTME — gürültülü kanal:  argmax  P(aday | bağlam) · P(yazılan | aday)
# ==========================================================================

# Kanal maliyetleri (-log olasılık, nat). Dil modeli log-olasılığıyla toplanır.
C_BILINMEYEN = 10.0     # morfolojik çözümü olmayan kelimeyi olduğu gibi bırakmak
C_BILINMEYEN_LM = 4.0   # ...ama dil modelinin dağarcığında varsa (derlemde sık geçen biçim) daha ucuz
C_AKSAN_TAVAN = 1.6     # yalnızca Türkçe karakter farkı (tartismayi -> tartışmayı) için üst sınır
C_SOZLUK = 0.3          # elle hazırlanmış sık-yanlışlar tablosu
C_BILINEN_YANLIS = 6.0  # tabloda "yanlış" diye geçen biçimi (orjinal, herkez) aynen bırakmanın bedeli
C_KONUSMA = 0.8         # konuşma dili -> yazı dili (gidiyodum -> gidiyordum)
C_OZEL_GORULMEMIS = 3.0  # yalnızca özel isim olarak "geçerli" olup büyük derlemde hiç geçmeyen kelimeyi aynen bırakmak (yaaa)
# Aşağıdaki iki kaynak gürültülü olduğu için yalnızca öğrenilmiş sıralayıcı (siralayici.json) varken kullanılır:
C_TABLO = 1.5           # Zemberek'in derlemden çıkardığı tablodaki karşılık (msj -> mesaj; ama tmm -> "tam" da var)
C_AKSAN_GECERLI = 2.0   # geçerli bir kelimenin Türkçe karakterli eşi (aksam -> akşam): ancak cümle ASCII yazılmışsa doğru
C_AYIR_EK = 3.0         # geçerli bir kelimenin sonundaki de/da/ki'yi ayırmak (bende -> ben de)
C_AYIR = 2.5            # eksik boşluk (vebeni -> ve beni)
C_OZEL = 1.0            # özel isim adayına ek ceza
# --- Temkin ayarları: emin olunmayan kelime olduğu gibi bırakılır (issue #2) ---
KISA_SINIR = 3          # bu uzunluğa kadarki bilinmeyen kelimeye düzenleme mesafesiyle dokunma (ler, du, ow)
YABANCI_HARF = frozenset('qwx')   # Türk alfabesinde olmayan harf: kelime büyük olasılıkla yabancı
OZEL_ADAY_ESIK = -12.5  # yalnızca özel isim olarak çözümlenen adayı kabul için en düşük log-sıklık
TEK_ADAY_UZUNLUK = 7    # dağarcıkta olmayan (nadir çekimli) adaya ancak kelime uzunsa ve aday tekse güven
GUVEN_KISA = 0.85       # 4-5 harfli kelimede 'yazım' düzeltmesini kabul için gereken sonsal olasılık
GUVEN_UZUN = 0.55       # 6+ harfli kelimede
KEYFI_DEGISIKLIK = 2.6  # bu maliyetten itibaren değişiklik yazım hatasına benzemez; bağlam kanıtı istenir
C_ILK_HARF = 1.0        # ilk harfi değiştiren düzeltmeye ek ceza: insanlar ilk harfi nadiren yanlış yazar (cringe -> "ringe" olmasın)
C_BIRLESTIR = 1.5       # bölünmüş kelimeyi birleştirmek; parçalardan biri tek başına kelime değilken (orma na)
# İki parça da geçerli kelimeyken (kaybet tik) birleştirmek için dil modelinin en az bu kadar (nat) kazanması gerekir.
# Ölçüm (40.013 kelimelik temiz metin; 450 yapay bölünme): eşik 8 -> 7 yanlış birleştirme, %84 yakalama;
# eşik 10 -> 3 "yanlış" (üçü de aslında doğru: arzu su, din le, büyükler den), %80 yakalama; eşik 12 -> 2, %71.
C_BIRLESTIR_GECERLI = 10.0
# GEÇERLİ bir kelimeyi bağlam yüzünden başka bir geçerli kelimeyle değiştirmek (yaşam -> yaşım) en
# riskli iş: yanlış yapılırsa anlam sessizce değişir. Ölçüm (5.681 kelimelik temiz metin + 603 yapay
# gerçek-kelime hatası; kazanç = dil modeli log-olasılık farkı, nat):
#   kazanç >= 6                      -> temiz metnin %0,77'sini bozuyor, hataların %48'ini yakalıyor
#   kazanç >= 8                      -> %0,14 bozuyor, %37 yakalıyor
#   kazanç >= 10 + aday iki yanındaki kelimeyle de derlemde görülmüş -> 0/5.681 bozuyor, %20 yakalıyor
# Varsayılan sonuncusu. Daha fazlasını yakalamak için eşiği düşürün; bedeli yukarıda.
GERCEK_KELIME_ESIK = 10.0

# TDK'nin "sıkça yapılan yanlışlar" listesi türünden gövde düzeltmeleri (yanlış -> doğru).
# Gövde olarak uygulanır: herkeze -> herkese, orjinalini -> orijinalini, farkettim -> fark ettim
GOVDE_DUZELT = {
    'herkez': 'herkes', 'yanlız': 'yalnız', 'yalnış': 'yanlış', 'müdahele': 'müdahale',
    'malesef': 'maalesef', 'mağlesef': 'maalesef', 'orjinal': 'orijinal', 'süpriz': 'sürpriz',
    'şöför': 'şoför', 'kirbit': 'kibrit', 'kiprik': 'kirpik', 'klavuz': 'kılavuz',
    'döküman': 'doküman', 'egzos': 'egzoz', 'eksoz': 'egzoz', 'ünvan': 'unvan',
    'zerafet': 'zarafet', 'insiyatif': 'inisiyatif', 'mütevazi': 'mütevazı',
    'entellektüel': 'entelektüel', 'makina': 'makine', 'meyva': 'meyve',
    'pardesü': 'pardösü', 'eşortman': 'eşofman', 'kareografi': 'koreografi',
    'laboratuar': 'laboratuvar', 'çünki': 'çünkü', 'heralde': 'herhalde', 'antreman': 'antrenman',
    'ambülans': 'ambulans', 'dinazor': 'dinozor', 'acaip': 'acayip',
    'tenefüs': 'teneffüs', 'traş': 'tıraş', 'şarz': 'şarj',
    'mahçup': 'mahcup', 'tesbit': 'tespit', 'mahsül': 'mahsul', 'kurdela': 'kurdele',
    'motorsiklet': 'motosiklet', 'müsade': 'müsaade', 'müsvette': 'müsvedde',
    'pantalon': 'pantolon', 'penbe': 'pembe', 'proğram': 'program', 'restorant': 'restoran',
    'sandöviç': 'sandviç', 'sezeryan': 'sezaryen', 'ukela': 'ukala', 'bayağa': 'bayağı',
    'bugünki': 'bugünkü', 'dünki': 'dünkü',
    'farket': 'fark et', 'terket': 'terk et', 'yardımet': 'yardım et', 'teşekküret': 'teşekkür et',
    'birşey': 'bir şey', 'herşey': 'her şey', 'hiçbirşey': 'hiçbir şey', 'hergün': 'her gün',
    'herzaman': 'her zaman', 'pekçok': 'pek çok', 'hiçkimse': 'hiç kimse', 'birsürü': 'bir sürü',
    'şuan': 'şu an', 'dahada': 'daha da',
    # TDK'de her bağlamda ayrı yazılanlar (boş vermek, peş peşe, sağ olmak, bir gün)
    'boşver': 'boş ver', 'peşpeşe': 'peş peşe', 'sağolsun': 'sağ olsun', 'sağolun': 'sağ olun', 'birgün': 'bir gün',
}
# Tam eşleşme: kısaltmalar ve bitişik yazılan bağlaçlar
TAM_DUZELT = {
    'yada': 'ya da', 'yinede': 'yine de', 'hemde': 'hem de', 'bide': 'bir de', 'dimi': 'değil mi',
    'bi': 'bir', 'bişey': 'bir şey', 'bişi': 'bir şey', 'saol': 'sağ ol', 'sağol': 'sağ ol',
    'nası': 'nasıl', 'naber': 'ne haber', 'nolur': 'ne olur', 'noldu': 'ne oldu',
    'napıyorsun': 'ne yapıyorsun', 'napıyosun': 'ne yapıyorsun', 'slm': 'selam', 'mrb': 'merhaba',
    'tmm': 'tamam', 'tşk': 'teşekkürler', 'bn': 'ben', 'sn': 'sen', 'çk': 'çok', 'gzl': 'güzel', 'gbi': 'gibi',
    'nbr': 'ne haber', 'cvp': 'cevap', 'öle': 'öyle', 'böle': 'böyle', 'şöle': 'şöyle', 'sora': 'sonra',
    # zarf/zamir + bağlaç de: bu biçimler bitişik hiçbir zaman doğru değil (bulunma eki almazlar)
    'şimdide': 'şimdi de', 'hiçde': 'hiç de', 'hiçte': 'hiç de', 'öylede': 'öyle de', 'böylede': 'böyle de',
    'şöylede': 'şöyle de', 'sonrada': 'sonra da', 'buradada': 'burada da', 'burdada': 'burada da',
    'oradada': 'orada da', 'ordada': 'orada da',
}
# Geçerli ama konuşma diline ait biçimler: dil modeli de isterse yazı diline çevrilir
KONUSMA_GECERLI = {
    'orda': 'orada', 'burda': 'burada', 'şurda': 'şurada', 'nerde': 'nerede', 'ordan': 'oradan',
    'burdan': 'buradan', 'şurdan': 'şuradan', 'nerden': 'nereden', 'içerde': 'içeride',
    'dışarda': 'dışarıda', 'yukarda': 'yukarıda', 'ilerde': 'ileride',
    'olum': 'oğlum',
}
def _uyumlu_soru_eki(govde, ek):
    """Soru ekinin ünlüsünü gövdenin son ünlüsüne uydur: hanım + mi -> mı, gül + mu -> mü."""
    son = next((h for h in reversed(govde) if h in 'aeıioöuü'), None)
    if son is None:
        return ek
    u = {'a': 'ı', 'ı': 'ı', 'e': 'i', 'i': 'i', 'o': 'u', 'u': 'u', 'ö': 'ü', 'ü': 'ü'}[son]
    return ''.join(u if h in 'ıiuü' else h for h in ek)


_SORU_EKI = re.compile(r'^(.+?)(m[ıiuü](?:y[ıiuü]m|s[ıiuü]n|y[ıiuü]z|s[ıiuü]n[ıiuü]z|d[ıiuü]r|yd[ıiuü][mnk]?)?)$')
KISA_KELIMELER = {'ve', 'de', 'da', 'ki', 'bu', 'şu', 'o', 'ne', 'mi', 'mı', 'mu', 'mü', 'ya', 'en'}

_AKSAN = {frozenset(p) for p in ('cç', 'gğ', 'ıi', 'oö', 'sş', 'uü')}
_ASCII = str.maketrans('çğıöşü', 'cgiosu')
_GOVDE_ASCII = {k: k.translate(_ASCII) for k in GOVDE_DUZELT}
_BENZER = {frozenset(p) for p in ('bp', 'dt', 'gk', 'kğ', 'yğ', 'sz', 'fv', 'ae', 'ei', 'ıu', 'iü',
                                    'ou', 'öü', 'mn', 'cj')}
_KLAVYE = set()
_SATIR = ['qwertyuıopğü', 'asdfghjklşi', 'zxcvbnmöç']
for _r, _row in enumerate(_SATIR):
    for _k, _ch in enumerate(_row):
        if _k + 1 < len(_row):
            _KLAVYE.add(frozenset((_ch, _row[_k + 1])))
        if _r + 1 < len(_SATIR):
            for _j in (_k - 1, _k):
                if 0 <= _j < len(_SATIR[_r + 1]):
                    _KLAVYE.add(frozenset((_ch, _SATIR[_r + 1][_j])))


_TEKRAR = re.compile(r'(.)\1+')


def _uzatmasiz(w, cift_harf=True):
    """Vurgu için uzatılmış harflerin kısaltılmış biçimleri: tatlımmm -> tatlım, yaaa -> ya,
    dikkattt -> dikkat. İki kez yazılmış harf Türkçede olağan olduğu için (dikkat, saat) her tekrar
    için hem tek hem çift harfli biçim denenir; en çok 8 biçim döner.
    cift_harf=False: yalnızca üç ve daha fazla tekrar kısaltılır (mood -> "mod" olmasın)."""
    parcalar, secenekler, son = [], [], 0
    for m in _TEKRAR.finditer(w):
        parcalar.append(w[son:m.start()])
        n = m.end() - m.start()
        secenekler.append((m.group(1), ((2, 1) if cift_harf else (2,)) if n == 2 else (1, 2)))
        son = m.end()
    if not secenekler or len(secenekler) > 3:
        return []
    kuyruk = w[son:]
    out = ['']
    for bas, (harf, sayilar) in zip(parcalar, secenekler):
        out = [o + bas + harf * k for o in out for k in sayilar]
    return [o + kuyruk for o in out if o + kuyruk != w]


def _degistir(a, b):
    p = frozenset((a, b))
    if p in _AKSAN:
        return 0.8
    if p in _BENZER:
        return 1.5
    if p in _KLAVYE:
        return 1.8
    return 2.6


def _ekle_sil(s, k):
    return 1.2 if (k > 0 and s[k - 1] == s[k]) or (k + 1 < len(s) and s[k + 1] == s[k]) else 2.0


_UNLU = frozenset('aeıioöuü')


def _ekle(b, j):
    """Hedefe b[j] harfini eklemenin bedeli. İki ünsüz arasına ünlü eklemek ucuz: yazarken ünlü düşürmek (giriyorm,
    göndersn, kıymetlisni) harf silmekten (giriyor) çok daha sık. Ölçüm: geliştirme tweetlerindeki yanlışlardan."""
    if (b[j] in _UNLU and 0 < j < len(b) - 1 and b[j - 1] not in _UNLU and b[j + 1] not in _UNLU):
        return 1.0
    return _ekle_sil(b, j)


def duzenleme_maliyeti(a, b):
    """Ağırlıklı Damerau-Levenshtein: aksan < ses benzerliği < komşu tuş < rastgele harf."""
    a, b = a.replace("'", ''), b.replace("'", '')
    n, m = len(a), len(b)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + _ekle_sil(a, i - 1)
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + _ekle(b, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            v = D[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else _degistir(a[i - 1], b[j - 1]))
            v = min(v, D[i - 1][j] + _ekle_sil(a, i - 1), D[i][j - 1] + _ekle(b, j - 1))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1] and a[i - 1] != a[i - 2]:
                v = min(v, D[i - 2][j - 2] + 1.5)      # yer değiştirme: zmaan -> zaman
            D[i][j] = v
    return D[n][m]


_TR_HARF = re.compile('[çğıöşü]')
_KAYNAKLAR = ('aynı', 'sözlük', 'konuşma dili', 'aksan', 'yazım', 'tablo', 'ayırma')
OZNITELIK_ADLARI = (
    ['kaynak=' + k for k in _KAYNAKLAR]
    + ['maliyet', 'uzunluk', 'aday_uzunluk', 'aday_kelime_sayisi']
    + ['gecerli', 'duzgun', 'dagarcikta', 'siklik', 'ascii', 'tr_harfli', 'yabanci_harf', 'en_uzun_tekrar',
       'lm_sol', 'lm_sag', 'ikili_sol', 'ikili_sag']
    + ['aday_duzgun', 'aday_dagarcikta', 'aday_siklik', 'aday_lm_sol', 'aday_lm_sag', 'aday_ikili_sol',
       'aday_ikili_sag', 'lm_kazanc', 'siklik_farki']
    + ['sonsal', 'sira', 'aday_sayisi', 'ilk_gecis_secimi']
    + ['mesafe', 'ilk_harf_farkli', 'son_harf_farkli', 'ascii_esit', 'tekrarsiz_esit', 'ayni_harfler']
    + ['cumle_uzunluk', 'cumle_gecersiz_orani', 'cumle_tr_harf_orani', 'cumle_aksan_duzeltme_orani',
       'sol_bilinmeyen', 'sag_bilinmeyen']
    + ['deasciifier_aday', 'deasciifier_ayni'])


def _lev(a, b):
    """Düz (ağırlıksız) Levenshtein mesafesi."""
    onceki = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        simdiki = [i]
        for j, cb in enumerate(b, 1):
            simdiki.append(min(onceki[j] + 1, simdiki[j - 1] + 1, onceki[j - 1] + (ca != cb)))
        onceki = simdiki
    return onceki[-1]


class Siralayici:
    """Öğrenilmiş aday sıralayıcı: gradyan artırmalı karar ağaçları. egit_siralayici.py LightGBM ile eğitip JSON'a
    döker; burada saf Python ile çalıştırılır (çalışma anında ek kütüphane gerekmez)."""

    def __init__(self, yol):
        with open(yol, encoding='utf-8') as f:
            m = json.load(f)
        if m['oznitelikler'] != OZNITELIK_ADLARI:
            raise ValueError('%s bu kod sürümüyle uyumlu değil: egit_siralayici.py ile yeniden eğitin' % yol)
        self.agaclar = [(a['f'], a['t'], a['l'], a['r'], a['v']) for a in m['agaclar']]
        self.esik = float(os.environ.get('TR_DUZELT_ESIK', m.get('esik', 0.5)))

    def olasilik(self, satirlar):
        out = []
        for x in satirlar:
            s = 0.0
            for f, t, sol, sag, deger in self.agaclar:
                d = 0
                while f[d] >= 0:
                    d = sol[d] if x[f[d]] <= t[d] else sag[d]
                s += deger[d]
            out.append(1.0 / (1.0 + math.exp(-s)) if s > -700 else 0.0)
        return out


class YazimDuzeltici:
    def __init__(self, lm, siralayici=None):
        """siralayici: öğrenilmiş aday sıralayıcının (siralayici.json) yolu; yoksa elle ayarlı güven kuralları çalışır."""
        from pkg_resources import resource_filename
        from zemberek import TurkishMorphology, TurkishSpellChecker
        from zemberek.core.turkish import SecondaryPos
        from zemberek.morphology.analysis.informal_analysis_converter import InformalAnalysisConverter
        self.lm = lm
        self.SecondaryPos = SecondaryPos
        self.morph = TurkishMorphology.create_with_defaults()
        # konuşma dili eklerini (-yo, -cam ...) tanıyan ve Türkçe karakter eksikliğine toleranslı çözümleyici
        self.gayriresmi = TurkishMorphology.builder(self.morph.lexicon).use_informal_analysis() \
            .ignore_diacritics_in_analysis_().build()
        self.cevirici = InformalAnalysisConverter(self.morph.word_generator)
        self.denetleyici = TurkishSpellChecker(self.morph)
        # Zemberek'in derlemden çıkardığı çok kelimeli düzeltmeler (bişey -> bir şey ...)
        self.tam = dict(TAM_DUZELT)
        yol = resource_filename('zemberek', os.path.join('resources', 'normalization', 'multi-word-replacements.txt'))
        with open(yol, encoding='utf-8') as f:
            for satir in f:
                if '=' in satir:
                    k, v = satir.split('=', 1)
                    self.tam.setdefault(k.strip(), v.strip())
        self._gecerli, self._duzgun, self._oneri = {}, {}, {}
        self._aksanli, self._aday = {}, {}
        self.tablo = {}
        self.siralayici = None
        if siralayici and os.path.exists(siralayici):
            self.siralayici = Siralayici(siralayici)
            self._tablo_yukle(resource_filename)

    def _tablo_yukle(self, resource_filename):
        """Zemberek'in büyük ve gürültülü derlemlerden çıkardığı "yazılan -> olası doğrusu" tabloları (~470 bin satır:
        benzemıyorsun -> benzemiyorsun, cekiyodu -> çekiyodu, msj -> mesaj). Kesin doğru değiller; yalnızca ADAY
        üretirler, kararı sıralayıcı verir."""
        for ad in ('lookup-from-graph.txt', 'ascii-map.txt', 'candidates-manual.txt'):
            yol = resource_filename('zemberek', os.path.join('resources', 'normalization', ad))
            with open(yol, encoding='utf-8') as f:
                for satir in f:
                    k, esit, v = satir.partition('=')
                    k = k.strip()
                    if esit and k:
                        hedef = self.tablo.setdefault(k, [])
                        for t in v.split(','):
                            t = t.strip()
                            if t and t != k and t not in hedef:
                                hedef.append(t)

    # ---- morfoloji yardımcıları ----
    def gecerli(self, w):
        """Morfolojik çözümü var mı? (Türkçe sondan eklemeli: sözlük listesi yetmez.)"""
        v = self._gecerli.get(w)
        if v is None:
            v = self._gecerli[w] = any(not a.is_unknown() for a in self.morph.analyze(w))
        return v

    def duzgun(self, w):
        """Özel isim / kısaltma dışında bir çözümü var mı?"""
        v = self._duzgun.get(w)
        if v is None:
            v = False
            for a in self.morph.analyze(w):
                if (not a.is_unknown() and not a.is_runtime()
                        and a.item.secondary_pos not in (self.SecondaryPos.ProperNoun, self.SecondaryPos.Abbreviation)):
                    v = True
                    break
            self._duzgun[w] = v
        return v

    def oneriler(self, w):
        """Düzenleme mesafesi 1 içindeki, morfolojik olarak geçerli biçimler."""
        v = self._oneri.get(w)
        if v is None:
            v = []
            for c in self.denetleyici.get_unranked_suggestions(w):
                c = tr_lower(c).replace('’', "'")
                if c != w and c not in v and (not re.search('[âîû]', c) or re.search('[âîû]', w)):
                    v.append(c)
            self._oneri[w] = v
        return v

    def konusma_dili(self, w, cift_harf=True):
        """Döndürür: (konuşma dili adayları, yalnız-aksan adayları)
        gidiyodum -> gidiyordum, istiodum -> istiyordum, gelcem -> geleceğim, tatlımmm -> tatlım
        | tartismayi -> tartışmayı, benım -> benim
        cift_harf=False: iki kez yazılmış harfi uzatma sayma (bkz. _uzatmasiz)."""
        wa = w.translate(_ASCII)
        bicimler = [w]
        for f in (re.sub(r'([ıiuü])o', r'\1yo', w),            # istiodum -> istiyodum
                  re.sub(r'([ıiuü])yc([ae])', r'\1c\2', w),    # yapiycaz -> yapicaz (-> yapacağız)
                  wa,                                            # benım -> benim (yanlış yerde ı)
                  *_uzatmasiz(w, cift_harf)):                    # tatlımmm -> tatlım, yaaa -> ya
            if f not in bicimler:
                bicimler.append(f)
        konusma, aksan = [], []
        for f in bicimler:
            yalniz_aksan = f == w or f == wa
            fa = f.translate(_ASCII)
            if f != w and self.gecerli(f) and f not in (aksan if yalniz_aksan else konusma):
                (aksan if yalniz_aksan else konusma).append(f)
            for a in self.gayriresmi.analyze(f):
                if a.is_unknown():
                    continue
                try:
                    gayriresmi = a.contains_informal_morpheme()
                    if gayriresmi:
                        r = self.cevirici.convert(f, a)
                        yuzeyler = [r.surface] if r is not None else []
                    else:
                        yuzeyler = [r.surface for r in
                                    self.morph.word_generator.generate(item=a.item, morphemes=a.get_morphemes())]
                except Exception:
                    continue
                hedef = konusma if (gayriresmi or not yalniz_aksan) else aksan
                for y in yuzeyler:
                    y = tr_lower(y)
                    if y == w or y in hedef or not self.gecerli(y):
                        continue
                    if not gayriresmi and y.translate(_ASCII) != fa:
                        continue                  # çözümleyicinin ürettiği biçim yalnızca Türkçe karakterde farklı olmalı
                    hedef.append(y)
        aksan = [y for y in aksan if y not in konusma]
        return konusma, aksan

    # ---- aday üretimi ----
    def tanidik(self, c):
        """Aday, dil modelinin dağarcığında var mı (büyük derlemde görülmüş bir biçim mi)?
        Yalnızca özel isim olarak çözümlenenler ayrıca yeterince sık olmalı (bogut, çitmen elensin)."""
        if self.lm.idx(c) == 0:
            return False
        if "'" in c or not self.duzgun(c):
            return self.lm.tekli(c) >= OZEL_ADAY_ESIK
        return True

    def aksan_secenekleri(self, w):
        """Geçerli ve tümüyle ASCII yazılmış bir kelimenin Türkçe karakterli eşleri: aksam -> akşam, isime -> işime."""
        v = self._aksanli.get(w)
        if v is None:
            v = []
            if w == w.translate(_ASCII) and any(ch in 'cgiosu' for ch in w):
                for a in self.gayriresmi.analyze(w):
                    try:
                        if a.is_unknown() or a.contains_informal_morpheme():
                            continue
                        yuzeyler = [r.surface for r in
                                    self.morph.word_generator.generate(item=a.item, morphemes=a.get_morphemes())]
                    except Exception:
                        continue
                    for y in yuzeyler:
                        y = tr_lower(y)
                        if (y != w and y not in v and y.translate(_ASCII) == w
                                and self.duzgun(y) and self.lm.idx(y) > 0):
                            v.append(y)
            self._aksanli[w] = v
        return v

    def adaylar(self, w, genis=False):
        """{aday(lar) demeti: (maliyet, kaynak)}
        genis=True: gürültülü kaynakları da kat (Zemberek tabloları, geçerli kelimelerin aksanlı eşleri).
        Bunlar tek başına zarar veriyor (ölçüm: tablolar 32 doğru düzeltmeye karşı 74 yanlış getirdi); yalnızca
        kararı öğrenilmiş sıralayıcı verirken kullanılır."""
        anahtar = (w, genis)
        if anahtar not in self._aday:
            self._aday[anahtar] = self._adaylar(w, genis)
        return self._aday[anahtar]

    def _adaylar(self, w, genis):
        C = {}

        def ekle(c, maliyet, kaynak):
            c = tuple(c.split()) if isinstance(c, str) else tuple(c)
            if c and (c not in C or maliyet < C[c][0]):
                C[c] = (maliyet, kaynak)

        if not w.isalpha():        # sayı, ya da kesme işaretli kelime (büyük olasılıkla özel isim + ek)
            return {(w,): (0.0, 'aynı')}
        gecerli = self.gecerli(w)
        if gecerli:
            ayni = 0.0 if (self.lm.idx(w) > 0 or self.duzgun(w)) else C_OZEL_GORULMEMIS
        else:
            ayni = C_BILINMEYEN_LM if self.lm.idx(w) > 0 else C_BILINMEYEN
        ekle((w,), ayni, 'aynı')

        olagan = gecerli and self.duzgun(w)      # özel isim / kısaltma dışında bir çözümü var
        if w in self.tam:
            ekle(self.tam[w], C_SOZLUK, 'sözlük')
        wa = w.translate(_ASCII)
        for yanlis, dogru in GOVDE_DUZELT.items():
            tam_eslesme = w.startswith(yanlis)
            if tam_eslesme or (not olagan and wa.startswith(_GOVDE_ASCII[yanlis])):   # sarz -> şarj, hersey -> her şey
                aday = (dogru + w[len(yanlis):]).split()
                if all(self.gecerli(t) for t in aday):
                    ekle(aday, C_SOZLUK + (0.0 if tam_eslesme else 0.8), 'sözlük')
                    # çözümleyici/dil modeli bu yanlışı "geçerli" sayabiliyor (derlemde çok geçtiği için)
                    C[(w,)] = (max(C[(w,)][0], C_BILINEN_YANLIS), 'aynı')
        if w in KONUSMA_GECERLI:
            ekle(KONUSMA_GECERLI[w], C_KONUSMA, 'konuşma dili')
        if olagan:
            if genis:
                for c in self.aksan_secenekleri(w):
                    ekle(c, C_AKSAN_GECERLI, 'aksan')
                # bitişik yazılmış de/da/ki bağlacı (bende -> ben de, oda -> o da): kelime geçerli olduğu için
                # ancak bağlama bakarak karar verilebilir; kararı sıralayıcı verir
                for ek, ayri in (('de', 'de'), ('da', 'da'), ('te', 'de'), ('ta', 'da'), ('ki', 'ki')):
                    govde = w[:-2]
                    if w.endswith(ek) and govde and self.duzgun(govde) and self.lm.idx(govde) > 0:
                        ekle((govde, ayri), C_AYIR_EK, 'ayırma')
            return C

        kisa = len(w) <= KISA_SINIR
        # özel isim olarak geçerli bir kelimede çift harf büyük olasılıkla kelimenin kendisidir (mood, emma)
        konusma, aksan = self.konusma_dili(w, cift_harf=not gecerli)
        for c in konusma:
            if self.duzgun(c) or self.tanidik(c):
                ekle(c, C_KONUSMA, 'konuşma dili')
        for c in aksan:
            # kısa kelimede (du -> dü) ancak aday belirgin biçimde daha sıksa
            if self.tanidik(c) and (not kisa or self.lm.tekli(c) >= self.lm.tekli(w) + 2.0):
                ekle(c, min(duzenleme_maliyeti(w, c), C_AKSAN_TAVAN), 'aksan')
        if genis:
            for c in list(C):
                if len(c) != 1 or C[c][1] not in ('aksan', 'konuşma dili'):
                    continue
                for ek, ayri in (('de', 'de'), ('da', 'da'), ('te', 'de'), ('ta', 'da')):
                    govde = c[0][:-2]
                    if c[0].endswith(ek) and len(govde) >= 2 and self.duzgun(govde) and self.lm.idx(govde) > 0:
                        ekle((govde, ayri), C[c][0] + C_AYIR_EK, 'ayırma')
        if genis:
            for t in self.tablo.get(w, ()):
                # tablodaki karşılık da konuşma dili olabilir (cekiyodu -> çekiyodu): yazı diline çevir
                for c in ([t] if self.duzgun(t) else self.konusma_dili(t)[0]):
                    if self.tanidik(c):
                        ekle(c, C_TABLO, 'tablo')
        if gecerli:
            # Yalnızca özel isim/kısaltma olarak "geçerli" (boyle, hayatin, yaaa): Zemberek'in özel isim sözlüğü
            # çok geniş olduğu için küçük harfle yazılmış böyle bir kelime çoğu zaman Türkçe karakteri eksik ya da
            # uzatılmış sıradan bir kelimedir. Yukarıdaki güvenli adaylar denenir; harf tahminine girilmez.
            return C
        # Düzenleme mesafesi adayları. Kısa ya da yabancı görünen kelimelerde komşu kelime sayısı çok
        # fazla ve hepsi "olası" görünür; oralarda tahmin yürütmek yerine kelimeyi koruyoruz.
        if not kisa:
            ham = [c for c in self.oneriler(w)
                   if not (len(c) == len(w) and sum(x != y for x, y in zip(w, c)) == 1
                           and any(x == 'ğ' and y == 'k' for x, y in zip(w, c)))]   # buçuğu -> "buçuku" değil
            if YABANCI_HARF & set(w):
                # q/w/x Türkçe bir kelimede ancak komşu tuşa kayma olabilir (yqrdım -> yardım, otellwri -> otelleri).
                # Bunun dışındaki her durumda kelime yabancıdır (what, show, follow) ve korunur.
                ham = [c for c in ham if len(c) == len(w) and all(
                    x == y or (x in YABANCI_HARF and frozenset((x, y)) in _KLAVYE) for x, y in zip(w, c))]
            secilen = [c for c in ham if self.tanidik(c)]
            if not secilen and len(w) >= TEK_ADAY_UZUNLUK:
                duzgunler = [c for c in ham if "'" not in c and self.duzgun(c)]
                if len(duzgunler) == 1:
                    secilen = duzgunler            # uzun kelime + tek aday: nadir çekimli biçim
            for c in secilen:
                ekle(c, duzenleme_maliyeti(w, c) + (C_OZEL if "'" in c else 0.0)
                     + (C_ILK_HARF if c[0] != w[0] else 0.0), 'yazım')
        m = _SORU_EKI.match(w)
        if m and len(m.group(1)) >= 3:
            if self.duzgun(m.group(1)):                      # atarmısınız -> atar mısınız
                ekle((m.group(1), _uyumlu_soru_eki(m.group(1), m.group(2))), C_AYIR, 'ayırma')
            else:                                            # gelcekmisin -> gelecek misin
                for c in self.konusma_dili(m.group(1))[0]:
                    ekle((c, _uyumlu_soru_eki(c, m.group(2))), C_AYIR + C_KONUSMA, 'ayırma')
        n = len(w)
        for k in range(1, n):                                # eksik boşluk: vebeni -> ve beni
            a, b = w[:k], w[k:]
            if (len(a) < 3 and a not in KISA_KELIMELER) or (len(b) < 3 and b not in KISA_KELIMELER):
                continue
            if self.duzgun(a) and self.duzgun(b) and self.lm.var(a, b):   # ikili derlemde görülmüş olmalı
                ekle((a, b), C_AYIR, 'ayırma')
        return C

    def birlesim_adaylari(self, a, b, on='<s>', art='</s>'):
        """Yanlışlıkla bölünmüş kelime: 'orma na' -> ormana, 'bogul du' -> boğuldu, 'herkez ler' -> herkesler.
        on/art: çiftin iki yanındaki kelimeler (bağlam kanıtı için). Döndürür: {(birleşik biçim,): (maliyet, kaynak)}

        Ne kadar kanıt istendiği, parçaların ne olduğuna bağlı (ölçüm: 15.734 kelimelik temiz metin + 1.128 yapay bölünme):
          - iki parça da kelime değil (orma na)        -> birleşik biçimin geçerli bir kelime olması yeter
          - biri kelime (son ra)                       -> birleşik biçim ayrıca derlemde görülmüş olmalı
          - parçalardan biri ve/de/mi/bu... (bro ve)   -> birleşik biçim iki komşusuyla da derlemde görülmüş olmalı
          - ikisi de kelime (kaybet tik)               -> yalnızca dil modeli çok isterse (C_BIRLESTIR_GECERLI)
          - tek harfli parça (i de)                    -> hiç birleştirme"""
        if not (a.isalpha() and b.isalpha()) or len(a) < 2 or len(b) < 2:
            return {}
        j = a + b
        da, db = self.duzgun(a), self.duzgun(b)
        ikisi_de_kelime = da and db
        taban = C_BIRLESTIR_GECERLI if ikisi_de_kelime else C_BIRLESTIR
        islev = a in KISA_KELIMELER or b in KISA_KELIMELER

        def kabul(c):
            if (da or db) and self.lm.idx(c) == 0:
                return False
            if islev and not ikisi_de_kelime and not (self.lm.var(on, c) and self.lm.var(c, art)):
                return False
            return True

        C = {}
        if self.duzgun(j):
            if kabul(j):
                C[(j,)] = (taban, 'birleştirme')
            return C
        for yanlis, dogru in GOVDE_DUZELT.items():           # herkezler -> herkesler
            if j.startswith(yanlis):
                aday = (dogru + j[len(yanlis):]).split()
                if len(aday) == 1 and self.duzgun(aday[0]) and kabul(aday[0]):
                    C[(aday[0],)] = (taban + C_SOZLUK, 'birleştirme')
        # Birleştirip üstüne bir de düzeltme (bogul du -> boğuldu) iki ayrı tahmin demek: yalnızca parçaların
        # ikisi de hiçbir biçimde kelime değilken ya da ikisi de kelimeyken (o zaman dil modeli eşiği korur) denenir.
        if not (ikisi_de_kelime or not (self.gecerli(a) or self.gecerli(b))):
            return C
        konusma, aksan = self.konusma_dili(j)
        for c in konusma:
            if self.tanidik(c) and kabul(c):
                C.setdefault((c,), (taban + C_KONUSMA, 'birleştirme'))
        for c in aksan:                                      # boguldu -> boğuldu
            if self.tanidik(c) and kabul(c):
                C.setdefault((c,), (taban + 0.8, 'birleştirme'))
        return C

    # ---- çözümleme: Viterbi (tek kelimelik ve iki kelimeyi birleştiren kenarlarla) ----
    def duzelt(self, kelimeler, birlesebilir=None):
        """kelimeler: küçük harfli kelime listesi (bir cümle/parça).
        birlesebilir[i]: i. ve (i+1). kelime arasında yalnızca boşluk var mı (birleştirme adayı olabilir mi).
        Döndürür: her kelime için (yeni_kelimeler_demeti, kaynak). İki kelime birleştirildiyse ilki için
        boş demet, ikincisi için birleşik biçim döner."""
        n = len(kelimeler)
        if not n:
            return []
        lm = self.lm
        tek = [self.adaylar(w) for w in kelimeler]
        cift = [{} for _ in range(n)]
        for i in range(n - 1):
            if birlesebilir is None or birlesebilir[i]:
                cift[i] = self.birlesim_adaylari(kelimeler[i], kelimeler[i + 1],
                                                 kelimeler[i - 1] if i else '<s>',
                                                 kelimeler[i + 2] if i + 2 < n else '</s>')
        # en_iyi[i]: ilk i kelime tüketildikten sonra {son belirteç: (skor, geri işaret)}
        en_iyi = [dict() for _ in range(n + 1)]
        en_iyi[0]['<s>'] = (0.0, None)
        for i in range(n):
            for adaylar_, adim in ((tek[i], 1), (cift[i], 2)):
                for c, (maliyet, kaynak) in adaylar_.items():
                    ic = sum(lm.lp(c[k - 1], c[k]) for k in range(1, len(c)))
                    skor, onceki = -1e18, None
                    for sonb, (ps, _) in en_iyi[i].items():
                        v = ps + lm.lp(sonb, c[0])
                        if v > skor:
                            skor, onceki = v, sonb
                    skor += ic - maliyet
                    hedef = en_iyi[i + adim]
                    if c[-1] not in hedef or skor > hedef[c[-1]][0]:
                        hedef[c[-1]] = (skor, (i, onceki, c, kaynak, adim))
        durum = max(en_iyi[n].items(), key=lambda kv: kv[1][0] + lm.lp(kv[0], '</s>'))[0]
        sonuc = [None] * n
        i = n
        while i > 0:
            pi, onceki, c, kaynak, adim = en_iyi[i][durum][1]
            if adim == 1:
                sonuc[pi] = (c, kaynak)
            else:
                sonuc[pi] = ((), kaynak)
                sonuc[pi + 1] = (c, kaynak)
            i, durum = pi, onceki
        self._guven_denetimi(kelimeler, sonuc, tek)
        if self.siralayici is not None:
            self._siralayici_karari(kelimeler, sonuc)
        self._kural_son_gecis(kelimeler, sonuc, birlesebilir)
        return self._gercek_kelime(sonuc)

    def _kural_son_gecis(self, kelimeler, sonuc, birlesebilir=None):
        """Seçilen düzeltmeyi bir kez de kurallardan geçir. Aday üretimi girdiye bakar; seçilen adayın kendisi kural
        tablolarından geçmemiş olabilir. Yalnızca zaten değiştirilmiş kelimelere dokunur (korunan kelime korunur):
          - tablonun "yanlış" dediği biçim: hersey -> (aksan) herşey -> her şey; bıde -> bide -> bir de
          - konuşma dili gövdesi: ordayim -> ordayım -> oradayım; burdakiler -> buradakiler
          - bitişik soru eki, ünlü uyumuyla: varmi -> varmı -> var mı
        Ayrıca TDK'ye göre her zaman bitişik yazılan hiçbir/birkaç/herhangi: hic bi -> hiç bir -> hiçbir."""
        for i, (c, kaynak) in enumerate(sonuc):
            if not c or c == (kelimeler[i],) or kaynak == 'sözlük':
                continue
            yeni = []
            for t in c:
                d = TAM_DUZELT.get(t)
                if d is None:
                    d = next((dogru + t[len(yanlis):] for yanlis, dogru in GOVDE_DUZELT.items()
                              if t.startswith(yanlis)), None)
                if d is None:
                    d = next((dogru + t[len(yanlis):] for yanlis, dogru in KONUSMA_GECERLI.items()
                              if yanlis[-2:] in ('da', 'de') and t.startswith(yanlis)), None)
                if d is None and not self.duzgun(t):
                    m = _SORU_EKI.match(t)
                    if m and len(m.group(1)) >= 2 and self.duzgun(m.group(1)):
                        d = m.group(1) + ' ' + _uyumlu_soru_eki(m.group(1), m.group(2))
                parca = d.split() if d else [t]
                yeni += parca if all(self.gecerli(x) for x in parca) else [t]
            if tuple(yeni) != c:
                sonuc[i] = (tuple(yeni), 'sözlük')
        for i in range(len(sonuc) - 1):
            a, b = sonuc[i][0], sonuc[i + 1][0]
            if len(a) != 1 or len(b) != 1 or (birlesebilir is not None and not birlesebilir[i]):
                continue
            if ((a[0] == 'hiç' and b[0].startswith('bir')) or (a[0] == 'her' and b[0].startswith('hangi'))
                    or (a[0], b[0]) == ('bir', 'kaç')):
                j = a[0] + b[0]
                # yalnızca biri zaten düzeltildiyse (hic bi -> hiçbir); yazarın kendi yazdığı "hiç bir" korunur
                if self.duzgun(j) and (a, b) != ((kelimeler[i],), (kelimeler[i + 1],)):   # hiç birlikte -> birleşmez
                    sonuc[i] = ((), 'birleştirme')
                    sonuc[i + 1] = ((j,), 'birleştirme')

    # ---- öğrenilmiş sıralayıcı: her aday için "bu aday doğru mu?" olasılığı ----
    def _deasciifier(self, kelimeler):
        """Zemberek'teki örüntü tabanlı Deasciifier'ın (Yüret) her kelime için önerisi. Tek başına uygulanınca doğru
        kelimelerin ~%2'sini bozuyor (go -> "gö"); burada yalnızca sıralayıcıya "ikinci görüş" özniteliği olarak girer."""
        try:
            from zemberek.normalization.deasciifier.deasciifier import Deasciifier
            out = [tr_lower(t) for t in Deasciifier(' '.join(kelimeler)).convert_to_turkish().split()]
        except Exception:
            return None
        return out if len(out) == len(kelimeler) else None

    def aday_satirlari(self, kelimeler, sonuc):
        """Sıralayıcının göreceği tablo: [(konum, [(aday, öznitelik vektörü)])]. Eğitim (egit_siralayici.py) ve
        çalışma anı AYNI fonksiyonu kullanır. sonuc: elle ayarlı ilk geçişin çıktısı (komşu kelimeler oradan alınır)."""
        lm = self.lm
        n = len(kelimeler)
        tek = [self.adaylar(w, genis=True) for w in kelimeler]
        gecersiz = [w.isalpha() and not self.gecerli(w) for w in kelimeler]
        ozet = [float(n), sum(gecersiz) / n,
                sum(1 for w in kelimeler if _TR_HARF.search(w)) / n,
                sum(1 for w, C in zip(kelimeler, tek)
                    if not (self.gecerli(w) and self.duzgun(w)) and any(k == 'aksan' for _, k in C.values())) / n]
        da = self._deasciifier(kelimeler)
        out = []
        for i, w in enumerate(kelimeler):
            if not sonuc[i][0] or sonuc[i][1] == 'birleştirme' or len(tek[i]) < 2:
                continue                     # birleştirmeler ilk geçişte kararlaştırılır; tek adaylıda karar yok
            on, art = '<s>', '</s>'
            for k in range(i - 1, -1, -1):
                if sonuc[k][0]:
                    on = sonuc[k][0][-1]
                    break
            for k in range(i + 1, n):
                if sonuc[k][0]:
                    art = sonuc[k][0][0]
                    break
            skor = {}
            for c, (maliyet, _) in tek[i].items():
                skor[c] = (lm.lp(on, c[0]) + sum(lm.lp(c[k - 1], c[k]) for k in range(1, len(c)))
                           + lm.lp(c[-1], art) - maliyet)
            tepe = max(skor.values())
            toplam = sum(math.exp(v - tepe) for v in skor.values())
            sira = {c: r for r, c in enumerate(sorted(skor, key=skor.get, reverse=True))}
            wt = lm.tekli(w)
            wt = wt if wt > -1e9 else -25.0
            sol_w, sag_w = lm.lp(on, w), lm.lp(w, art)
            w_oz = [float(self.gecerli(w)), float(self.duzgun(w)), float(lm.idx(w) > 0), wt,
                    float(w == w.translate(_ASCII)), float(bool(_TR_HARF.search(w))), float(bool(YABANCI_HARF & set(w))),
                    float(max((m.end() - m.start() for m in _TEKRAR.finditer(w)), default=1)),
                    sol_w, sag_w, float(lm.var(on, w)), float(lm.var(w, art))]
            komsu = [float(i > 0 and gecersiz[i - 1] and sonuc[i - 1][0] == (kelimeler[i - 1],)),
                     float(i + 1 < n and gecersiz[i + 1] and sonuc[i + 1][0] == (kelimeler[i + 1],))]
            satirlar = []
            for c, (maliyet, kaynak) in tek[i].items():
                cl = ''.join(c).replace("'", '')
                ct = min(lm.tekli(t) for t in c)
                ct = ct if ct > -1e9 else -25.0
                sol_c, sag_c = lm.lp(on, c[0]), lm.lp(c[-1], art)
                ic = sum(lm.lp(c[k - 1], c[k]) for k in range(1, len(c)))
                v = [float(kaynak == k) for k in _KAYNAKLAR]
                v += [maliyet, float(len(w)), float(len(cl)), float(len(c))]
                v += w_oz
                v += [float(all(self.duzgun(t) for t in c)), float(all(lm.idx(t) > 0 for t in c)), ct,
                      sol_c, sag_c, float(lm.var(on, c[0])), float(lm.var(c[-1], art)),
                      (sol_c + ic + sag_c) - (sol_w + sag_w), ct - wt]
                v += [math.exp(skor[c] - tepe) / toplam, float(sira[c]), float(len(tek[i])), float(sonuc[i][0] == c)]
                v += [float(_lev(w, cl)), float(w[0] != cl[0]), float(w[-1] != cl[-1]),
                      float(w.translate(_ASCII) == cl.translate(_ASCII)),
                      float(_TEKRAR.sub(r'\1', w) == _TEKRAR.sub(r'\1', cl)), float(sorted(w) == sorted(cl))]
                v += ozet + komsu
                v += [float(da is not None and len(c) == 1 and c[0] == da[i]), float(da is not None and da[i] == w)]
                satirlar.append((c, v))
            out.append((i, satirlar))
        return out

    def _siralayici_karari(self, kelimeler, sonuc):
        """Elle ayarlı ilk geçişin kararlarını öğrenilmiş sıralayıcının kararlarıyla değiştir. Her aday için
        "doğru kelime bu mu?" olasılığı hesaplanır; kelimenin kendisinden farklı en iyi aday, olasılığı hem eşiği
        hem de "olduğu gibi bırak" seçeneğini geçiyorsa uygulanır. Eşik yükseldikçe daha az ama daha emin düzeltir."""
        esik = self.siralayici.esik
        for i, satirlar in self.aday_satirlari(kelimeler, sonuc):
            if sonuc[i][1] == 'sözlük':
                continue        # elle yazılmış kural tablosundan gelen düzeltme kesindir: eklenen kural her zaman işler
            p = self.siralayici.olasilik([v for _, v in satirlar])
            ayni = (kelimeler[i],)
            p_ayni = max((pi for (c, _), pi in zip(satirlar, p) if c == ayni), default=0.0)
            en_iyi, p_en = max(((c, pi) for (c, _), pi in zip(satirlar, p) if c != ayni), key=lambda x: x[1])
            if p_en >= esik and p_en > p_ayni:
                sonuc[i] = (en_iyi, self.adaylar(kelimeler[i], genis=True)[en_iyi][1])
            else:
                sonuc[i] = (ayni, 'aynı')

    def _guven_denetimi(self, kelimeler, sonuc, tek):
        """Düzenleme mesafesiyle bulunan ('yazım') düzeltmeleri süz. Kelime şu durumlarda tahmine
        kurban edilmez, yerine en iyi GÜVENİLİR seçenek (tablo/konuşma dili/aksan ya da kelimenin kendisi) gelir:
          - seçilen aday, o konumdaki diğer seçeneklere göre açık ara önde değilse (belirsizlik),
          - değişiklik yazım hatasına benzemiyorsa (komşu tuş / ses benzerliği / yer değiştirme değil,
            keyfi bir harf) ve aday bu bağlamda derlemde hiç görülmemişse,
          - hemen yanında yabancı bir kelime varsa (ow şitmen: büyük olasılıkla o da yabancı)."""
        lm = self.lm
        n = len(sonuc)
        for i, (c, kaynak) in enumerate(sonuc):
            if kaynak != 'yazım':
                continue
            on, art = '<s>', '</s>'
            for k in range(i - 1, -1, -1):
                if sonuc[k][0]:
                    on = sonuc[k][0][-1]
                    break
            for k in range(i + 1, n):
                if sonuc[k][0]:
                    art = sonuc[k][0][0]
                    break
            skor = {}
            for a, (maliyet, _) in tek[i].items():
                skor[a] = (lm.lp(on, a[0]) + sum(lm.lp(a[k - 1], a[k]) for k in range(1, len(a)))
                           + lm.lp(a[-1], art) - maliyet)
            tepe = max(skor.values())
            p = math.exp(skor[c] - tepe) / sum(math.exp(v - tepe) for v in skor.values())
            w = kelimeler[i]
            guvenli = p >= (GUVEN_KISA if len(w) <= 5 else GUVEN_UZUN)
            if guvenli and tek[i][c][0] >= KEYFI_DEGISIKLIK and not (lm.var(on, c[0]) or lm.var(c[-1], art)):
                guvenli = False
            if guvenli and ((i > 0 and YABANCI_HARF & set(kelimeler[i - 1]))
                            or (i + 1 < n and YABANCI_HARF & set(kelimeler[i + 1]))):
                guvenli = False
            if not guvenli:
                yedek = max((a for a in tek[i] if tek[i][a][1] != 'yazım'), key=lambda a: skor[a])
                sonuc[i] = (yedek, tek[i][yedek][1])

    def gercek_kelime_adayi(self, on, w, art):
        """Geçerli w kelimesi için bağlama daha iyi oturan komşu biçim: (kazanç, aday, kanıt) ya da None.
        kanıt = (özgün kelime bu bağlamda derlemde görülmüş mü, aday iki yanıyla da görülmüş mü)"""
        lm = self.lm
        taban = lm.lp(on, w) + lm.lp(w, art)
        ozgun_gorulmus = lm.var(on, w) or lm.var(w, art)
        en_iyi = None
        for a in self.oneriler(w):
            if "'" in a or lm.idx(a) == 0 or not self.duzgun(a):
                continue
            sol, sag = lm.var(on, a), lm.var(a, art)
            if not (sol or sag):
                continue                                  # kanıt gözlenmiş bir ikiliden gelmeli
            kazanc = lm.lp(on, a) + lm.lp(a, art) - taban
            if en_iyi is None or kazanc > en_iyi[0]:
                en_iyi = (kazanc, a, (ozgun_gorulmus, sol and sag))
        return en_iyi

    def _gercek_kelime(self, sonuc):
        """2. geçiş: yazımı DOĞRU ama bağlama uymayan kelimeler (yaşam küçük -> yaşım küçük).
        Yanlış alarm pahalı olduğu için yalnızca güçlü kanıtla değiştir."""
        lm = self.lm
        duz = [t for c, _ in sonuc for t in c]
        yer = 0
        out = []
        for c, kaynak in sonuc:
            if kaynak == 'aynı' and len(c) == 1 and len(c[0]) >= 3 and c[0].isalpha() and lm.idx(c[0]) > 0:
                on = duz[yer - 1] if yer > 0 else '<s>'
                art = duz[yer + 1] if yer + 1 < len(duz) else '</s>'
                en_iyi = self.gercek_kelime_adayi(on, c[0], art)
                if en_iyi and gercek_kelime_kabul(*en_iyi):
                    c, kaynak = (en_iyi[1],), 'bağlam (gerçek kelime)'
                    duz[yer] = en_iyi[1]
            out.append((c, kaynak))
            yer += len(c)
        return out


def gercek_kelime_kabul(kazanc, aday, kanit):
    ozgun_gorulmus, aday_iki_yanla_gorulmus = kanit
    return kazanc >= GERCEK_KELIME_ESIK and aday_iki_yanla_gorulmus


# ==========================================================================
# 3-4) NOKTALAMA + BÜYÜK HARF + ÖZEL İSİM
# ==========================================================================

# Ölçüm: nokta F1 ~0,64-0,78 (türe göre); virgülde kesinlik 0,5 eşiğinde yalnızca %28-42.
# Yanlış virgül, eksik virgülden daha rahatsız edici olduğu için virgül eşiği yüksek tutuluyor.
NOKTA_ESIK = 0.5
VIRGUL_ESIK = 0.8
BUYUK_ESIK = 0.5                # kelimenin özel isim dışında anlamı da varsa (deniz, gül) CRF'in vermesi gereken olasılık
BUYUK_ESIK_YALNIZ_OZEL = 0.5    # yalnızca özel isimse (trabzon) gereken olasılık

AYLAR = ('ocak', 'şubat', 'mart', 'nisan', 'mayıs', 'haziran', 'temmuz', 'ağustos', 'eylül',
         'ekim', 'kasım', 'aralık')
_AY_RE = re.compile(r"^(%s)(?:'?([a-zçğıöşü]+))?$" % '|'.join(AYLAR))


def _windows_dll_dizinleri():
    """Windows'ta python-crfsuite, Visual C++ çalışma zamanına (MSVCP140.dll) ihtiyaç duyar. Sistemde
    kurulu değilse `pip install msvc-runtime` onu ortamın köküne ve Scripts klasörüne koyar; o
    klasörleri DLL arama yoluna ekliyoruz ki yönetici hakkı isteyen kurulum şart olmasın."""
    if os.name != 'nt' or not hasattr(os, 'add_dll_directory'):
        return
    for d in (sys.prefix, os.path.join(sys.prefix, 'Scripts')):
        try:
            if os.path.isfile(os.path.join(d, 'msvcp140.dll')):
                os.add_dll_directory(d)
        except OSError:
            pass


class Noktalayici:
    def __init__(self, model_yolu, lm, morph):
        _windows_dll_dizinleri()
        import pycrfsuite
        from zemberek.core.turkish import SecondaryPos, RootAttribute
        self.tagger = pycrfsuite.Tagger()
        # Modeli Python okuyup bellekten veriyoruz: CRFsuite dosya yolunu C'nin fopen'ıyla açtığı için
        # Windows'ta Türkçe karakterli klasörlerde (C:\Users\Oğuzhan\...) Tagger.open(yol) başarısız
        # olur. CRFsuite bu belleği kopyalamaz; nesne yaşadıkça baytları da elde tutmak gerekir.
        with open(model_yolu, 'rb') as f:
            self._model_baytlari = f.read()
        self.tagger.open_inmemory(self._model_baytlari)
        self.lm = lm
        self.morph = morph
        self.SecondaryPos = SecondaryPos
        self.RootAttribute = RootAttribute

    def etiketle(self, kelimeler, parca=300):
        """Her kelime için 'lO', 'cP' ... (harf durumu + ardından gelen noktalama)."""
        temiz = [w.replace("'", '') or w for w in kelimeler]
        etiketler, buyuk = [], []
        for i in range(0, len(temiz), parca):
            e, b = crf_coz(self.tagger, crf_features(temiz[i:i + parca], self.lm), NOKTA_ESIK, VIRGUL_ESIK)
            etiketler += e
            buyuk += b
        return etiketler, buyuk

    def cozumle(self, cumle):
        """Bir cümlenin kelimeleri için Zemberek çözümlemesi + belirsizlik giderme.
        Döndürür: [(lemma, tür, özel_isim_mi, yalnız_özel_mi, kök_yazımı, ek_ve_kesme)]"""
        if not cumle:
            return []
        analizler = [self.morph.analyze(w) for w in cumle]
        try:
            en_iyiler = self.morph.disambiguate(' '.join(cumle), analizler).best_analysis()
        except Exception:
            en_iyiler = [a.analysis_results[0] if a.analysis_results else None for a in analizler]
        ozel_turler = (self.SecondaryPos.ProperNoun, self.SecondaryPos.Abbreviation)
        out = []
        for w, wa, b in zip(cumle, analizler, en_iyiler):
            if b is None or b.is_unknown():
                out.append((w, 'Bilinmiyor', False, False, w, ''))
                continue
            bilinen = [a for a in wa.analysis_results if not a.is_unknown()]
            ozel = b.item.secondary_pos in ozel_turler and not b.is_runtime()
            yalniz_ozel = bool(bilinen) and all(a.item.secondary_pos in ozel_turler for a in bilinen)
            ek = b.get_ending()
            if ek and ozel:
                # TDK: özel ada gelen ÇEKİM eki kesmeyle ayrılır (Trabzon'a, Almanya'daki);
                # yapım eki ve çokluk eki ayrılmaz (Ankaralı, Ahmetler, Türkçenin)
                kesme = self.RootAttribute.NoQuote not in (b.item.attributes or ())
                lem = tr_lower(b.item.lemma)
                if lem[-2:] in ('ce', 'ca', 'çe', 'ça') and any(     # dil adları: Türkçenin, Almancada
                        not a.is_unknown() and a.item.secondary_pos == self.SecondaryPos.ProperNoun
                        for a in self.morph.analyze(lem[:-2])):
                    kesme = False
                for md in b.morpheme_data_list[1:]:
                    if md.surface:
                        if md.morpheme.derivational_ or md.morpheme.id_ == 'A3pl':
                            kesme = False
                        break
                ek = ("'" if kesme else '') + ek
            out.append((b.item.lemma, b.item.primary_pos.name, ozel, yalniz_ozel, b.item.lemma, ek))
        return out


# ==========================================================================
# 5) PARAGRAF — TextTiling (Hearst 1997) mantığı, cümle düzeyinde   [DENEYSEL, varsayılan KAPALI]
#
# Ölçüm (paragrafları belli 120 öğrenci yazısı — Bilkent Turkish Writings Dataset, Yilmaz 2025,
# yalnızca ölçüm için; Pk: düşük = iyi):
#   bu yöntem, en uygun hedef uzunlukla 0,46-0,50 | rastgele 0,49 | hiç bölmemek 0,36-0,39
# Yani rastgele bölmekten iyi değil; kısa metinde lemma örtüşmesi ayırt edici olmuyor.
# 7.344 yazıyla eğitilen denetimli bir CRF (yüzey öznitelikleri) bile ancak 0,356'ya indi
# (hiç bölmemek: 0,371). Paragraf, bu betikteki klasik yöntemlerin çözemediği kısım.
# ==========================================================================

ICERIK_TURLERI = {'Noun', 'Verb', 'Adjective', 'Adverb'}
DURAK_LEMMA = {'olmak', 'etmek', 'yapmak', 'demek', 'var', 'yok', 'bir', 'çok', 'daha', 'gibi', 'kadar',
               'şey', 'zaman', 'almak', 'gelmek', 'gitmek', 'vermek', 'değil', 'en', 'pek', 'ne', 'her',
               'tüm', 'bütün', 'hiç', 'zaten', 'gerçekten', 'tabi', 'tabii', 'şekil', 'doğru', 'için'}


def _cos(a, b):
    ort = set(a) & set(b)
    if not ort:
        return 0.0
    pay = sum(a[k] * b[k] for k in ort)
    return pay / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def paragraf_sinirlari(cumle_lemmalari, cumle_uzunluklari, pencere=2, hedef_kelime=60, min_kelime=12):
    """cumle_lemmalari: her cümle için içerik lemmalarının listesi.
    Döndürür: kendisinden SONRA paragraf açılacak cümle indeksleri."""
    n = len(cumle_lemmalari)
    if n < 3:
        return []
    from collections import Counter
    torba = [Counter(l) for l in cumle_lemmalari]
    benzerlik = []
    for g in range(n - 1):                      # g: cümle g ile g+1 arasındaki boşluk
        sol, sag = Counter(), Counter()
        for k in range(max(0, g - pencere + 1), g + 1):
            sol.update(torba[k])
        for k in range(g + 1, min(n, g + 1 + pencere)):
            sag.update(torba[k])
        benzerlik.append(_cos(sol, sag))
    derinlik = []
    for g, s in enumerate(benzerlik):            # vadinin iki yanındaki tepelere göre derinlik
        sol_tepe = s
        for k in range(g - 1, -1, -1):
            if benzerlik[k] >= sol_tepe:
                sol_tepe = benzerlik[k]
            else:
                break
        sag_tepe = s
        for k in range(g + 1, len(benzerlik)):
            if benzerlik[k] >= sag_tepe:
                sag_tepe = benzerlik[k]
            else:
                break
        derinlik.append((sol_tepe - s) + (sag_tepe - s))
    toplam = sum(cumle_uzunluklari)
    kac = max(0, round(toplam / float(hedef_kelime)) - 1)      # uzunluğa göre beklenen sınır sayısı
    adaylar = sorted(range(n - 1), key=lambda g: (-derinlik[g], benzerlik[g]))
    secilen = []
    for g in adaylar:
        if len(secilen) >= kac:
            break
        if derinlik[g] <= 0 and benzerlik[g] > 0:
            continue
        sinirlar = sorted(secilen + [g])
        bas, uygun = 0, True
        for s in sinirlar + [n - 1]:
            if sum(cumle_uzunluklari[bas:s + 1]) < min_kelime:
                uygun = False
            bas = s + 1
        if uygun:
            secilen.append(g)
    return sorted(secilen)


# ==========================================================================
# Hepsi bir arada
# ==========================================================================

_YALNIZ_BOSLUK = re.compile(r'[ \t]+')
_NOKTA_ONCESI_BOSLUK = re.compile(r'[ \t]+(?=[.,!?;:…])')
_COK_BOSLUK = re.compile(r'[ \t]{2,}')


class Duzeltici:
    def __init__(self, model_yolu):
        self.lm = DilModeli()
        self.yazim = YazimDuzeltici(self.lm, os.path.join(os.path.dirname(os.path.abspath(model_yolu)), 'siralayici.json'))
        self.nokta = Noktalayici(model_yolu, self.lm, self.yazim.morph)

    def duzelt(self, metin, paragraf=False, ayrinti=False, izle=None):
        """izle: liste verilirse her aşamanın ara sonucu ona eklenir (--yolculuk bunu yazdırır)."""
        metin = metin.replace('’', "'")
        degisiklikler = []
        # --- 0a) hiç Türkçe karakter yoksa metin ASCII klavyeyle yazılmıştır: önce hepsini geri koy
        #         (Yüret'in örüntü tabanlı deasciifier'ı; Zemberek içinde geliyor).
        #         Sıralayıcı varken bu ön adım atlanır: o, Türkçe karakterli eşleri kelime kelime aday olarak tartıyor.
        #         Ölçüm (tümüyle ASCII yazılmış 364 gerçek tweet): yalnız sıralayıcı %79,7 düzeltiyor, doğru kelimelerin
        #         %0,40'ını bozuyor; ön adımla birlikte %85,6 düzeltiyor ama %2,13'ünü bozuyor (go -> "gö" gibi).
        if (self.yazim.siralayici is None and sum(ch.isalpha() for ch in metin) >= 20
                and not re.search('[çğışöüÇĞİŞÖÜ]', metin)):
            from zemberek.normalization.deasciifier.deasciifier import Deasciifier
            metin = Deasciifier(metin).convert_to_turkish()
            degisiklikler.append(('(tüm metin)', 'Türkçe karakterler geri kondu', 'deasciifier'))
            if izle is not None:
                izle.append(('deasciifier', metin))
        # --- 0b) metni (ayraç, kelime) parçalarına böl; ayraçlar (boşluk/noktalama) korunur
        kelimeler, ayraclar, son = [], [], 0
        bitisler = []
        for m in WORD_RE.finditer(metin):
            ayraclar.append(metin[son:m.start()])
            kelimeler.append(m.group())
            son = m.end()
            bitisler.append(son)
        kuyruk = metin[son:]
        if not kelimeler:
            return (metin, []) if ayrinti else metin
        if izle is not None:
            izle.append(('kelimeler', list(kelimeler)))
        n = len(kelimeler)
        # kullanıcının satır sonu = kesin cümle sonu (uzun satırlar hariç: otomatik kaydırılmış olabilir)
        sert = [False] * n
        for i in range(n - 1):
            ayrac = ayraclar[i + 1]
            if '\n' in ayrac:
                satir_basi = metin.rfind('\n', 0, bitisler[i]) + 1
                sert[i] = ayrac.count('\n') >= 2 or bitisler[i] - satir_basi < 60
        # kullanıcının cümle ortasında büyük yazdığı kelimeler (özel isim ipucu)
        kullanici_buyuk = []
        for i, w in enumerate(kelimeler):
            cumle_basi = i == 0 or bool(CUMLE_SONU.search(ayraclar[i]))
            kullanici_buyuk.append(w[0].isupper() and not cumle_basi)
        if sum(kullanici_buyuk) > 0.5 * n:          # TAMAMI BÜYÜK HARF yazılmış metin: ipucu değil
            kullanici_buyuk = [False] * n

        # --- 1-2) yazım + sayı eki kuralı (kullanıcının koyduğu cümle sonlarına göre parça parça)
        kucuk = [tr_lower(w) for w in kelimeler]
        yeni = [None] * n
        parca = []

        def parcayi_isle():
            if not parca:
                return
            girdi = []
            for i in parca:
                s = sayi_eki_duzelt(kucuk[i]) if kucuk[i][0].isdigit() else None
                girdi.append(s or kucuk[i])
            # iki kelime ancak aralarında yalnızca boşluk varsa ve kullanıcı ikisini de küçük yazdıysa birleşebilir
            birlesebilir = [bool(_YALNIZ_BOSLUK.fullmatch(ayraclar[i + 1]))
                            and not kullanici_buyuk[i] and not kullanici_buyuk[i + 1]
                            for i in parca[:-1]]
            cikti = self.yazim.duzelt(girdi, birlesebilir)
            if izle is not None:
                genis = self.yazim.siralayici is not None
                izle.append(('yazım', [(g, self.yazim.adaylar(g, genis), c, kaynak)
                                       for g, (c, kaynak) in zip(girdi, cikti)]))
            for k, (i, g, (c, kaynak)) in enumerate(zip(parca, girdi, cikti)):
                # kullanıcının büyük yazdığı ve sıradan kelime olarak çözümlenemeyen kelimelere (ad, yabancı ad) dokunma
                if kullanici_buyuk[i] and not self.yazim.duzgun(kucuk[i]) and kaynak != 'aynı':
                    c, kaynak = (g,), 'aynı'
                if g != kucuk[i] and kaynak == 'aynı':
                    kaynak = 'sayı eki kuralı'
                yeni[i] = c
                if not c:
                    continue                    # sonraki kelimeyle birleştirildi; kaydı o kelime düşer
                if kaynak == 'birleştirme':
                    degisiklikler.append((kelimeler[parca[k - 1]] + ' ' + kelimeler[i], ' '.join(c), kaynak))
                elif ' '.join(c) != kucuk[i]:
                    degisiklikler.append((kelimeler[i], ' '.join(c), kaynak))
            del parca[:]

        for i in range(n):
            if i > 0 and (CUMLE_SONU.search(ayraclar[i]) or sert[i - 1]):
                parcayi_isle()
            parca.append(i)
        parcayi_isle()

        # --- düz belirteç listesi (ayrılan kelimeler birden çok belirteç)
        duz, sahip = [], []
        for i in range(n):
            for t in yeni[i]:
                duz.append(t)
                sahip.append(i)
        N = len(duz)

        # --- 3) CRF: harf durumu + noktalama (kullanıcının satır sonları ayrı diziler olarak)
        etiket, p_buyuk, bas = [], [], 0
        for j in range(N):
            son_belirtec = j + 1 == N or sahip[j + 1] != sahip[j]
            if j + 1 == N or (son_belirtec and sert[sahip[j]]):
                e, b = self.nokta.etiketle(duz[bas:j + 1])
                if e[-1][1] in 'OC':
                    e[-1] = e[-1][0] + 'P'                   # dizinin sonu = cümle sonu
                etiket += e
                p_buyuk += b
                bas = j + 1
        nokta = [e[1] for e in etiket]
        # kullanıcının zaten koyduğu noktalama korunur
        kullanici_nokta = [None] * N
        for j in range(N):
            i = sahip[j]
            if j + 1 < N and sahip[j + 1] == i:
                continue                                    # ayrılmış kelimenin iç belirteci
            ayrac = ayraclar[i + 1] if i + 1 < n else kuyruk
            m = re.match(r"\s*([.,!?…;:])", ayrac)
            if m:
                kullanici_nokta[j] = m.group(1)

        def cumle_sonu_mu(j):
            if kullanici_nokta[j] is not None:
                return kullanici_nokta[j] in '.!?…'
            return nokta[j] in 'PQ'

        # --- 4) cümle cümle Zemberek çözümlemesi: özel isim + lemma
        cumleler, bas = [], 0
        for j in range(N):
            if cumle_sonu_mu(j) or j == N - 1:
                cumleler.append((bas, j + 1))
                bas = j + 1
        bilgi = [None] * N
        for a, b in cumleler:
            for j, x in zip(range(a, b), self.nokta.cozumle(duz[a:b])):
                bilgi[j] = x

        # Büyük harf kararı üç kanıtın kesişimi: (1) Zemberek sözlüğünde özel isim, (2) büyük derlemde
        # gerçekten özel isim gibi kullanılmış (DilModeli.ozel_isimler), (3) kelimenin başka anlamı da
        # varsa CRF bağlamdan özel isim diyor. CRF tek başına (tanımadığı kelimeye) büyük harf koymaz;
        # kullanıcının kendi yazdığı büyük harf ise her zaman korunur.
        taninan_ozel = self.lm.ozel_isimler
        bicim = []
        for j, t in enumerate(duz):
            lemma, tur, ozel, yalniz_ozel, kok, ek = bilgi[j]
            kb = kullanici_buyuk[sahip[j]]
            if t[0].isdigit():
                s = t
            elif "'" in t:
                s = tr_capitalize(t) if (kb or t.split("'")[0] in taninan_ozel) else t
            elif ozel and (kb or (tr_lower(kok) in taninan_ozel and
                                  p_buyuk[j] >= (BUYUK_ESIK_YALNIZ_OZEL if yalniz_ozel else BUYUK_ESIK))):
                s = kok + ek                      # ek, gerekiyorsa kesme işaretiyle birlikte gelir
            elif kb:
                s = tr_capitalize(t)
            else:
                s = t
            bicim.append(s)
        # tarih kuralı: "15 temmuz" -> "15 Temmuz", "15 temmuzda" -> "15 Temmuz'da"
        for j, t in enumerate(duz):
            m = _AY_RE.match(t)
            if not m:
                continue
            onceki_gun = j > 0 and duz[j - 1].isdigit() and 1 <= int(duz[j - 1]) <= 31
            sonraki_yil = j + 1 < N and re.fullmatch(r'(1[5-9]|20)\d\d', duz[j + 1]) is not None
            if onceki_gun or sonraki_yil:
                bicim[j] = tr_capitalize(m.group(1)) + ("'" + m.group(2) if m.group(2) else '')
        # cümle başı büyük harf
        for a, b in cumleler:
            bicim[a] = tr_capitalize(bicim[a])

        # --- 5) paragraf (metinde zaten satır başı varsa dokunma)
        if izle is not None:
            izle.append(('noktalama', [(duz[j], nokta[j], kullanici_nokta[j]) for j in range(N)]))
            izle.append(('büyük harf', [(duz[j], bicim[j]) for j in range(N) if duz[j] != bicim[j]]))
        yeni_paragraf = set()
        if paragraf and '\n' not in metin.strip() and len(cumleler) >= 3:
            lemmalar = []
            for a, b in cumleler:
                lemmalar.append([bilgi[j][0] for j in range(a, b)
                                 if bilgi[j][1] in ICERIK_TURLERI and tr_lower(bilgi[j][0]) not in DURAK_LEMMA])
            for g in paragraf_sinirlari(lemmalar, [b - a for a, b in cumleler]):
                yeni_paragraf.add(cumleler[g][1] - 1)       # bu belirteçten sonra paragraf

        # --- birleştir
        isaret = {'C': ',', 'P': '.', 'Q': '?'}
        out = [ayraclar[0]]
        for j in range(N):
            i = sahip[j]
            out.append(bicim[j])
            ic = j + 1 < N and sahip[j + 1] == i
            if kullanici_nokta[j] is None and nokta[j] != 'O':
                out.append(isaret[nokta[j]])
            if ic:
                out.append(' ')
            else:
                ayrac = ayraclar[i + 1] if i + 1 < n else kuyruk
                ayrac = _COK_BOSLUK.sub(' ', _NOKTA_ONCESI_BOSLUK.sub('', ayrac))   # "bilmiyom . bunu  da"
                if j in yeni_paragraf and j != N - 1:
                    ayrac = re.sub(r'\s+$', '', ayrac) + '\n\n'
                out.append(ayrac)
        sonuc = ''.join(out)
        if izle is not None:
            izle.append(('çıktı', sonuc))
        if ayrinti:
            return sonuc, degisiklikler
        return sonuc


def yolculuk_yaz(metin, izle, dosya=sys.stdout):
    """Bir metnin düzelticiden geçerken uğradığı durakları okunur biçimde yaz."""
    yaz = lambda *a: print(*a, file=dosya)
    yaz('GİRDİ\n  ' + metin.strip())
    for ad, veri in izle:
        if ad == 'deasciifier':
            yaz('\n0. DURAK: metinde hiç Türkçe karakter (ç ğ ı ö ş ü) yok; önce onlar geri konur\n  ' + veri.strip())
        elif ad == 'kelimeler':
            yaz('\n1. DURAK: kelimelere ayır (%d kelime)\n  %s' % (len(veri), ' | '.join(veri)))
            yaz('\n2. DURAK: her kelime için adaylar üret, en uygununu seç')
            yaz('  (yalnızca birden çok seçeneği olan ya da tanınmayan kelimeler gösteriliyor)')
        elif ad == 'yazım':
            bekleyen = None
            for g, adaylar, c, kaynak in veri:
                if not c:
                    bekleyen = g                    # sonraki kelimeyle birleşecek
                    continue
                if kaynak == 'birleştirme':
                    yaz('  %-16s -> %-18s [birleştirme: iki parça tek kelime]' % (bekleyen + ' ' + g, ' '.join(c)))
                    continue
                digerleri = sorted(((m, ' '.join(a), k) for a, (m, k) in adaylar.items() if a != (g,)))
                if not digerleri and kaynak == 'aynı':
                    continue
                liste = ', '.join('%s (%s)' % (a, k) for _, a, k in digerleri[:5]) or 'yok'
                if ' '.join(c) == g:
                    yaz('  %-16s =  korundu%-11s adaylar: %s' % (g, '', liste))
                else:
                    yaz('  %-16s -> %-18s [%s]  adaylar: %s' % (g, ' '.join(c), kaynak, liste))
        elif ad == 'noktalama':
            parcalar = []
            for t, n_, k in veri:
                parcalar.append(t + (k or '') + ({'C': '[,]', 'P': '[.]', 'Q': '[?]'}.get(n_, '') if k is None else ''))
            yaz('\n3. DURAK: noktalama ([.] [,] [?] = modelin eklediği işaret)\n  ' + ' '.join(parcalar))
        elif ad == 'büyük harf':
            yaz('\n4. DURAK: büyük harf ve kesme işareti\n  ' + (', '.join('%s -> %s' % x for x in veri) or '(değişiklik yok)'))
        elif ad == 'çıktı':
            yaz('\nÇIKTI\n  ' + veri.strip())


def dosya_oku(yol):
    """UTF-8 (BOM'lu/BOM'suz), UTF-16 (Windows PowerShell'in `>` ile ürettiği) ve ANSI (cp1254) okur."""
    with open(yol, 'rb') as f:
        ham = f.read()
    if ham[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return ham.decode('utf-16')
    try:
        return ham.decode('utf-8-sig')
    except UnicodeDecodeError:
        return ham.decode('cp1254', errors='replace')


def main():
    ap = argparse.ArgumentParser(description='LLM kullanmadan Türkçe metin düzeltme')
    ap.add_argument('metin', nargs='?')
    ap.add_argument('--dosya', help='düzeltilecek metin dosyası (UTF-8, UTF-16 ya da ANSI)')
    ap.add_argument('--cikti', help='sonucu ekrana değil bu dosyaya yaz (UTF-8)')
    ap.add_argument('--model', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noktalama.crfsuite'))
    ap.add_argument('--ayrinti', action='store_true', help='yapılan kelime düzeltmelerini de göster')
    ap.add_argument('--yolculuk', action='store_true',
                    help='metnin aşamalardan nasıl geçtiğini adım adım göster (hangi kelimeye neden dokunuldu)')
    ap.add_argument('--eval', action='store_true',
                    help='düzeltmeyi Gemini ile uzaktan değerlendir (metni Google’a gönderir)')
    ap.add_argument('--eval-model', default=DEFAULT_MODEL,
                    help='--eval için Gemini model adı (varsayılan: %(default)s)')
    ap.add_argument('--paragraf', action='store_true',
                    help='deneysel paragraf bölmeyi aç (ölçümde rastgeleden iyi çıkmadı)')
    args = ap.parse_args()
    # Çıktı dosyaya/boruya yönlendirilmişse Windows varsayılan olarak ANSI yazar; UTF-8'e sabitle.
    for akis in (sys.stdout, sys.stderr):
        if hasattr(akis, 'reconfigure') and not akis.isatty():
            akis.reconfigure(encoding='utf-8')
    metin = dosya_oku(args.dosya) if args.dosya else (args.metin or sys.stdin.read())
    api_key = os.environ.get('GEMINI_API_KEY')
    if args.eval and not api_key:
        ap.error('--eval için GEMINI_API_KEY ortam değişkenini ayarlayın.')
    d = Duzeltici(args.model)
    izle = [] if args.yolculuk else None
    sonuc, degisiklikler = d.duzelt(metin, paragraf=args.paragraf, ayrinti=True, izle=izle)
    if args.yolculuk:
        yolculuk_yaz(metin, izle)
    elif args.cikti:
        with open(args.cikti, 'w', encoding='utf-8', newline='') as f:
            f.write(sonuc)
        print('Yazıldı:', args.cikti)
    else:
        print(sonuc)
    if args.ayrinti:
        print('\n--- kelime düzeltmeleri ---')
        for eski, yeni, kaynak in degisiklikler:
            print('  %-20s -> %-22s [%s]' % (eski, yeni, kaynak))
    if args.eval:
        try:
            evaluation = evaluate_correction(metin, sonuc, api_key, model=args.eval_model)
        except EvaluationError as exc:
            ap.exit(1, 'Gemini eval başarısız: %s\n' % exc)
        print('\n--- Gemini eval (öneridir; çıktı değiştirilmedi) ---')
        if not evaluation.issues:
            print('Belirgin bir sorun bulmadı; bu, çıktının doğru olduğunu garanti etmez.')
        for issue in evaluation.issues:
            print('  %r -> %r: %s' % (issue.original, issue.corrected, issue.reason))
            if issue.rule_suggestion:
                print('    Olası kural (elle doğrula): %s' % issue.rule_suggestion)


if __name__ == '__main__':
    main()
