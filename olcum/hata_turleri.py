#!/usr/bin/env python3
"""Gerçek tweet setinde hata TÜRÜNE göre döküm: hangi tür hataların kaçı düzeltildi, kaçı bırakıldı?

  python olcum/hata_turleri.py                                   # bu depo, geliştirme bölümü
  python olcum/hata_turleri.py --sistem zemberek-normalizer      # başka bir yöntem
  python olcum/hata_turleri.py --ornek 12                        # her türden düzelmeyen 12 örnek

Türler veri kümesinin kendi etiketleridir (Koksal vd., 2020). Sık görülenler:
  deascii = Türkçe karakter eksik (aksam)     accent = konuşma dili (gelcem)     int = bilerek yapılmış (çoook)
  unint   = yazım hatası                      sep    = ayrı yazılmalı (bende)    joint = bitişik yazılmalı (bir çok)
  vowel   = ünlüsüz kısaltma (tmm)            proper = özel isim
Yeni bir kural ya da aday kaynağı eklemeden önce buraya bakın: en çok hata hangi türde kalıyor?
Varsayılan bölüm "gelistirme"dir; test bölümüne bakarak karar vermeyin, o yalnızca son sonucu bildirmek içindir.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import veri        # noqa: E402
import olcut       # noqa: E402
import sistemler   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sistem', default='tr-duzelt')
    ap.add_argument('--kod', help='tr-duzelt için başka bir sürümün klasörü')
    ap.add_argument('--bolum', default='gelistirme', choices=['gelistirme', 'test'])
    ap.add_argument('--ornek', type=int, default=6, help='her türden gösterilecek düzelmeyen örnek sayısı')
    a = ap.parse_args()

    sistem = sistemler.kur(a.sistem, a.kod)
    tw = veri.tweetler()
    tw = tw[:1000] if a.bolum == 'gelistirme' else tw[1000:]
    say = collections.defaultdict(collections.Counter)
    ornek = collections.defaultdict(list)
    for girdi, dogru, etiket in tw:
        cikti = sistem(girdi)
        altin = olcut.duzenlemeler(girdi, dogru)
        yapilan = olcut.duzenlemeler(girdi, cikti)
        for bas, son, yeni in altin:
            if etiket is None or son <= bas:
                tur = '(etiketsiz)'
            else:
                tur = '&'.join(sorted(set(x.replace('OOV-ill_formed-', '').replace('OOV-well_formed-', 'wf-')
                                          for x in etiket[bas].split('&'))))
            cakisan = [x for x in yapilan if x[0] < son and bas < x[1]]
            if (bas, son, yeni) in yapilan:
                durum = 'düzeltildi'
            elif cakisan:
                durum = 'yanlış'
            else:
                durum = 'bırakıldı'
            say[tur][durum] += 1
            if durum != 'düzeltildi' and len(ornek[tur]) < a.ornek:
                ornek[tur].append('%s→%s%s' % (' '.join(girdi[bas:son]), ' '.join(yeni),
                                               ' [çıktı: %s]' % ' '.join(cakisan[0][2]) if cakisan else ''))
    print('sistem: %s | bölüm: %s' % (sistem.ad, a.bolum))
    print('%-22s %5s  %10s %9s %7s   düzelmeyen örnekler (yazılan→doğrusu)' % ('tür', 'hata', 'düzeltildi', 'bırakıldı', 'yanlış'))
    for tur, c in sorted(say.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        if n < 8:
            continue
        print('%-22s %5d  %9.0f%% %8.0f%% %6.0f%%   %s' % (
            tur, n, 100.0 * c['düzeltildi'] / n, 100.0 * c['bırakıldı'] / n, 100.0 * c['yanlış'] / n,
            '; '.join(ornek[tur])))


if __name__ == '__main__':
    main()
