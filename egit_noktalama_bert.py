#!/usr/bin/env python3
"""BERTurk tabanlı noktalama etiketleyicisini eğitir (isteğe bağlı; CRF'in yerine değil, yanına).

Veri ve yöntem egit_noktalama.py ile aynı: düzgün metinden noktalama silinir, model geri koymayı öğrenir. Etiket,
kelimeden SONRA gelen işaret: O (yok), C (virgül), P (nokta), Q (soru). Büyük harfi yine CRF verir.
Teste ayrılanlar da aynı: TrMor2018'in son %10 belgesi ve --test-kitap belgeleri. UD Turkish-IMST hiç kullanılmaz;
olcum/noktalama_olc.py onunla ölçer.

Ek kurulum:  pip install torch transformers   (ekran kartıyla birkaç dakika; işlemciyle saatler)
Kullanım:
  python egit_noktalama_bert.py --trmor2018 trmor2018.train --ud *.conllu --cikti noktalama_bert --test-kitap 14 17
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egit_noktalama import read_trmor, read_ud, sentence_items, docs_to_sequences   # noqa: E402
from collections import Counter                                                     # noqa: E402

ETIKETLER = ['O', 'C', 'P', 'Q']
EN_COK_KELIME = 120          # bir dizide en çok bu kadar kelime (512 alt parçanın rahatça altında)


def gurultulu(seqs, oran, rng):
    """Girdi kelimelerinin bir kısmını boz (etiketler aynı): çalışma anında model, yazım hataları tam düzelmemiş metin
    görür. Türkçe karaktersiz yazım, harf düşmesi, harf tekrarı, yan yana iki harfin yer değiştirmesi."""
    ascii_ = str.maketrans('çğıöşü', 'cgiosu')
    out = []
    for w, e in seqs:
        yeni = []
        for x in w:
            if len(x) >= 3 and x.isalpha() and rng.random() < oran:
                k = rng.randrange(1, len(x))
                x = rng.choice([x.translate(ascii_), x[:k] + x[k + 1:], x[:k] + x[k] + x[k:],
                                x[:k - 1] + x[k] + x[k - 1] + x[k + 1:]]) or x
            yeni.append(x)
        out.append((yeni, e))
    return out


def parcala(seqs):
    out = []
    for w, e in seqs:
        for b in range(0, len(w), EN_COK_KELIME):
            out.append((w[b:b + EN_COK_KELIME], [x[1] for x in e[b:b + EN_COK_KELIME]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trmor2018', required=True)
    ap.add_argument('--ud', nargs='*', default=[])
    ap.add_argument('--test-kitap', dest='test_kitap', type=int, nargs='*', default=[])
    ap.add_argument('--model', default='dbmdz/bert-base-turkish-cased')
    ap.add_argument('--cikti', default='noktalama_bert')
    ap.add_argument('--tur', type=int, default=3, help='kaç tur (her turda cümleler başka pencerelere bölünür)')
    ap.add_argument('--parti', type=int, default=32)
    ap.add_argument('--oran', type=float, default=3e-5)
    ap.add_argument('--gurultu', type=float, default=0.0,
                    help='eğitim girdisindeki kelimelerin bu oranını boz (ör. 0.05); test verisi bozulmaz')
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    belgeler = [('trmor2018', read_trmor(a.trmor2018))] + [(os.path.basename(p), read_ud(p)) for p in a.ud]
    egitim, test = [], []
    for ad, docs in belgeler:
        docs = [[it for it in (sentence_items(s) for s in d) if it] for d in docs]
        if ad == 'trmor2018':
            k = int(len(docs) * 0.9)
            for i, d in enumerate(docs):
                (test if i in a.test_kitap or i >= k else egitim).append(d)
        else:
            egitim += docs
    bos = Counter()
    test_seqs = parcala(docs_to_sequences(test, bos, bos, bos, random.Random(7), min_s=4, max_s=10))
    print('eğitim belgeleri: %d, test dizileri: %d' % (len(egitim), len(test_seqs)), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    cihaz = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(ETIKETLER), id2label=dict(enumerate(ETIKETLER)),
        label2id={e: i for i, e in enumerate(ETIKETLER)}).to(cihaz)

    def kodla(parti):
        g = tok([w for w, _ in parti], is_split_into_words=True, truncation=True, max_length=512,
                padding=True, return_tensors='pt')
        y = torch.full(g['input_ids'].shape, -100, dtype=torch.long)
        for b, (_, e) in enumerate(parti):
            onceki = None
            for k, wi in enumerate(g.word_ids(b)):
                if wi is not None and wi != onceki:
                    y[b, k] = ETIKETLER.index(e[wi])      # yalnızca kelimenin ilk alt parçası etiketlenir
                onceki = wi
        return g.to(cihaz), y.to(cihaz)

    turlar = [parcala(gurultulu(docs_to_sequences(egitim, bos, bos, bos, random.Random(13 + t)), a.gurultu,
                                random.Random(99 + t))) for t in range(a.tur)]
    adim = sum((len(x) + a.parti - 1) // a.parti for x in turlar)
    opt = torch.optim.AdamW(model.parameters(), lr=a.oran, weight_decay=0.01)
    plan = get_linear_schedule_with_warmup(opt, int(0.06 * adim), adim)
    olcek = torch.amp.GradScaler(enabled=cihaz == 'cuda')
    t0, n = time.time(), 0
    for t, seqs in enumerate(turlar):
        random.Random(100 + t).shuffle(seqs)
        model.train()
        for b in range(0, len(seqs), a.parti):
            g, y = kodla(seqs[b:b + a.parti])
            with torch.autocast(cihaz, dtype=torch.float16, enabled=cihaz == 'cuda'):
                kayip = model(**g, labels=y).loss
            olcek.scale(kayip).backward()
            olcek.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            olcek.step(opt)
            olcek.update()
            opt.zero_grad()
            plan.step()
            n += 1
            if n % 200 == 0:
                print('  tur %d, adım %d/%d, kayıp %.4f, %.0f sn' % (t + 1, n, adim, kayip.item(), time.time() - t0),
                      flush=True)
        # ayrılmış TrMor belgelerinde F1
        model.eval()
        tp, fp, fn = Counter(), Counter(), Counter()
        for b in range(0, len(test_seqs), 64):
            parti = test_seqs[b:b + 64]
            g, y = kodla(parti)
            with torch.no_grad():
                p = model(**g).logits.argmax(-1)
            for gi, pi in zip(y[y != -100].tolist(), p[y != -100].tolist()):
                if gi == pi:
                    tp[gi] += 1
                else:
                    fp[pi] += 1
                    fn[gi] += 1
        f1 = {ETIKETLER[e]: 2 * tp[e] / max(1, 2 * tp[e] + fp[e] + fn[e]) for e in (1, 2, 3)}
        print('tur %d bitti — ayrılmış TrMor: nokta F1 %.3f, virgül F1 %.3f, soru F1 %.3f' % (
            t + 1, f1['P'], f1['C'], f1['Q']), flush=True)
    model.half().save_pretrained(a.cikti)          # yarım duyarlık: ~220 MB
    tok.save_pretrained(a.cikti)
    print('yazıldı:', a.cikti)


if __name__ == '__main__':
    main()
