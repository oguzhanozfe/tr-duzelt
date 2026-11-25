#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
egit_noktalama.py — tr_duzelt.py'nin kullandığı noktalama + büyük harf CRF modelini eğitir.

Fikir (kendinden denetimli): düzgün yazılmış Türkçe metinden noktalama ve büyük harfleri SİL;
model, silineni geri koymayı öğrenir. Elle etiketleme gerekmez, temiz metin yeterlidir.

Etiket = <harf durumu><noktalama>
   harf durumu : l (küçük) | c (İlk harf büyük: özel isim) | u (TÜMÜ BÜYÜK: kısaltma)
   noktalama   : O (yok) | C (virgül) | P (nokta) | Q (soru işareti)   -> kelimeden SONRA gelen

Teslim edilen noktalama.crfsuite şu verilerle eğitildi (≈858 bin kelime):
   TrMor2018 (github.com/ai-ku/TrMor2018, MIT)         — gazete + roman, belge düzeninde
   UD Turkish BOUN, Kenet, Penn, Tourism, FrameNet, GB, PUD (CC BY-SA)
      (github.com/UniversalDependencies/UD_Turkish-BOUN vb.; her deponun .conllu dosyaları)
Ticari üründe kullanmadan önce bu lisansları kontrol edin; en temizi aynı betikle kendi
metninizle (ör. uygulamanızın alanından düzgün yazılmış metinler) yeniden eğitmektir.

Ayrılmış test sonuçları (bu modelle, Viterbi çözümleme):
                        nokta F1   virgül F1   soru F1   büyük harf F1
   gazete (21,6 bin kel.)   0,806      0,332     0,580      0,784
   roman  (13,2 bin kel.)   0,760      0,194     0,580      0,584
   yalnız dil modeli, CRF'siz (gazete / roman): nokta 0,782 / 0,705
Alan dışı (150 öğrenci yazısı, 88 bin kelime): nokta 0,660 — virgül 0,218 — soru 0,507
   Bu ölçümün verisi: Bilkent Turkish Writings Dataset (S. F. Yilmaz, 2025, doi:10.5281/zenodo.15498155;
   özgün kaynak Bilkent Üniversitesi Türkçe Birimi). Akademik kullanım lisanslı olduğu için yalnızca
   ölçümde kullanıldı; teslim edilen model bu veriyle EĞİTİLMEDİ.

Kullanım:
  python egit_noktalama.py --trmor2018 trmor2018.train --ud a.conllu b.conllu ... \
      --cikti noktalama.crfsuite --iter 200 --c1 1.0 --c2 0.1 --minfreq 3 --test-kitap 14 17
"""
import os
import re
import sys
import json
import random
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tr_duzelt import tr_lower, tokenize, is_word, crf_features, lm_tahmin, DilModeli  # noqa: E402

PUNCT_PRI = {'O': 0, 'C': 1, 'P': 2, 'Q': 3}
OPEN_QUOTES = {'"', "'", '“', '‘', '«', '``', "''", '(', '-', '—', '–'}
SUFFIX_LIKE = re.compile(r"^[a-zçğıöşü]{1,6}$")


# ----------------------------- derlem okuyucular -----------------------------

def read_trmor(path, skip_titles=True):
    """TrMor2006/2018: satır başına bir belirteç, XML benzeri etiketlerle belge/cümle sınırı.
    Döndürür: belgeler = [ [cümle=[belirteç,...]], ... ]"""
    docs, doc, sent, in_title = [], [], None, False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            tok = line.rstrip('\n').split('\t')[0].strip()
            if not tok:
                continue
            if tok.startswith('<') and tok.endswith('>') and len(tok) > 2:
                tag = tok[1:-1].split()[0].upper()
                if tag == 'DOC':
                    if doc:
                        docs.append(doc)
                    doc = []
                elif tag == 'TITLE':
                    in_title = True
                elif tag == '/TITLE':
                    in_title = False
                elif tag == 'S':
                    sent = []
                elif tag == '/S':
                    if sent and not (in_title and skip_titles):
                        doc.append(sent)
                    sent = None
                continue
            if sent is not None:
                sent.append(tok)
    if doc:
        docs.append(doc)
    return docs


def read_ud(path):
    """CoNLL-U: yalnızca '# text = ...' satırlarını kullan, kendi belirteçleyicimizle böl."""
    sents = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('# text = '):
                sents.append(tokenize(line[9:].strip()))
    return [sents]  # tek "belge"


def merge_apostrophes(toks):
    """'Derviş', "'", 'in'  ->  "Derviş'in" (bazı derlemlerde kesme ayrı belirteç)."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if (t in ("'", '’') and out and i + 1 < len(toks) and is_word(out[-1])
                and (out[-1][0].isupper() or out[-1][0].isdigit())
                and SUFFIX_LIKE.match(toks[i + 1])):
            out[-1] = out[-1] + "'" + toks[i + 1]
            i += 2
            continue
        out.append(t)
        i += 1
    return out


# ----------------------------- etiket üretimi -----------------------------

def sentence_items(toks):
    """Cümle belirteçleri -> [(özgün_kelime, noktalama_etiketi, cümle_başı_gibi_mi)]
    Sonu . ? ! ile bitmeyen cümleler (başlık vb.) None döner."""
    toks = merge_apostrophes(toks)
    items, initial_like = [], True
    for t in toks:
        if is_word(t):
            items.append([t, 'O', initial_like])
            initial_like = False
        else:
            if items:
                if t in ('.', '!'):
                    lab = 'P'
                elif t == '?':
                    lab = 'Q'
                elif t in (',', ';'):
                    lab = 'C'
                elif t in ('...', '…', '..'):
                    lab = 'E'
                else:
                    lab = None
                if lab and PUNCT_PRI.get(lab, 2) > PUNCT_PRI.get(items[-1][1], 2):
                    items[-1][1] = lab
            if t in OPEN_QUOTES:
                initial_like = True  # tırnak/parantez sonrası büyük harf güvenilmez -> istatistiğe bak
    if not items:
        return None
    last = items[-1][1]
    if last not in ('P', 'Q', 'E'):
        return None
    letters = ''.join(ch for it in items for ch in it[0] if ch.isalpha())
    if len(letters) > 3 and letters.upper() == letters and letters.lower() != letters:
        return None  # tamamı büyük harf: başlık
    n = len(items)
    for k, it in enumerate(items):
        if k == n - 1:
            it[1] = 'Q' if it[1] == 'Q' else 'P'
        else:  # cümle içi: nokta = kısaltma (yok say), üç nokta = duraksama (virgül)
            if it[1] == 'P':
                it[1] = 'O'
            elif it[1] == 'E':
                it[1] = 'C'
    return items


def core_of(tok):
    return tr_lower(re.split(r"['’]", tok)[0])


def build_case_stats(all_docs):
    cap, tot, upp = Counter(), Counter(), Counter()
    for doc in all_docs:
        for items in doc:
            for w, _, init in items:
                if init or not w[0].isalpha():
                    continue
                c = core_of(w)
                tot[c] += 1
                if w[0].isupper():
                    cap[c] += 1
                core_orig = re.split(r"['’]", w)[0]
                if len(core_orig) > 1 and core_orig.isupper():
                    upp[c] += 1
    return cap, tot, upp


def case_label(w, init, cap, tot, upp):
    if not w[0].isalpha():
        return 'l'
    c = core_of(w)
    core_orig = re.split(r"['’]", w)[0]
    p_cap = cap[c] / tot[c] if tot[c] else None
    if len(core_orig) > 1 and core_orig.isupper():
        if tot[c] and upp[c] / tot[c] > 0.5:
            return 'u'
        if not init and len(core_orig) <= 5 and (p_cap is None or p_cap > 0.5):
            return 'u'
        return 'c' if (p_cap is not None and p_cap > 0.5) else 'l'
    if w[0].isupper():
        if not init:
            return 'c'
        if "'" in w or '’' in w:
            return 'c'
        return 'c' if (p_cap is not None and p_cap > 0.5) else 'l'
    return 'l'


def clean_word(w):
    return tr_lower(w.replace("'", '').replace('’', ''))


def docs_to_sequences(docs, cap, tot, upp, rng, min_s=1, max_s=10):
    """Belgeleri 1-10 cümlelik pencerelere böl -> (kelimeler, etiketler) dizileri."""
    seqs = []
    for doc in docs:
        i = 0
        while i < len(doc):
            k = rng.randint(min_s, max_s)
            words, labels = [], []
            for items in doc[i:i + k]:
                for w, p, init in items:
                    cw = clean_word(w)
                    if not cw:
                        continue
                    words.append(cw)
                    labels.append(case_label(w, init, cap, tot, upp) + p)
            if words:
                seqs.append((words, labels))
            i += k
    return seqs


# ----------------------------- değerlendirme -----------------------------

def evaluate(tagger, seqs, name, lm, yalniz_lm=False):
    """yalniz_lm=True: CRF yerine yalnızca 2-gram dil modeli ("gizli olay") tahmini."""
    tp, fp, fn = Counter(), Counter(), Counter()
    cap_tp = cap_fp = cap_fn = 0
    for words, gold in seqs:
        if yalniz_lm:
            pred = ['l' + p for p in lm_tahmin(words, lm)]
        else:
            tagger.set(crf_features(words, lm))
            pred = tagger.tag()
        for g, p in zip(gold, pred):
            gp, pp = g[1], p[1]
            if gp == pp:
                tp[gp] += 1
            else:
                fp[pp] += 1
                fn[gp] += 1
            gc, pc = g[0] != 'l', p[0] != 'l'
            cap_tp += gc and pc
            cap_fp += pc and not gc
            cap_fn += gc and not pc
    res = {}
    print('--- %s (%d dizi) ---' % (name, len(seqs)))

    def prf(t, f_p, f_n):
        pr = t / (t + f_p) if t + f_p else 0.0
        rc = t / (t + f_n) if t + f_n else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    for lab, nm in (('P', 'nokta'), ('C', 'virgül'), ('Q', 'soru')):
        pr, rc, f1 = prf(tp[lab], fp[lab], fn[lab])
        res[nm] = dict(P=round(pr, 3), R=round(rc, 3), F1=round(f1, 3), n=tp[lab] + fn[lab])
        print('  %-7s P=%.3f R=%.3f F1=%.3f (n=%d)' % (nm, pr, rc, f1, tp[lab] + fn[lab]))
    if not yalniz_lm:
        pr, rc, f1 = prf(cap_tp, cap_fp, cap_fn)
        res['buyuk_harf'] = dict(P=round(pr, 3), R=round(rc, 3), F1=round(f1, 3), n=cap_tp + cap_fn)
        print('  %-7s P=%.3f R=%.3f F1=%.3f (n=%d)' % ('b.harf', pr, rc, f1, cap_tp + cap_fn))
    return res


# ----------------------------- ana akış -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trmor2018')
    ap.add_argument('--trmor2006')
    ap.add_argument('--ud', nargs='*', default=[])
    ap.add_argument('--cikti', default='noktalama.crfsuite')
    ap.add_argument('--iter', type=int, default=200)
    ap.add_argument('--c1', type=float, default=1.0)
    ap.add_argument('--c2', type=float, default=0.1)
    ap.add_argument('--minfreq', type=int, default=3)
    ap.add_argument('--ornek', type=float, default=1.0, help='eğitim verisinin kullanılacak oranı')
    ap.add_argument('--rapor', default=None)
    ap.add_argument('--lmsiz', action='store_true', help='dil modeli özniteliklerini kullanma (karşılaştırma için)')
    ap.add_argument('--test-kitap', dest='test_kitap', type=int, nargs='*', default=[],
                    help='TrMor2018 içinde teste ayrılacak belge sıra numaraları (0 tabanlı)')
    args = ap.parse_args()

    import pycrfsuite
    rng = random.Random(13)

    raw_sets = []  # (ad, belgeler[ham cümleler])
    if args.trmor2018:
        raw_sets.append(('trmor2018', read_trmor(args.trmor2018)))
    if args.trmor2006:
        raw_sets.append(('trmor2006', read_trmor(args.trmor2006)))
    for p in args.ud:
        raw_sets.append((os.path.basename(p), read_ud(p)))

    item_sets = []
    for name, docs in raw_sets:
        out = []
        kept = dropped = 0
        for doc in docs:
            d = []
            for s in doc:
                it = sentence_items(s)
                if it is None:
                    dropped += 1
                else:
                    d.append(it)
                    kept += 1
            out.append(d)          # boş belgeler de yerinde kalsın: sıra numaraları kaymasın
        item_sets.append((name, out))
        print('%-28s belge=%5d cümle=%6d (atılan %d)' % (name, sum(1 for d in out if d), kept, dropped), flush=True)

    # test (eğitimde hiç görülmez): TrMor2018'in son %10 belgesi (gazete) + --test-kitap ile verilen belgeler
    train_docs, test_docs, kitap_docs = [], [], []
    for name, docs in item_sets:
        if name == 'trmor2018':
            k = int(len(docs) * 0.9)
            for di, d in enumerate(docs):
                if di in args.test_kitap:
                    kitap_docs.append(d)
                elif di >= k:
                    test_docs.append(d)
                else:
                    train_docs.append(d)
        else:
            train_docs += docs

    cap, tot, upp = build_case_stats(train_docs)
    train = docs_to_sequences(train_docs, cap, tot, upp, rng)
    test = docs_to_sequences(test_docs, cap, tot, upp, random.Random(7), min_s=4, max_s=10)
    kitap = docs_to_sequences(kitap_docs, cap, tot, upp, random.Random(7), min_s=4, max_s=10)
    if args.ornek < 1.0:
        rng.shuffle(train)
        train = train[:int(len(train) * args.ornek)]
    n_tok = sum(len(w) for w, _ in train)
    labs = Counter(l for _, ls in train for l in ls)
    print('eğitim: %d dizi, %d kelime | test: %d dizi, %d kelime' %
          (len(train), n_tok, len(test), sum(len(w) for w, _ in test)), flush=True)
    print('etiket dağılımı:', dict(labs.most_common()), flush=True)

    lm = None if args.lmsiz else DilModeli()
    tr = pycrfsuite.Trainer(verbose=True)
    for words, labels in train:
        tr.append(crf_features(words, lm), labels)
    tr.select('lbfgs')
    tr.set_params({'c1': args.c1, 'c2': args.c2, 'max_iterations': args.iter,
                   'feature.minfreq': args.minfreq, 'feature.possible_transitions': True})
    tr.train(args.cikti)

    tagger = pycrfsuite.Tagger()
    tagger.open(args.cikti)
    rapor = {'egitim_kelime': n_tok}
    rapor['gazete_crf'] = evaluate(tagger, test, 'CRF — ayrılmış gazete belgeleri', lm)
    if kitap:
        rapor['kitap_crf'] = evaluate(tagger, kitap, 'CRF — ayrılmış kitap belgeleri', lm)
    if lm is not None:
        rapor['gazete_yalniz_lm'] = evaluate(None, test, 'yalnız dil modeli — gazete', lm, yalniz_lm=True)
        if kitap:
            rapor['kitap_yalniz_lm'] = evaluate(None, kitap, 'yalnız dil modeli — kitap', lm, yalniz_lm=True)
    if args.rapor:
        with open(args.rapor, 'w', encoding='utf-8') as f:
            json.dump(rapor, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
