import re
import os
import glob
import collections
import argparse
import sacrebleu
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def float_trunc(x, significant=2):
    if x > 1:
        return float(format(x, '.{}f'.format(significant)))
    else:
        return float(format(x, '.{}g'.format(significant)))


def parse_examples(log):
    results = collections.defaultdict(dict)
    with open(log, "r", encoding="utf-8") as fin:
        for line in fin:
            match = re.match("([SDCT]{1})-(\d+)", line)
            if match:
                role, idx = match.groups()
                idx = int(idx)
                seq = line.split("\t")[-1].strip()
                if role == "C":
                    if role in results[idx]:
                        results[idx][role].append(seq)
                    else:
                        results[idx][role] = [seq]
                else:
                    results[idx][role] = seq
    entries = []
    for key, values in results.items():
        entries.append(dict(I=key, **values))
    entries.sort(key=lambda x: x["I"])
    return entries


def parse_bleu(log):
    with open(log, "r", encoding="utf-8") as fin:
        for line in fin:
            pass
        bleu = re.findall("(?<=BLEU4 = )\d+\.\d+", line)
    if not bleu:
        return None
    else:
        return float(bleu[0])


def parse_tb_dev(path):
    path = os.path.join(path, 'valid')
    if not os.path.exists(path):
        return
    event_acc = EventAccumulator(path).Reload()
    keys = event_acc.scalars.Keys()
    results = {}
    train_minutes = None
    for key in keys:
        if "best" in key:
            # note that only one best exists
            if train_minutes is None:
                train_minutes = (event_acc.Scalars(key)[-1].wall_time - \
                                 event_acc.Scalars(key)[0].wall_time) // 60
                results["minutes"] = int(train_minutes)
            value = event_acc.Scalars(key)[-1].value
            if value == float('nan'):
                value = event_acc.Scalars(key)[-2].value
            best_step = min(s.step for s in event_acc.Scalars("best_loss") if s.value == value)
            key = key.replace("best", "dev")
            results[key.replace("best", "dev")] = float_trunc(value)
            results["dev_step"] = best_step
    return results


def compute_chrf(preds, refs, sent_chrf=False, print_key=False):
    args = argparse.Namespace(chrf_order=6, chrf_beta=3, chrf_whitespace=False)
    chrf = sacrebleu.CHRF(args)
    corpus_stats = [0] * (chrf.order * 3)
    sent_chrfs = []
    for pred, ref in zip(preds, refs):
        stats = chrf.get_sentence_statistics(pred, [ref])
        if sent_chrf:
            chrf = float_trunc(chrf.compute_chrf(stats, chrf.order, chrf.beta).score * 100)
            sent_chrfs.append(chrf)
        for i, stat in enumerate(stats):
            corpus_stats[i] += stat
    corpus_chrf = chrf.compute_chrf(corpus_stats, chrf.order, chrf.beta)
    if sent_chrf:
        return sent_chrfs, float_trunc(corpus_chrf.score * 100)
    else:
        return float_trunc(corpus_chrf.score * 100)


def compute_bleu(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="none", force=True)
    return float_trunc(bleu.score)


# for unigram f1
def get_word2bin(filename, bpe=True):
    if not os.path.exists(filename):
        return None

    word2count = collections.Counter()
    with open(filename, encoding="utf-8") as fin:
        for line in fin.readlines():
            if bpe:
                word2count.update(line.replace("@@ ", "").strip().split())
            else:
                word2count.update(''.join(line.split()).replace("▁", " ").strip().split())
    word2bin = dict()
    for word, count in word2count.items():
        word2bin[word] = math.ceil(math.log10(count + 1))
    return word2bin


def compute_binned_unigram_f1(results, word2bin, last=True):
    if word2bin is None:
        return {}
    target = collections.Counter()
    hypo = collections.Counter()
    correct = collections.Counter()
    for entry in results:
        h = collections.Counter(entry.D.split())
        t = collections.Counter(entry.T.split())
        c = t - (t - h)
        hypo.update({word2bin.get(w, 0): count for w, count in h.items()})
        target.update({word2bin.get(w, 0): count for w, count in t.items()})
        correct.update({word2bin.get(w, 0): count for w, count in c.items()})
    bin2results = {}
    for key in sorted(correct.keys()):
        p = correct[key] / hypo[key]
        r = correct[key] / target[key]
        f1 = 2 * p * r / (p + r)
        bin2results[key] = {"p": p, "r": r, "f1": f1}
    results = {}
    for key, value in bin2results.items():
        if last and key > 2:
            break
        results["{}_f1".format(key)] = value["f1"]
    return results


