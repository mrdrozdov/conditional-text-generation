import argparse
import json
import numpy as np
import collections


parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./results_raw_256/lm_xent.txt,./results_num_words_256/lm_xent.txt,./results_template_256/lm_xent.txt')
parser.add_argument('--names', default='Empty,Number of Words,Look-N')
parser.add_argument('--html', default='./dist/index.html')
options = parser.parse_args()


class Sentence:
    def __init__(self, s):
        self.s = s

        self._score = None

    @property
    def score(self):
        if self._score is None:
            self._score = np.mean([ex['p'] for ex in self.s if ex['p'] >= 0])
        return self._score



class Group:
    def __init__(self, s_lst):
        self.s_lst = s_lst


def sort_groups(groups):
    keys = []
    other = []

    for i, g in enumerate(groups):
        s_lst = g.s_lst
        baseline = s_lst[0].score
        approach = s_lst[-1].score
        keys.append(approach - baseline)

    new_groups = []

    for i in np.argsort(keys)[::-1]:
        new_groups.append(groups[i])

    return new_groups



class Token:
    def __init__(self, o):
        self.o = o


def readfile(path):
    def check(s):
        return len(s) > 20
    with open(path) as f:
        s = []
        corpus = []
        for line in f:
            line = line.strip()
            if not line or len(line) == 0:
                if check(s):
                    corpus.append(Sentence(s))
                    s = []
                continue
            ex = json.loads(line)
            s.append(ex)
        if check(s):
            corpus.append(Sentence(s))
    return corpus


corpus_lst = []
for path in options.src.split(','):
    corpus_lst.append(readfile(path))



def make_css():
    css_template = """<style type="text/css" style="display: none">
    {}
</style>
"""

    lookup = []
    sofar = 0

    body = ''

    k = 'red'
    boundary = 0.1
    mass = boundary - sofar
    buckets = 10
    for i in range(buckets):
        alpha = 0.5 + 0.5 * (i+1)/buckets
        assert alpha >= 0 and alpha <= 1
        clz = '{}-{}'.format(k, i)
        row = '.{} {{ background: rgba(255, 0, 0, {}); }}'.format(clz, alpha)
        body += row + '\n'
        sofar += (mass/buckets)
        lookup.append([sofar, clz])

    k = 'yellow'
    boundary = 0.5
    mass = boundary - sofar
    buckets = 10
    for i in range(buckets):
        alpha = 0.5 + 0.5 * (i+1)/buckets
        assert alpha >= 0 and alpha <= 1
        clz = '{}-{}'.format(k, i)
        row = '.{} {{ background: rgba(255, 255, 0, {}); }}'.format(clz, alpha)
        body += row + '\n'
        sofar += mass/buckets
        lookup.append([sofar, clz])

    k = 'green'
    boundary = 1.0
    mass = boundary - sofar
    buckets = 10
    for i in range(buckets):
        alpha = 0.5 + 0.5 * (i+1)/buckets
        assert alpha >= 0 and alpha <= 1
        clz = '{}-{}'.format(k, i)
        row = '.{} {{ background: rgba(0, 255, 0, {}); }}'.format(clz, alpha)
        body += row + '\n'
        sofar += mass/buckets
        lookup.append([sofar, clz])

    assert sofar < 1.0001, sofar

    def convert_fn(p):
        assert p >= 0 and p <= 1

        new_clz = None
        lb = 0
        for i, v in enumerate(lookup):
            b, clz = v
            if p >= lb and p <= b:
                new_clz = clz
                break
            lb = b

        assert new_clz is not None, (p, lookup)

        return new_clz


    return css_template.format(body), convert_fn


with open(options.html, 'w') as f:
    css, convert_p = make_css()
    f.write(css)

    groups = [Group(s_lst) for s_lst in zip(*corpus_lst)]

    groups = sort_groups(groups)

    for g in groups:
        for i_corpus, s in enumerate(g.s_lst):
            name = options.names.split(',')[i_corpus]
            block = ''
            for ex in s.s:
                tok = ex['tok']
                if ex['p'] == -1:
                    block += tok
                    continue
                clz = convert_p(ex['p'])
                block += '<span class="{}">{}</span>'.format(clz, tok)
            f.write('<p>{} ({})</p>\n'.format(block, name))
            f.write('\n')
