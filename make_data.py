import nltk
import os

from get_oracle import get_actions_x

path_map = {
        'train': os.path.expanduser('~/data/comp-pcfg-data/parsed-data/ptb-train-gold-filtered.txt'),
        'dev': os.path.expanduser('~/data/comp-pcfg-data/parsed-data/ptb-valid-gold-filtered.txt'),
        }


def maketree(tr):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return '({} {})'.format(tr.label(), tr[0])
        nodes = []
        for x in tr:
            nodes.append(helper(x))
        label = tr.label()
        s = '({} {})'.format(label, ' '.join(nodes))
        return s
    s = helper(tr)
    return s

def maketemplate(tr):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return '{}'.format(tr.label())
        nodes = []
        for x in tr:
            nodes.append(helper(x))
        label = tr.label()
        s = '( {} {} )'.format(label, ' '.join(nodes))
        return s
    s = helper(tr)
    s = ' '.join(s.split())
    return s

def maketemplate_dummy(tr):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return 'X'
        nodes = []
        for x in tr:
            nodes.append(helper(x))
        s = '( {} )'.format(' '.join(nodes))
        return s
    s = helper(tr)
    s = ' '.join(s.split())
    return s

def maketemplate_word_slots(tr):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return '( )'
        nodes = []
        for x in tr:
            nodes.append(helper(x))
        s = ' '.join(nodes)
        return s
    s = helper(tr)
    s = '( ' + ' '.join(s.split()) + ' )'
    return s

def makesent(tr):
    return ' '.join([x for x in tr.leaves()])

def readfile(path):
    corpus = []
    with open(path) as f:
        for line in f:
            tr = nltk.Tree.fromstring(line)
            if len(tr.leaves()) > 40:
                continue
            o = {}
            o['actions'] = get_actions_x(line)
            o['tree'] = maketree(tr)
            o['s'] = makesent(tr)
            o['template'] = maketemplate(tr)
            o['dummy'] = maketemplate_dummy(tr)
            o['word_slots'] = maketemplate_word_slots(tr)
            corpus.append(o)
    return corpus

def writefile(path, corpus, mode='template'):
    with open(path, 'w') as f:
        for ex in corpus:
            if mode == 'template':
                f.write(ex['template'] + ' _ ')
                f.write(ex['s'])
                f.write(' _ ' + ex['tree'])
                f.write(' _ ' + ' '.join(ex['actions']))
            elif mode == 'dummy':
                f.write(ex['dummy'] + ' _ ')
                f.write(ex['s'])
                f.write(' _ ' + ex['tree'])
                f.write(' _ ' + ' '.join(ex['actions']))
            elif mode == 'word_slots':
                f.write(ex['word_slots'] + ' _ ')
                f.write(ex['s'])
                f.write(' _ ' + ex['tree'])
                f.write(' _ ' + ' '.join(ex['actions']))
            elif mode == 'prefix':
                f.write('a ' * 20 + '_ ')
                f.write(ex['s'])
                f.write(' _ ' + ex['tree'])
                f.write(' _ ' + ' '.join(ex['actions']))
            elif mode == 'text_only':
                f.write(ex['s'])
            else:
                f.write('_ ' + ex['s'])
                f.write(' _ ' + ex['tree'])
                f.write(' _ ' + ' '.join(ex['actions']))
            f.write('\n')

actions = set()
for k, v in path_map.items():
    corpus = readfile(v)
    #writefile('{}-template.txt'.format(k), corpus, mode='template')
    #writefile('{}-raw.txt'.format(k), corpus, mode='raw')
    #writefile('{}-dummy.txt'.format(k), corpus, mode='dummy')
    #writefile('{}-word_slots.txt'.format(k), corpus, mode='word_slots')
    #writefile('{}-prefix.txt'.format(k), corpus, mode='prefix')
    writefile('{}-text_only.txt'.format(k), corpus, mode='text_only')
    for ex in corpus:
        for x in ex['actions']:
            actions.add(x)
with open('actions.vocab', 'w') as f:
    for x in sorted(actions):
        f.write('{}\n'.format(x))

