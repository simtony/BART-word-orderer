import copy
import random
import shutil
from functools import reduce
import itertools
import re

random.seed(10)


def norm_token(token):
    if token in ["-LRB-", "-LCB-"]:
        return "("
    if token in ["-RRB-", "-RCB-"]:
        return ")"
    return re.sub(r"\\", "", token)


def norm_tag(tag):
    if tag in [",", ".", "$", "``", "''", ":", "-RRB-", "-LRB-", "#"]:
        return "PUNCT"
    else:
        return tag


class Node(object):
    def __init__(self, word, pos, parent_id, rel):
        self.word = " " + word
        self.pos = "<" + pos + ":>"
        self.parent_id = parent_id
        self.rel = "<:" + rel + ">"

        self.parent = None
        self.children = []
        self.start = "<[>"
        self.end = "<]>"

    def set_parent(self, parent):
        self.parent = parent
        if parent is not None:
            self.parent.children.append(self)

    def linearize(self, bracket=True, shuffle=True, rel=True):
        tokens = []
        if bracket:
            tokens.append(self.start)
        tokens.append(self.word)
        if self.pos is not None:
            tokens.append(self.pos)

        if not self.children:
            if bracket:
                tokens.append(self.end)
            return tokens
        child_ids = list(range(len(self.children)))
        if shuffle:
            random.shuffle(child_ids)
        for i in child_ids:
            child = self.children[i]
            if rel:
                tokens.append(child.rel)
            tokens.extend(child.linearize(bracket=bracket, shuffle=shuffle, rel=rel))

        if bracket:
            tokens.append(self.end)
        return tokens

    def __repr__(self):
        return "('{}', '{}')".format(self.word, self.pos)


def partial_tree(nodes, keep_pos, keep_dep):
    nodes = copy.deepcopy(nodes)
    roots = []
    num_nodes = len(nodes)
    for i in random.sample(range(num_nodes), int((1 - keep_pos) * num_nodes)):
        nodes[i].pos = None
    for i in random.sample(range(num_nodes), int((1 - keep_dep) * num_nodes)):
        nodes[i].parent_id = -1
    for node in nodes:
        if node.parent_id > -1:
            node.set_parent(nodes[node.parent_id])
        else:
            roots.append(node)
    return roots


def join(strs, sep=""):
    if isinstance(strs[0], str):
        return sep.join(strs).strip()
    elif isinstance(strs[0], list):
        return sep.join(reduce(lambda x, y: x + y, strs, [])).strip()


def linearize(roots, bracket=True, shuffle=True, rel=True):
    tokens = []
    root_ids = list(range(len(roots)))
    if shuffle:
        random.shuffle(root_ids)
    for i in root_ids:
        tokens.extend(roots[i].linearize(bracket=bracket, shuffle=shuffle, rel=rel))
    return join(tokens)


def parse_ptb_tree_data(path):
    entries = []
    with open(path, "r") as fin:
        nodes = []
        for line in fin:
            frags = line.strip().split("\t")
            if len(frags) == 4:
                node = Node(word=norm_token(frags[0]), pos=norm_tag(frags[1]),
                            parent_id=int(frags[2]), rel=frags[3])
                nodes.append(node)
            else:
                entries.append(nodes)
                nodes = []
    return entries
