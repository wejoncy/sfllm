"""Compressed Full-KV token paths, optional checkpoints and one LRU."""

from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import islice

import numpy as np


def common_prefix(a, b, start=0):
    for i, (x, y) in enumerate(zip(a, islice(b, start, None))):
        if x != y:
            return i
    return min(len(a), len(b) - start)


@dataclass(eq=False)
class Node:
    tokens: tuple = ()
    parent: "Node | None" = None
    children: dict = field(default_factory=dict)
    full_slots: tuple = ()
    checkpoint: object = None
    descendants: int = 0


class PrefixCache:
    def __init__(self, full_pools, release_checkpoint=None):
        self.root = Node()
        self.full_pools = full_pools
        self.release_checkpoint = release_checkpoint
        self.lru = OrderedDict()
        self.protected = set()

    def match(self, tokens):
        node, matched, resume, checkpoint = self.root, 0, 0, None
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break
            n = common_prefix(child.tokens, tokens, matched)
            matched += n
            if self.release_checkpoint is None:
                resume, checkpoint = matched, child
            if n < len(child.tokens):
                break
            node = child
            if node.checkpoint is not None:
                resume, checkpoint = matched, node
        if self.release_checkpoint is None and checkpoint is not None:
            while checkpoint not in self.lru:
                checkpoint = next(iter(checkpoint.children.values()))
        return matched, resume, checkpoint

    def touch(self, node):
        self.lru.move_to_end(node)

    def full_prefix(self, node):
        path = []
        while node.parent is not None:
            path.append(node.full_slots)
            node = node.parent
        return tuple(
            np.concatenate([edge[i] for edge in reversed(path)]) if path else np.empty(0, dtype=np.int64)
            for i in range(len(self.full_pools))
        )

    def insert(self, tokens, full_slots, checkpoint=None):
        node, pos = self.root, 0
        tokens = tuple(tokens)
        while pos < len(tokens):
            child = node.children.get(tokens[pos])
            if child is None:
                child = Node(tokens[pos:], node,
                             full_slots=tuple(np.array(s[pos:], dtype=np.int64) for s in full_slots))
                for pool, slots in zip(self.full_pools, child.full_slots):
                    pool.retain(slots, cache=True, unique=True)
                node.children[tokens[pos]] = child
                node = child
                break
            n = common_prefix(child.tokens, tokens, pos)
            if n < len(child.tokens):
                middle = Node(child.tokens[:n], node,
                              full_slots=tuple(s[:n].copy() for s in child.full_slots),
                              descendants=child.descendants)
                node.children[child.tokens[0]] = middle
                child.tokens = child.tokens[n:]
                child.full_slots = tuple(s[n:].copy() for s in child.full_slots)
                child.parent = middle
                middle.children[child.tokens[0]] = child
                child = middle
            node = child
            pos += n
        if node in self.lru:
            if checkpoint is not None:
                self.release_checkpoint(checkpoint)
            self.touch(node)
            return node
        node.checkpoint = checkpoint
        self.lru[node] = None
        ancestor = node
        while ancestor is not None:
            ancestor.descendants += 1
            ancestor = ancestor.parent
        return node

    def evict(self, eligible=None):
        if not self.lru:
            return False
        node = next((n for n in self.lru if n not in self.protected
                     and (eligible is None or eligible(n))), None)
        if node is None:
            return False
        del self.lru[node]
        if node.checkpoint is not None:
            self.release_checkpoint(node.checkpoint)
        node.checkpoint = None
        while node is not None:
            node.descendants -= 1
            parent = node.parent
            if parent is not None and node.descendants == 0:
                for pool, slots in zip(self.full_pools, node.full_slots):
                    pool.release(slots, cache=True, unique=True)
                del parent.children[node.tokens[0]]
            elif parent is not None and node not in self.lru and len(node.children) == 1:
                child = next(iter(node.children.values()))
                child.tokens = node.tokens + child.tokens
                child.full_slots = tuple(
                    np.concatenate((a, b)) for a, b in zip(node.full_slots, child.full_slots)
                )
                child.parent = parent
                parent.children[node.tokens[0]] = child
            node = parent
        return True

    def clear(self):
        while self.evict():
            pass
