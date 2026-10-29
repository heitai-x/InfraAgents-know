"""CPU prefix identity, publication, residency, eviction and cost examples.

The causal scalar recurrence has an independent closed-form reference. This is
not a Transformer, GPU allocator, concurrency proof, or benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def check(name, condition, **details):
    if not condition:
        raise AssertionError((name, details))
    CHECKS.append({"name": name, "result": "PASS", **details})


def rejects(name, action, error=AssertionError):
    try:
        action()
    except error:
        check(name, True)
    else:
        raise AssertionError("Expected rejection: " + name)


def initial(eta):
    return sum(ord(c) for c in eta)


def reference(eta, tokens):
    """Independent full-prefix expression, not a page-table walk/recurrence."""
    return tuple(initial(eta) * 3 ** (t + 1)
                 + sum(value * 3 ** (t - i) for i, value in enumerate(tokens[:t + 1]))
                 for t in range(len(tokens)))


@dataclass
class Page:
    generation: int = 0
    data: tuple | None = None
    cache: bool = False
    owners: set = field(default_factory=set)
    readers: set = field(default_factory=set)
    writer: str | None = None

    def free(self):
        return not (self.cache or self.owners or self.readers or self.writer)


@dataclass(frozen=True)
class Node:
    eta: str
    tokens: tuple
    parent: tuple | None
    handle: tuple
    bucket: bytes

    @property
    def key(self):
        return self.eta, self.tokens


@dataclass
class Request:
    name: str
    eta: str
    tokens: tuple
    p: int
    q: int
    hits: tuple
    new: tuple
    active: bool = True
    ready: bool = False
    committed: bool = False
    aborted: bool = False

    @property
    def handles(self):
        return self.hits + self.new


class Cache:
    """Atomic host transitions, full-page cache, explicit execution readers."""
    def __init__(self, page_tokens=4, capacity=5, collide=False):
        self.p = page_tokens
        self.pages = [Page() for _ in range(capacity)]
        self.nodes = {}
        self.buckets = {}
        self.requests = {}
        self.readers = {}
        self.counter = 0
        self.collide = collide

    def bucket(self, eta, tokens):
        if self.collide:
            return b"forced-same-bucket"
        h = hashlib.sha256(json.dumps(["root", eta]).encode()).digest()
        for offset in range(0, len(tokens), self.p):
            h = hashlib.sha256(h + json.dumps(["block", offset, tokens[offset:offset + self.p]]).encode()).digest()
        return h

    def page(self, handle):
        pid, generation = handle
        result = self.pages[pid]
        assert result.generation == generation, "stale physical generation"
        return result

    def collect(self):
        for page in self.pages:
            if page.free():
                page.data = None

    def find(self, eta, tokens):
        # Exact identity includes the full dependency prefix despite collisions.
        for node in self.buckets.get(self.bucket(eta, tokens), []):
            if node.eta == eta and node.tokens == tokens:
                page = self.page(node.handle)
                assert page.cache and page.writer is None and len(page.data) == self.p
                return node
        return None

    def lookup(self, eta, tokens, generation=True):
        tokens = tuple(tokens)
        if generation and not tokens:
            raise ValueError("Generation needs a known initial input")
        limit = len(tokens) - 1 if generation else len(tokens)
        handles = []
        for end in range(self.p, limit + 1, self.p):
            node = self.find(eta, tokens[:end])
            if node is None:
                break
            handles.append(node.handle)
        return len(handles) * self.p, tuple(handles)

    def victims(self, wanted):
        removed = set()
        free = sum(p.free() for p in self.pages)
        while free + len(removed) < wanted:
            choice = next((node for key, node in self.nodes.items() if key not in removed
                           and not self.page(node.handle).owners
                           and not self.page(node.handle).readers
                           and self.page(node.handle).writer is None
                           and not any(other.parent == key and other.key not in removed
                                       for other in self.nodes.values())), None)
            if choice is None:
                raise MemoryError("No complete legal reservation")
            removed.add(choice.key)
        return [self.nodes[key] for key in self.nodes if key in removed]

    def evict(self, node):
        page = self.page(node.handle)
        assert not page.owners and not page.readers and page.writer is None
        assert not any(x.parent == node.key for x in self.nodes.values()), "non-leaf eviction"
        del self.nodes[node.key]
        self.buckets[node.bucket].remove(node)
        page.cache = False
        self.collect()

    def admit(self, name, eta, tokens, quota=None, generation=True):
        assert name not in self.requests or not self.requests[name].active
        self.counter += 1
        identity = name + "@" + str(self.counter)
        tokens = tuple(tokens)
        p, hits = self.lookup(eta, tokens, generation)
        q = len(tokens) - p if quota is None else min(quota, len(tokens) - p)
        assert q >= 0
        for handle in hits:
            self.page(handle).owners.add(identity)
        needed = math.ceil(q / self.p)
        try:
            victims = self.victims(needed)
        except MemoryError:
            for handle in hits:
                self.page(handle).owners.remove(identity)
            raise
        # Planner order may contain parent after child; apply in leaf order.
        while victims:
            node = next(x for x in victims if not any(y.parent == x.key for y in self.nodes.values()))
            self.evict(node)
            victims.remove(node)
        new = []
        for i, page in enumerate(self.pages):
            if len(new) == needed:
                break
            if page.free():
                page.generation += 1
                page.writer = identity
                new.append((i, page.generation))
        assert len(new) == needed
        request = Request(identity, eta, tokens, p, q, hits, tuple(new))
        self.requests[name] = request
        for handle in hits:
            self.page(handle).readers.add("compute:" + identity)
        return request

    def complete(self, request):
        assert not request.ready and not request.aborted
        state = self.page(request.hits[-1]).data[-1] if request.hits else initial(request.eta)
        result = []
        for token in request.tokens[request.p:request.p + request.q]:
            state = 3 * state + token
            result.append(state)
        for j, handle in enumerate(request.new):
            page = self.page(handle)
            assert page.writer == request.name
            page.data = tuple(result[j * self.p:(j + 1) * self.p])
        for handle in request.hits:
            self.page(handle).readers.remove("compute:" + request.name)
        request.ready = True
        self.collect()

    def publish(self, request):
        assert request.active and request.ready and not request.committed
        for j, handle in enumerate(request.new):
            page = self.page(handle)
            assert page.writer == request.name
            page.owners.add(request.name)
            page.writer = None
            if len(page.data) == self.p:
                end = request.p + (j + 1) * self.p
                prefix = request.tokens[:end]
                if self.find(request.eta, prefix) is None:
                    parent = (request.eta, prefix[:-self.p]) if end > self.p else None
                    assert parent is None or parent in self.nodes
                    bucket = self.bucket(request.eta, prefix)
                    node = Node(request.eta, prefix, parent, handle, bucket)
                    self.nodes[node.key] = node
                    self.buckets.setdefault(bucket, []).append(node)
                    page.cache = True
        request.committed = True

    def materialize(self, request):
        result = []
        for handle in request.handles:
            result.extend(self.page(handle).data)
        return tuple(result[:request.p + request.q])

    def cancel(self, request):
        assert request.active
        request.active = False
        for handle in request.handles:
            self.page(handle).owners.discard(request.name)
        self.collect()

    def abort(self, request):
        assert not request.active and request.ready and not request.committed
        for handle in request.new:
            page = self.page(handle)
            assert page.writer == request.name
            page.writer = None
        request.aborted = True
        self.collect()

    def read(self, request):
        assert request.active and request.committed
        self.counter += 1
        ticket = "reader:" + str(self.counter)
        self.readers[ticket] = (request, request.handles)
        for handle in request.handles:
            self.page(handle).readers.add(ticket)
        return ticket

    def finish(self, ticket):
        if ticket not in self.readers:
            return False
        request, handles = self.readers.pop(ticket)
        assert self.materialize(request) == reference(request.eta, request.tokens[:request.p + request.q])
        for handle in handles:
            self.page(handle).readers.remove(ticket)
        self.collect()
        return True

    def invariant(self):
        for key, node in self.nodes.items():
            assert key == node.key
            assert node.parent is None or node.parent in self.nodes
            page = self.page(node.handle)
            assert page.cache and len(page.data) == self.p
            assert page.data == reference(node.eta, node.tokens)[-self.p:]
        for request in self.requests.values():
            if request.active and request.committed:
                assert self.materialize(request) == reference(request.eta, request.tokens[:request.p + request.q])
        for page in self.pages:
            assert not page.free() or page.data is None


def tokens(text):
    return tuple(ord(c) for c in text)


def calculated(cache, name, eta, sequence):
    req = cache.admit(name, eta, sequence)
    cache.complete(req)
    cache.publish(req)
    return req


def check_identity_and_resume():
    cache = Cache(capacity=16, collide=True)
    a = calculated(cache, "A", "model:adapter-A", tokens("abcdefghij"))
    cache.cancel(a)
    check("full pages survive request completion through cache ownership",
          len(cache.nodes) == 2 and sum(not p.free() for p in cache.pages) == 2)
    check("full hit without logits keeps a known input suffix", cache.lookup(a.eta, tokens("abcdefgh"))[0] == 4)
    check("n=9 can reuse eight complete states", cache.lookup(a.eta, tokens("abcdefghx"))[0] == 8)
    check("KV-only restore may use all completed states", cache.lookup(a.eta, tokens("abcdefgh"), False)[0] == 8)
    check("n=1 reuses no KV before first output", cache.lookup(a.eta, tokens("a"))[0] == 0)
    rejects("empty generation input requires declared start state", lambda: cache.lookup(a.eta, ()), ValueError)
    check("forced bucket collision preserves namespace isolation", cache.lookup("model:adapter-B", a.tokens)[0] == 0)
    check("changed early prefix invalidates equal later token block",
          cache.lookup(a.eta, tokens("WXYZefghij"))[0] == 0)
    b = calculated(cache, "B", "model:adapter-B", a.tokens)
    check("second namespace with identical tokens computes its own states",
          cache.materialize(b) == reference(b.eta, b.tokens) and cache.materialize(b) != reference(a.eta, a.tokens))
    c = calculated(cache, "C", a.eta, tokens("abcdefghx"))
    check("resumed prefix produces independent full causal reference", c.p == 8 and cache.materialize(c) == reference(c.eta, c.tokens))
    duplicate = calculated(cache, "D", a.eta, tokens("abcdefgh"))
    check("duplicate completion need not remap an active private copy",
          duplicate.p == 4 and not cache.page(duplicate.new[-1]).cache and cache.materialize(duplicate) == reference(a.eta, duplicate.tokens))
    cache.invariant()
    # Deliberately broken namespace identity gives the wrong state, not rounding.
    wrong = reference(a.eta, a.tokens[:8])
    check("omitted namespace has a detectable causal counterexample", wrong != reference(b.eta, b.tokens[:8]))
    cache = Cache(capacity=8)
    pending = cache.admit("P", "m", tokens("abcdefgh"))
    check("token IDs alone do not publish computed state", cache.lookup("m", tokens("abcdefghx"))[0] == 0)
    rejects("uncomputed pages cannot be published", lambda: cache.publish(pending))
    cache.cancel(pending)
    check("cancelled unfinished writer still holds its pages", all(cache.page(h).writer == pending.name for h in pending.new))
    cache.complete(pending)
    check("completed unpublished writer still holds its pages", all(cache.page(h).writer == pending.name for h in pending.new))
    rejects("cancelled completed work cannot publish", lambda: cache.publish(pending))
    cache.abort(pending)
    check("explicit abort releases exactly the uncommitted writer retention", all(p.free() for p in cache.pages))
    old = cache.admit("same-slot", "m", tokens("abcdefgh"))
    cache.cancel(old)
    replacement = calculated(cache, "same-slot", "m", tokens("QRSTU"))
    cache.complete(old)
    rejects("old request lifetime cannot publish over a replacement", lambda: cache.publish(old))
    cache.abort(old)
    check("old abort cannot release a replacement's mappings",
          cache.materialize(replacement) == reference("m", replacement.tokens)
          and all(replacement.name in cache.page(h).owners for h in replacement.handles))


def check_trajectory():
    cache = Cache(capacity=5)
    a = calculated(cache, "A", "m", tokens("abcdefghij"))
    cache.cancel(a)
    b = calculated(cache, "B", "m", tokens("abcdefghx"))
    reader = cache.read(b)
    old = b.hits[-1]
    cache.cancel(b)
    c = calculated(cache, "C", "m", tokens("abcdwxyzz"))
    check("five-page trajectory protects all cache/request/reader objects", sum(not p.free() for p in cache.pages) == 5)
    before = [(p.generation, p.cache, set(p.owners), set(p.readers), p.writer, p.data) for p in cache.pages]
    rejects("cold cached leaf with an in-flight reader is not free capacity",
            lambda: cache.admit("D", "m", tokens("QRSTU")), MemoryError)
    check("failed reservation preserves cache and existing state",
          before == [(p.generation, p.cache, set(p.owners), set(p.readers), p.writer, p.data) for p in cache.pages])
    cache.finish(reader)
    d = calculated(cache, "D", "m", tokens("QRSTU"))
    check("leaf eviction frees exactly enough after the last reader ends", {h[0] for h in d.new} == {1, 2})
    check("retained common ancestor preserves the other branch", cache.lookup("m", c.tokens)[0] == 8)
    check("evicted prefix becomes a shorter real hit", cache.lookup("m", b.tokens)[0] == 4)
    rejects("recycled physical page rejects its old generation", lambda: cache.page(old))
    current = [(p.generation, set(p.owners), set(p.readers)) for p in cache.pages]
    check("duplicate reader completion cannot change a new page lifetime",
          cache.finish(reader) is False and current == [(p.generation, set(p.owners), set(p.readers)) for p in cache.pages])
    cache.cancel(c)
    cache.cancel(d)
    b2 = calculated(cache, "B2", "m", b.tokens)
    check("recomputed evicted suffix matches independent causal reference", b2.p == 4 and cache.materialize(b2) == reference("m", b.tokens))
    cache.invariant()


def check_differential_and_costs():
    rng = random.Random(2803)
    cases = 0
    for p in [1, 3, 4, 7]:
        for collide in [False, True]:
            cache = Cache(page_tokens=p, capacity=40, collide=collide)
            for i in range(40):
                n = rng.randrange(1, 20)
                seq = tuple(rng.randrange(1, 6) for _ in range(n))
                eta = "m" + str(i % 2)
                req = calculated(cache, str(i), eta, seq)
                assert cache.materialize(req) == reference(eta, seq)
                cache.invariant()
                cache.cancel(req)
                cases += 1
    check("independent causal reference across page sizes, namespaces and collisions", True, cases=cases)
    shared = set(range(0, 13)) & {4, 8, 12} & {8, 12} & set(range(0, 9))
    check("joint restore uses a set intersection", max(shared) == 8)
    matched_boundaries = {0, 4, 8, 12}
    check("a non-restorable intermediate boundary does not end prefix matching",
          4 not in {8, 12} and max(matched_boundaries & {8, 12} & set(range(0, 10))) == 8)
    check("taking only independent maxima can select a nonexistent restore point",
          min(max(s) for s in [{4, 12}, {8}, {8, 12}]) == 8 and not ({4, 12} & {8} & {8, 12}))
    transfer_ms = 32 * 1024**2 / (8 * 1024**3) * 1000
    remote = .6 + .4 + transfer_ms + 7
    check("local and remote restoration arithmetic", transfer_ms == 3.90625 and remote == 11.90625)
    check("queue delay reverses remote reuse versus recompute", remote < 20 < remote + 12)
    check("reuse probability reverses the retention choice",
          math.isclose(.5 * 12.4 - .2 - 2, 4) and math.isclose(.05 * 12.4 - .2 - 2, -1.58))
    tau = 128 * 1024
    check("long shared-prefix capacity example",
          100 * (8192 + 256) * tau / 1024**3 == 103.125 and
          (8192 + 100 * 256) * tau / 1024**3 == 4.125)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check_identity_and_resume()
    check_trajectory()
    cases = check_differential_and_costs()
    names = ["knowledge/algorithms/prefix-cache-identity.md",
             "knowledge/sources/S-SGLANG.md", "knowledge/sources/S-VLLM-PREFIX.md",
             "knowledge/sources/S-HF-CACHE.md", "knowledge/sources/S-MOONCAKE.md"]
    report = {"result": "PASS", "checked_at": "2026-10-28", "count": len(CHECKS),
              "checks": CHECKS, "differential_cases": cases,
              "documents": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "CPU exact identity/full-page directory with serialized host transitions, independent causal scalar reference, declared five-page lifecycle and synthetic cost arithmetic.",
              "limits": ["Not a Transformer implementation, GPU kernel, graph, real concurrent allocator, network transport, performance or model-quality experiment.",
                         "The reference is an integer causal recurrence; it checks recovery identity, not floating-point equivalence across real model dataflows.",
                         "Finite explicit transitions do not exhaust every interleaving or establish target-device visibility.",
                         "Cost inputs are synthetic assumptions; there is no controlled knowledge-benefit claim."]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PASS", "checks": len(CHECKS), "differential_cases": cases}))


if __name__ == "__main__":
    main()
