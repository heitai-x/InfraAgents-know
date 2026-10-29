"""Finite CPU lifecycle model; deterministic, offline, read-only, stdlib only.
No model, CUDA, network, filesystem, subprocess, or performance measurement.
"""
from itertools import permutations


class Engine:
    def __init__(self, bug=''):
        self.bug = bug
        self.instances = {}
        self.epoch = 0
        self.route = None
        self.bound = {}
        self.live = set()
        self.uses = {}
        self.bindings = set()
        self.outputs = set()
        self.used_tickets = set()

    def snapshot(self):
        instances = {g: (x['v'], x['phase'], frozenset(x['done']))
                     for g, x in self.instances.items()}
        return (instances, self.epoch, self.route, dict(self.bound),
                frozenset(self.live), dict(self.uses),
                frozenset(self.bindings), frozenset(self.outputs),
                frozenset(self.used_tickets))

    def act(self, event):
        op, *a = event
        if op == 'new':
            g, v = a
            if g in self.instances:
                return False
            self.instances[g] = {'v': v, 'phase': 'PREPARING', 'done': set()}
        elif op == 'rank':
            g, rank, v = a
            x = self.instances.get(g)
            if (x is None or x['phase'] != 'PREPARING' or x['v'] != v
                    or rank not in (0, 1) or rank in x['done']):
                return False
            x['done'].add(rank)
            if x['done'] == {0, 1} or self.bug == 'any_rank':
                x['phase'] = 'READY'
        elif op == 'publish':
            expected, g = a
            x = self.instances.get(g)
            if (x is None or g == self.route
                    or x['phase'] not in ('READY', 'DRAINING')
                    or (expected != self.epoch and self.bug != 'stale_publish')):
                return False
            old = self.route
            if old is not None:
                self.instances[old]['phase'] = 'DRAINING'
            self.route = g
            self.epoch += 1
            x['phase'] = 'OPEN'
            if self.bug == 'rebind':
                for rid in self.live:
                    self.bound[rid] = g
        elif op == 'admit':
            rid, epoch, g, shape = a
            x = self.instances.get(g)
            if (rid in self.bound or x is None or self.route != g
                    or (epoch != self.epoch and self.bug != 'stale_admit')
                    or x['phase'] != 'OPEN' or shape != 'short'
                    or sum(self.bound[r] == g for r in self.live) >= 2):
                return False
            self.bound[rid] = g
            self.live.add(rid)
        elif op == 'issue':
            rid, uid = a
            if rid not in self.live:
                return False
            g = self.bound[rid]
            key = (g, uid)
            if (self.instances[g]['phase'] not in ('OPEN', 'DRAINING')
                    or key in self.used_tickets):
                return False
            self.uses[key] = rid
            self.used_tickets.add(key)
        elif op == 'complete':
            g, uid = a
            key = (g, uid)
            if key not in self.uses:
                if self.bug != 'wrong_completion':
                    return False
                matches = [k for k in self.uses if k[1] == uid]
                if not matches:
                    return False
                key = matches[0]
            del self.uses[key]
        elif op == 'bind':
            g, uid = a
            x = self.instances.get(g)
            if (x is None or x['phase'] not in
                    ('PREPARING', 'READY', 'OPEN', 'DRAINING')
                    or (g, uid) in self.bindings):
                return False
            self.bindings.add((g, uid))
        elif op == 'unbind':
            key = tuple(a)
            if key not in self.bindings:
                return False
            self.bindings.remove(key)
        elif op == 'hold':
            rid, uid = a
            if rid not in self.live:
                return False
            g = self.bound[rid]
            if (self.instances[g]['phase'] not in ('OPEN', 'DRAINING')
                    or (g, uid) in self.outputs):
                return False
            self.outputs.add((g, uid))
        elif op == 'consume':
            key = tuple(a)
            if key not in self.outputs:
                return False
            self.outputs.remove(key)
        elif op == 'finish':
            rid, = a
            if rid not in self.live:
                return False
            self.live.remove(rid)
            if self.bug == 'cancel_clears_use':
                self.uses = {k: r for k, r in self.uses.items() if r != rid}
        elif op == 'retire':
            g, = a
            x = self.instances.get(g)
            if (x is None or self.route == g
                    or x['phase'] not in ('READY', 'DRAINING', 'FAILED')
                    or any(self.bound[r] == g for r in self.live)):
                return False
            x['phase'] = 'RETIRING'
        elif op == 'release':
            g, = a
            x = self.instances.get(g)
            if x is None or x['phase'] != 'RETIRING':
                return False
            if any(k[0] == g for k in self.uses) and self.bug != 'ignore_use':
                return False
            if any(k[0] == g for k in self.bindings) and self.bug != 'ignore_binding':
                return False
            if any(k[0] == g for k in self.outputs) and self.bug != 'ignore_output':
                return False
            x['phase'] = 'RETIRED'
        elif op == 'fail':
            g, = a
            x = self.instances.get(g)
            if x is None or x['phase'] in ('FAILED', 'RETIRING', 'RETIRED'):
                return False
            x['phase'] = 'FAILED'
            if self.route == g:
                self.route = None
                self.epoch += 1
        else:
            raise ValueError(op)
        return True


def observe(history):
    """Rebuild ownership from accepted facts, without reading Engine flags.

    Request completion never subtracts device or consumer facts. Immutable
    admission facts determine all later ownership. Release is checked against
    the complete outstanding history before being recorded as retirement.
    """
    versions, phases, ranks = {}, {}, {}
    epoch, route = 0, None
    admitted, ended = {}, set()
    issued, completed = {}, set()
    attached, detached = set(), set()
    held, consumed = set(), set()
    for event in history:
        op, *a = event
        if op == 'new':
            g, v = a
            assert g not in versions, 'instance identity reused'
            versions[g], phases[g], ranks[g] = v, 'PREPARING', set()
        elif op == 'rank':
            g, rank, v = a
            assert phases[g] == 'PREPARING', 'late rank accepted'
            assert v == versions[g] and rank in (0, 1), 'rank identity mismatch'
            assert rank not in ranks[g], 'duplicate rank accepted'
            ranks[g].add(rank)
            if ranks[g] == {0, 1}:
                phases[g] = 'READY'
        elif op == 'publish':
            expected, g = a
            assert expected == epoch and g != route, 'stale publication'
            assert ranks[g] == {0, 1}, 'incomplete execution group'
            assert phases[g] in ('READY', 'DRAINING'), 'invalid target'
            if route is not None:
                phases[route] = 'DRAINING'
            route, epoch, phases[g] = g, epoch + 1, 'OPEN'
        elif op == 'admit':
            rid, e, g, shape = a
            assert e == epoch and g == route, 'stale admission'
            assert phases[g] == 'OPEN' and ranks[g] == {0, 1}, 'unready admission'
            assert rid not in admitted and shape == 'short', 'identity or scope'
            active = [r for r in admitted if r not in ended and admitted[r] == g]
            assert len(active) < 2, 'logical slot budget exceeded'
            admitted[rid] = g
        elif op == 'issue':
            rid, uid = a
            assert rid in admitted and rid not in ended, 'issue after logical end'
            g = admitted[rid]
            assert phases[g] in ('OPEN', 'DRAINING'), 'issue after invalidation'
            key = (g, uid)
            assert key not in issued, 'operation identity reused'
            issued[key] = rid
        elif op == 'complete':
            key = tuple(a)
            assert key in issued and key not in completed, 'invalid completion'
            completed.add(key)
        elif op == 'bind':
            key = tuple(a)
            assert phases[key[0]] in ('PREPARING', 'READY', 'OPEN', 'DRAINING')
            assert key not in attached, 'binding identity reused in this fixture'
            attached.add(key)
        elif op == 'unbind':
            key = tuple(a)
            assert key in attached and key not in detached, 'invalid detach'
            detached.add(key)
        elif op == 'hold':
            rid, uid = a
            assert rid in admitted and rid not in ended, 'unprotected output'
            g = admitted[rid]
            assert phases[g] in ('OPEN', 'DRAINING')
            key = (g, uid)
            assert key not in held, 'output identity reused in this fixture'
            held.add(key)
        elif op == 'consume':
            key = tuple(a)
            assert key in held and key not in consumed, 'invalid consumer finish'
            consumed.add(key)
        elif op == 'finish':
            rid, = a
            assert rid in admitted and rid not in ended, 'invalid logical finish'
            ended.add(rid)
        elif op == 'retire':
            g, = a
            assert route != g and phases[g] in ('READY', 'DRAINING', 'FAILED')
            assert not any(r not in ended and owner == g for r, owner in admitted.items())
            phases[g] = 'RETIRING'
        elif op == 'release':
            g, = a
            assert phases[g] == 'RETIRING', 'release before retirement fence'
            assert not any(k[0] == g for k in issued.keys() - completed), 'live device use'
            assert not any(k[0] == g for k in attached - detached), 'live address binding'
            assert not any(k[0] == g for k in held - consumed), 'live output borrower'
            phases[g] = 'RETIRED'
        elif op == 'fail':
            g, = a
            assert phases[g] not in ('FAILED', 'RETIRING', 'RETIRED')
            phases[g] = 'FAILED'
            if route == g:
                route, epoch = None, epoch + 1
        else:
            raise AssertionError('unknown observed event')
    instances = {g: (versions[g], phases[g], frozenset(ranks[g])) for g in versions}
    active_uses = {k: owner for k, owner in issued.items() if k not in completed}
    live = frozenset(admitted.keys() - ended)
    for g in versions:
        if phases[g] == 'RETIRED':
            assert not any(admitted[r] == g for r in live), 'retired logical owner'
            assert not any(k[0] == g for k in active_uses), 'retired device owner'
            assert not any(k[0] == g for k in attached - detached), 'retired graph owner'
            assert not any(k[0] == g for k in held - consumed), 'retired output owner'
    return (instances, epoch, route, dict(admitted), live, active_uses,
            frozenset(attached - detached), frozenset(held - consumed),
            frozenset(issued))


class Harness:
    def __init__(self, bug=''):
        self.engine = Engine(bug)
        self.history = []
        self.steps = 0

    def step(self, event, expected=True):
        before = self.engine.snapshot()
        accepted = self.engine.act(event)
        self.steps += 1
        assert accepted == expected, ('wrong decision', event, accepted, expected)
        if accepted:
            self.history.append(event)
        else:
            assert self.engine.snapshot() == before, ('rejection mutated state', event)
        assert self.engine.snapshot() == observe(self.history), ('history mismatch', event)

    def ready(self, g, v):
        self.step(('new', g, v))
        self.step(('rank', g, 0, v))
        self.step(('rank', g, 1, v))


def protocol_suite(bug=''):
    h = Harness(bug)
    h.step(('new', 10, 'v1'))
    h.step(('bind', 10, 'graph'))
    h.step(('rank', 10, 0, 'v1'))
    h.step(('publish', 0, 10), False)
    h.step(('rank', 10, 1, 'v1'))
    h.step(('publish', 0, 10))
    h.step(('admit', 'r1', 1, 10, 'short'))
    h.step(('issue', 'r1', 'same_uid'))
    h.step(('hold', 'r1', 'response'))
    h.step(('new', 20, 'v2'))
    h.step(('rank', 20, 0, 'v2'))
    h.step(('publish', 1, 20), False)
    h.step(('admit', 'r2', 1, 20, 'short'), False)
    h.step(('rank', 20, 1, 'v2'))
    h.step(('publish', 1, 20))
    h.step(('admit', 'r2', 1, 10, 'short'), False)
    h.step(('admit', 'r2', 2, 20, 'short'))
    h.step(('issue', 'r2', 'same_uid'))
    h.step(('complete', 999, 'same_uid'), False)
    h.step(('publish', 1, 10), False)
    h.step(('finish', 'r1'))
    h.step(('issue', 'r1', 'late'), False)
    h.step(('retire', 10))
    h.step(('bind', 10, 'new_graph'), False)
    h.step(('publish', 2, 10), False)
    h.step(('release', 10), False)
    h.step(('complete', 20, 'same_uid'))
    h.step(('complete', 20, 'same_uid'), False)
    h.step(('release', 10), False)
    h.step(('complete', 10, 'same_uid'))
    h.step(('complete', 10, 'same_uid'), False)
    h.step(('release', 10), False)
    h.step(('unbind', 10, 'graph'))
    h.step(('release', 10), False)
    h.step(('consume', 10, 'response'))
    h.step(('release', 10))
    h.step(('rank', 10, 0, 'v1'), False)
    h.step(('fail', 20))
    h.step(('issue', 'r2', 'after_failure'), False)
    h.step(('finish', 'r2'))
    h.step(('retire', 20))
    h.step(('release', 20))
    return h.steps


def rollback_suite(bug=''):
    h = Harness(bug)
    h.ready(10, 'v1')
    h.step(('publish', 0, 10))
    h.step(('admit', 'a', 1, 10, 'short'))
    h.ready(20, 'v2')
    h.step(('publish', 1, 20))
    h.step(('admit', 'b', 2, 20, 'short'))
    h.step(('publish', 2, 10))
    h.step(('admit', 'stale', 1, 10, 'short'), False)
    h.step(('publish', 1, 20), False)
    h.step(('issue', 'a', 'queued_before_cut'))
    h.step(('issue', 'b', 'queued_on_v2'))
    h.step(('finish', 'a'))
    h.step(('finish', 'b'))
    h.step(('complete', 10, 'queued_before_cut'))
    h.step(('complete', 20, 'queued_on_v2'))
    h.step(('fail', 10))
    h.step(('retire', 10))
    h.step(('release', 10))
    h.step(('retire', 20))
    h.step(('release', 20))
    return h.steps


def failure_and_scope_suite():
    h = Harness()
    h.step(('new', 1, 'same_artifact'))
    h.step(('bind', 1, 'graph'))
    h.step(('fail', 1))
    h.step(('new', 2, 'same_artifact'))
    h.step(('rank', 1, 0, 'same_artifact'), False)
    h.step(('rank', 2, 0, 'wrong_artifact'), False)
    h.step(('rank', 2, 9, 'same_artifact'), False)
    h.step(('rank', 2, 0, 'same_artifact'))
    h.step(('rank', 2, 0, 'same_artifact'), False)
    h.step(('rank', 2, 1, 'same_artifact'))
    h.step(('admit', 'early', 0, 2, 'short'), False)
    h.step(('publish', 0, 2))
    h.step(('admit', 'long', 1, 2, 'long'), False)
    h.step(('admit', 'a', 1, 2, 'short'))
    h.step(('admit', 'b', 1, 2, 'short'))
    h.step(('admit', 'c', 1, 2, 'short'), False)
    h.step(('admit', 'a', 1, 2, 'short'), False)
    h.step(('finish', 'a'))
    h.step(('admit', 'c', 1, 2, 'short'))
    h.step(('retire', 1))
    h.step(('release', 1), False)
    h.step(('unbind', 1, 'graph'))
    h.step(('release', 1))
    h.step(('fail', 2))
    h.step(('rank', 2, 1, 'same_artifact'), False)
    h.step(('admit', 'd', 1, 2, 'short'), False)
    h.step(('retire', 2), False)
    h.step(('finish', 'b'))
    h.step(('finish', 'c'))
    h.step(('retire', 2))
    h.step(('release', 2))
    return h.steps


def ordering_suite(bug=''):
    total = 0
    endings = {
        'request': ('finish', 'r'),
        'kernel': ('complete', 10, 'k'),
        'graph': ('unbind', 10, 'g'),
        'output': ('consume', 10, 'o'),
    }
    for order in permutations(('request', 'kernel', 'graph', 'output')):
        h = Harness(bug)
        h.ready(10, 'v1')
        h.step(('bind', 10, 'g'))
        h.step(('publish', 0, 10))
        h.step(('admit', 'r', 1, 10, 'short'))
        h.step(('issue', 'r', 'k'))
        h.step(('hold', 'r', 'o'))
        h.ready(20, 'v2')
        h.step(('publish', 1, 20))
        remaining = set(endings)
        retiring = False
        freed = False
        h.step(('retire', 10), False)
        for name in order:
            h.step(endings[name])
            remaining.remove(name)
            if 'request' in remaining:
                h.step(('retire', 10), False)
            elif not retiring:
                h.step(('retire', 10))
                retiring = True
            if not freed:
                should_free = retiring and not remaining
                h.step(('release', 10), should_free)
                freed = should_free
        assert freed, 'finite completed schedule did not reclaim old instance'
        h.step(('fail', 20))
        h.step(('retire', 20))
        h.step(('release', 20))
        total += h.steps
    return total


def expect_detected(label, experiment):
    try:
        experiment()
    except AssertionError as error:
        print('NEGATIVE detected:', label, '|', str(error)[:180])
        return
    raise AssertionError('negative control escaped: ' + label)


def main():
    fixed_steps = protocol_suite() + rollback_suite() + failure_and_scope_suite()
    ordered_steps = ordering_suite()
    print('POSITIVE accepted-history and rejection checks:', fixed_steps, 'events')
    print('POSITIVE responsibility completion orders: 24;', ordered_steps, 'events')
    controls = (
        ('any_rank', protocol_suite),
        ('rebind', protocol_suite),
        ('cancel_clears_use', protocol_suite),
        ('wrong_completion', protocol_suite),
        ('stale_admit', rollback_suite),
        ('stale_publish', rollback_suite),
        ('ignore_use', ordering_suite),
        ('ignore_binding', ordering_suite),
        ('ignore_output', ordering_suite),
    )
    for bug, suite in controls:
        expect_detected(bug, lambda b=bug, s=suite: s(b))
    print('SCOPE: finite CPU protocol; 2 ranks, 2 request slots per instance, short path only.')
    print('No GPU execution, real byte capacity, distributed persistence, migration, or performance claim.')


if __name__ == '__main__':
    main()
