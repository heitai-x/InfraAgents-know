from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import permutations
import json


LAYERS = (0, 1)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@dataclass
class State:
    r: tuple
    branch: int
    x: tuple
    param: int
    k: int = 0
    g: int = 0
    v: int = 0
    rng: int = 0
    grammar: tuple = ()
    o: int = 0
    events: tuple = ()
    live: bool = True
    active: object = None
    versions: tuple = ('model-1', 'adapter-0', 'processor-1')
    kv: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Row:
    r: tuple
    branch: int
    g: int
    p: int
    token: str
    param: int
    versions: tuple
    sample: bool
    ordinal: object


def atom_key(row):
    return (row.r, row.branch, row.g, row.p)


@dataclass(frozen=True)
class Segment:
    key: tuple
    g: int
    p0: int
    p1: int
    n0: int
    o0: int
    v0: int
    rng0: int
    grammar0: tuple
    x0: tuple
    kind: str


@dataclass
class Page:
    epoch: int
    tag: tuple
    execution: set = field(default_factory=set)
    owners: set = field(default_factory=set)


class PagePool:
    # A cell is one symbolic page for one layer and position, not a real layout.
    def __init__(self):
        self.pages = {}

    def entry(self, handle):
        number, epoch = handle
        page = self.pages[number]
        require(page.epoch == epoch, 'stale page handle')
        return page

    def allocate(self, tag, view_id):
        free = [number for number, page in self.pages.items()
                if not page.execution and not page.owners]
        if free:
            number = min(free)
            epoch = self.pages[number].epoch + 1
        else:
            number = len(self.pages)
            epoch = 0
        self.pages[number] = Page(epoch, tag, {view_id}, set())
        return (number, epoch)

    def reusable(self, handle):
        page = self.entry(handle)
        return not page.execution and not page.owners

    def add_execution(self, handle, view_id):
        self.entry(handle).execution.add(view_id)

    def drop_execution(self, handle, view_id):
        page = self.entry(handle)
        require(view_id in page.execution, 'execution reference released twice')
        page.execution.remove(view_id)


def audit_pages(state, pool):
    # Independent invariant: required cells come from the committed boundary,
    # fixed layer domain, token history and version, not from an execution view.
    key = (state.r, state.branch)
    expected_cells = {(layer, p) for layer in LAYERS for p in range(state.k)}
    require(set(state.kv) == expected_cells, 'committed KV coverage mismatch')
    for layer, p in sorted(expected_cells):
        handle = state.kv[(layer, p)]
        page = pool.entry(handle)
        expected_tag = (key, layer, p, state.versions, state.x[p])
        require(page.tag == expected_tag, 'KV page identity/content mismatch')
        require(key in page.owners, 'committed KV lacks long-lived owner')
        require(not pool.reusable(handle), 'committed history is reusable')


@dataclass
class Record:
    view_id: str
    epoch: int
    rows: list
    segments: list
    planned: dict
    holders: set
    columns: list = field(default_factory=list)
    inverse: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)
    started: list = field(default_factory=list)
    outcomes: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    new_pages: dict = field(default_factory=dict)
    page_refs: set = field(default_factory=set)
    steps: list = field(default_factory=list)
    phase: str = 'PREPARING'
    business_done: bool = False
    claimed: bool = False
    released: bool = False
    release_count: int = 0


def refresh(view):
    view.columns = [(a.token, a.param, a.versions, a.ordinal) for a in view.rows]
    view.inverse = {atom_key(a): i for i, a in enumerate(view.rows)}
    view.targets = [(i, atom_key(a), a.ordinal, a.param)
                    for i, a in enumerate(view.rows) if a.sample]


def verify_view(authority, selections, view):
    expected = {}
    for key, q in selections:
        s = authority[key]
        end = s.k + q
        require(type(q) is int and q > 0 and end <= len(s.x), 'bad reference selection')
        for p in range(s.k, end):
            full = (s.r, s.branch, s.g, p)
            require(full not in expected, 'duplicate reference selection')
            sample = end == len(s.x) and p == end - 1
            expected[full] = Row(s.r, s.branch, s.g, p, s.x[p], s.param,
                                 s.versions, sample, s.o if sample else None)
    errors = []
    if Counter(atom_key(a) for a in view.rows) != Counter({k: 1 for k in expected}):
        errors.append('identity coverage/duplicate mismatch')
    if len(view.inverse) != len(expected):
        errors.append('inverse cardinality mismatch')
    if len(view.columns) != len(view.rows):
        errors.append('column cardinality mismatch')
    wanted_targets = []
    for i, row in enumerate(view.rows):
        full = atom_key(row)
        truth = expected.get(full)
        if truth is None:
            errors.append('unknown complete identity')
            continue
        if row != truth:
            errors.append('authoritative row field mismatch')
        wanted = (truth.token, truth.param, truth.versions, truth.ordinal)
        if i >= len(view.columns) or view.columns[i] != wanted:
            errors.append('authoritative derived-column mismatch')
        if view.inverse.get(full) != i:
            errors.append('inverse row mismatch')
        if truth.sample:
            wanted_targets.append((i, full, truth.ordinal, truth.param))
    if view.targets != wanted_targets:
        errors.append('authoritative sampling-target mismatch')
    return errors


class Runtime:
    # Serial coordinator, one execution slot, two symbolic Transformer layers.
    # No device computation, real KV payload, network or persistent transaction.
    def __init__(self, states):
        self.states = {(s.r, s.branch): s for s in states}
        require(len(self.states) == len(states), 'duplicate branch state')
        self.epoch = 0
        self.owner = None
        self.views = {}
        self.pool = PagePool()
        self.kv_commits = set()
        self.sample_commits = set()
        self.output_commits = set()

    def build(self, view_id, selections, bad_cid=False, omit_active=False):
        if not selections:
            return None
        require(view_id not in self.views, 'view_id reused')
        require(self.owner is None, 'slot still protected')
        rows, segments, planned = [], [], {}
        seen = set()
        # Entire private draft is checked before allocating or binding anything.
        for key, q in selections:
            s = self.states[key]
            if key in seen:
                raise ValueError('branch selected twice')
            seen.add(key)
            if (type(q) is not int or q <= 0 or s.k < 0 or s.k + q > len(s.x)
                    or not s.live or s.active is not None):
                raise ValueError('illegal work')
            audit_pages(s, self.pool)
            p0, p1 = s.k, s.k + q
            kind = 'SAMPLE' if p1 == len(s.x) else 'KV_ONLY'
            segments.append(Segment(key, s.g, p0, p1, len(s.x), s.o,
                                    s.v, s.rng, s.grammar, s.x, kind))
            for p in range(p0, p1):
                sample = kind == 'SAMPLE' and p == p1 - 1
                rows.append(Row(s.r, s.branch, s.g, p, s.x[p], s.param,
                                s.versions, sample, s.o if sample else None))
            if kind == 'SAMPLE':
                serial = len(planned) + 1
                cid = ((view_id, s.o) if bad_cid else
                       (view_id, s.r, s.branch, s.g, s.o, 'transport', serial))
                planned[key] = cid
        rec = Record(view_id, self.epoch, rows, segments, planned, set())
        refresh(rec)
        require(len(rec.inverse) == len(rows), 'duplicate complete placement key')
        rec.steps.append('draft_checked')
        # No external event can interleave this model's publication transaction.
        for seg in segments:
            s = self.states[seg.key]
            require(s.active is None and s.v == seg.v0 and s.g == seg.g,
                    'publication snapshot conflict')
        for seg in segments:
            s = self.states[seg.key]
            rec.history[seg.key] = dict(s.kv)
            rec.new_pages[seg.key] = {}
            for handle in s.kv.values():
                self.pool.add_execution(handle, view_id)
                rec.page_refs.add(handle)
            for layer in LAYERS:
                for p in range(seg.p0, seg.p1):
                    tag = (seg.key, layer, p, s.versions, s.x[p])
                    handle = self.pool.allocate(tag, view_id)
                    rec.new_pages[seg.key][(layer, p)] = handle
                    rec.page_refs.add(handle)
        self.owner = (view_id, self.epoch)
        self.views[view_id] = rec
        if not omit_active:
            for seg in segments:
                self.states[seg.key].active = view_id
        rec.steps.append('published_E_owner_bindings_and_references')
        rec.holders.add(('prep', view_id))
        rec.steps.append('registered_preparation_before_start')
        return rec

    def abort_unsubmitted(self, rec):
        require(rec.phase in ('PREPARING', 'READY', 'ABORTED'),
                'cannot assume an accepted/unknown submission was not accepted')
        if not rec.business_done:
            for seg in rec.segments:
                rec.outcomes[seg.key] = 'aborted'
            rec.business_done = True
            rec.phase = 'ABORTED'
            rec.steps.append('unsubmitted_business_resolved')
        self.finalize(rec)

    def prepare_done(self, rec, success=True):
        uid = ('prep', rec.view_id)
        if uid not in rec.holders:
            return 'duplicate_noop'
        rec.holders.remove(uid)
        rec.steps.append('preparation_access_ended')
        if rec.business_done:
            self.finalize(rec)
            return 'aborted'
        if not success:
            for seg in rec.segments:
                s = self.states[seg.key]
                if s.active == rec.view_id and s.g == seg.g:
                    s.live = False
        if not success or any(not self.states[s.key].live for s in rec.segments):
            self.abort_unsubmitted(rec)
            return 'aborted'
        rec.phase = 'READY'
        return 'ready'

    def submit(self, rec, outcome='accepted'):
        require(rec.phase == 'READY' and not rec.business_done, 'not ready')
        require(outcome in ('accepted', 'rejected', 'unknown'), 'bad submit outcome')
        valid = self.owner == (rec.view_id, rec.epoch)
        for seg in rec.segments:
            s = self.states[seg.key]
            valid = valid and (s.live and s.active == rec.view_id and s.g == seg.g
                               and s.v == seg.v0 and s.k == seg.p0
                               and s.x == seg.x0)
        if not valid:
            self.abort_unsubmitted(rec)
            return 'aborted'
        rec.holders.add(('device', rec.view_id))
        rec.steps.append('registered_device_before_submit')
        if outcome == 'rejected':
            # This event includes proof that no device acquired access.
            rec.holders.remove(('device', rec.view_id))
            self.abort_unsubmitted(rec)
            return 'rejected'
        rec.phase = 'SUBMITTED' if outcome == 'accepted' else 'UNKNOWN'
        rec.steps.append(rec.phase)
        return rec.phase.lower()

    def certificate(self, rec, drained=True, layers=True, success=True):
        writes = [(layer, atom_key(row)) for layer in LAYERS for row in rec.rows]
        if not layers:
            writes = writes[:-1]
        return {'view_id': rec.view_id, 'slot': 'slot-0', 'epoch': rec.epoch,
                'drained': drained, 'success': success, 'visible': True,
                'keys': tuple(atom_key(row) for row in rec.rows),
                'writes': tuple(writes),
                'logits': tuple(atom_key(row) for row in rec.rows if row.sample)}

    def finalize(self, rec):
        if rec.holders or not rec.business_done or rec.released:
            return
        require(self.owner == (rec.view_id, rec.epoch), 'release owner mismatch')
        for handle in rec.page_refs:
            self.pool.drop_execution(handle, rec.view_id)
        rec.released = True
        rec.release_count += 1
        self.owner = None
        self.epoch += 1
        for seg in rec.segments:
            s = self.states[seg.key]
            if s.active == rec.view_id:
                s.active = None
        rec.steps.append('released_execution_references_and_bindings')

    def retain_kv(self, s, seg, rec, omit_owner):
        # Called within the same serial state transaction as k and J publication.
        for cell, handle in rec.new_pages[seg.key].items():
            page = self.pool.entry(handle)
            require(rec.view_id in page.execution, 'new page lost protection before commit')
            if not omit_owner:
                page.owners.add(seg.key)
            s.kv[cell] = handle
        s.k = seg.p1

    def complete(self, rec, cert, proposals=None, faulty_kv_emit=False, omit_owner=False):
        proposals = {} if proposals is None else proposals
        if (cert['view_id'] != rec.view_id or cert['slot'] != 'slot-0'
                or cert['epoch'] != rec.epoch):
            return 'identity_mismatch'
        if rec.claimed:
            return 'duplicate_noop'
        if rec.business_done:
            return 'not_submitted_noop'
        require(rec.phase in ('SUBMITTED', 'UNKNOWN'), 'completion before submission')
        if not cert['drained']:
            return 'drain_wait'
        rec.claimed = True
        required_writes = Counter((layer, atom_key(row))
                                  for layer in LAYERS for row in rec.rows)
        required_logits = Counter(atom_key(row) for row in rec.rows if row.sample)
        physical_ok = (cert['success'] and cert['visible']
                       and Counter(cert['keys']) == Counter(atom_key(row) for row in rec.rows)
                       and Counter(cert['writes']) == required_writes
                       and Counter(cert['logits']) == required_logits)
        if not physical_ok:
            for seg in rec.segments:
                s = self.states[seg.key]
                if s.active == rec.view_id and s.g == seg.g:
                    s.live = False
                rec.outcomes[seg.key] = 'failed'
        else:
            for seg in rec.segments:
                s = self.states[seg.key]
                valid = (s.live and s.active == rec.view_id and s.g == seg.g
                         and self.owner == (rec.view_id, rec.epoch)
                         and s.v == seg.v0 and s.k == seg.p0
                         and len(s.x) == seg.n0 and s.x == seg.x0
                         and s.o == seg.o0 and s.rng == seg.rng0
                         and s.grammar == seg.grammar0)
                if not valid:
                    rec.outcomes[seg.key] = 'discarded'
                    continue
                r, branch = seg.key
                j = (rec.view_id, r, branch, seg.g, seg.p0, seg.p1, 'KV')
                require(j not in self.kv_commits, 'KV range replay')
                emit = seg.kind == 'SAMPLE' or faulty_kv_emit
                token = None
                if emit:
                    if seg.kind == 'SAMPLE':
                        require(seg.key in proposals, 'missing synthetic sampling outcome')
                    token = proposals.get(seg.key, 'BUG-intermediate-output')
                if emit and token == '<FAIL>':
                    self.retain_kv(s, seg, rec, omit_owner)
                    s.v += 1
                    s.live = False
                    self.kv_commits.add(j)
                    rec.outcomes[seg.key] = 'failed_sampling_KV_retained'
                    continue
                if emit:
                    q = (r, branch, seg.g, seg.o0)
                    u = (r, branch, seg.o0)
                    require(q not in self.sample_commits, 'sample commit replay')
                    require(u not in self.output_commits, 'logical output replay')
                # Each retaining path establishes the same long-lived ownership.
                self.retain_kv(s, seg, rec, omit_owner)
                if emit:
                    stop = token == '<EOS>'
                    if not stop:
                        s.x = s.x + (token,)
                    else:
                        s.live = False
                    s.rng += 1
                    s.grammar = s.grammar + (token,)
                    s.events = s.events + ((seg.o0, 'STOP' if stop else 'TOKEN', token),)
                    s.o += 1
                    self.sample_commits.add(q)
                    self.output_commits.add(u)
                s.v += 1
                self.kv_commits.add(j)
                rec.outcomes[seg.key] = 'resolved'
                if emit and seg.kind == 'SAMPLE':
                    cid = rec.planned[seg.key]
                    rec.holders.add(cid)
                    rec.started.append((seg.key, seg.o0, 'transport', cid))
        rec.business_done = True
        rec.phase = 'DONE'
        rec.holders.remove(('device', rec.view_id))
        self.finalize(rec)
        return 'resolved'

    def consumer_done(self, rec, cid):
        if cid not in rec.holders:
            return False
        rec.holders.remove(cid)
        self.finalize(rec)
        return True


def launch(rt, view_id, selections, **kwargs):
    rec = rt.build(view_id, selections, **kwargs)
    if rec is None:
        return None
    require(rt.prepare_done(rec) == 'ready', 'normal preparation failed')
    require(rt.submit(rec) == 'submitted', 'normal submission failed')
    return rec


def state_tuple(s):
    return (s.k, len(s.x), s.events, s.rng, s.grammar, s.o)


def expect_rejected(check, label, observations):
    try:
        check()
    except AssertionError:
        observations.append(label)
        return
    raise AssertionError('negative control was accepted: ' + label)


def finish_consumers(rt, rec):
    for cid in tuple(rec.holders):
        rt.consumer_done(rec, cid)


def mixed_requests(observations):
    a = State(('A', 1), 1, ('X',), 11)
    b = State(('B', 1), 1, ('X',), 22)
    rt = Runtime([a, b])
    selections = [((a.r, 1), 1), ((b.r, 1), 1)]
    authority = deepcopy(rt.states)
    rec = launch(rt, 'mixed', selections)
    require(not verify_view(authority, selections, rec), 'mixed request view')
    require(len(rec.inverse) == 2 and len(rec.targets) == 2, 'local identities collapsed')
    broken = deepcopy(rec)
    broken.inverse = {(row.r[1], row.branch, row.g, row.p): i
                      for i, row in enumerate(broken.rows)}
    expect_rejected(lambda: require(not verify_view(authority, selections, broken),
                                    'omitted request_id'),
                    'placement_key_without_request_id', observations)
    rt.complete(rec, rt.certificate(rec), {(a.r, 1): '17', (b.r, 1): '99'})
    require(a.events == ((0, 'TOKEN', '17'),), 'A result attribution')
    require(b.events == ((0, 'TOKEN', '99'),), 'B result attribution')
    require(len(rt.output_commits) == len(rt.sample_commits) == 2,
            'cross-request commit namespace collision')
    require(a.rng == b.rng == 1, 'independent symbolic RNG states')
    finish_consumers(rt, rec)
    require(rec.release_count == 1, 'mixed view release')
    audit_pages(a, rt.pool)
    audit_pages(b, rt.pool)


def permutations_and_reference(observations):
    a = State(('P', 1), 1, ('a', 'b', 'c'), 3)
    b = State(('Q', 1), 1, ('X',), 5)
    c = State(('Q', 1), 2, ('X',), 7,
              versions=('model-1', 'adapter-2', 'processor-1'))
    rt = Runtime([a, b, c])
    selections = [((a.r, 1), 2), ((b.r, 1), 1), ((c.r, 2), 1)]
    authority = deepcopy(rt.states)
    rec = launch(rt, 'permutation', selections)
    for order in permutations(range(4)):
        candidate = deepcopy(rec)
        candidate.rows = [rec.rows[i] for i in order]
        refresh(candidate)
        require(not verify_view(authority, selections, candidate), 'row permutation ownership')
    bad = deepcopy(rec)
    row = bad.columns[0]
    bad.columns[0] = (row[0], row[1] + 100, row[2], row[3])
    expect_rejected(lambda: require(not verify_view(authority, selections, bad), 'array fault'),
                    'derived_parameter_array', observations)
    bad = deepcopy(rec)
    bad.rows = [replace(row, param=row.param + 100) for row in rec.rows]
    refresh(bad)
    expect_rejected(lambda: require(not verify_view(authority, selections, bad), 'synchronized fault'),
                    'row_and_arrays_consistently_wrong', observations)
    bad = deepcopy(rec)
    bad.rows = [replace(row, g=row.g + 1) for row in rec.rows]
    refresh(bad)
    require(verify_view(authority, selections, bad), 'authority missed wrong attempt')
    bad = deepcopy(rec)
    bad.rows[0] = replace(bad.rows[0], versions=('wrong', 'wrong', 'wrong'))
    refresh(bad)
    require(verify_view(authority, selections, bad), 'authority missed wrong version')
    return 24


def prefill_trace(observations):
    s = State(('prefill', 1), 1, tuple('p' + str(i) for i in range(128)), 9)
    prompt = s.x
    rt = Runtime([s])
    key = (s.r, 1)
    first = launch(rt, 'prefill-1', [(key, 64)])
    require(s.active == first.view_id and s.k == 0, 'publication binding/order')
    cert1 = rt.certificate(first)
    require(rt.complete(first, rt.certificate(first, drained=False)) == 'drain_wait',
            'early completion')
    require(state_tuple(s) == (0, 128, (), 0, (), 0), 'state changed before drain')
    require(not first.released and not first.claimed, 'premature claim/release')
    rt.complete(first, cert1)
    require(state_tuple(s) == (64, 128, (), 0, (), 0), 'partial prefill transition')
    require(s.x == prompt and not first.started and first.release_count == 1, 'pure KV side effects')
    require(s.active is None and first.business_done, 'first binding not ended')
    audit_pages(s, rt.pool)
    first_handles = set(s.kv.values())
    require(len(first_handles) == 128, 'two-layer symbolic page coverage')
    for handle in first_handles:
        page = rt.pool.entry(handle)
        require(not page.execution and page.owners == {key}, 'history lifetime after finalize')
        require(not rt.pool.reusable(handle), 'first-round history may be reused')
    v1 = s.v
    require(rt.complete(first, cert1) == 'duplicate_noop', 'first repeat')
    require(state_tuple(s) == (64, 128, (), 0, (), 0) and s.v == v1, 'pure KV replay')
    second = launch(rt, 'prefill-2', [(key, 64)])
    require(second.segments[0].p0 == 64 and second.segments[0].p1 == 128, 'next input boundary')
    require(second.targets[0][1][-1] == 127, 'sample row off by one')
    require(set(second.history[key].values()) == first_handles, 'second history mapping')
    require(first_handles.isdisjoint(second.new_pages[key].values()), 'live history reallocated')
    for handle in first_handles:
        require(second.view_id in rt.pool.entry(handle).execution, 'history read unprotected')
    cert2 = rt.certificate(second)
    rt.complete(second, cert2, {key: 'Y'})
    wanted = (128, 129, ((0, 'TOKEN', 'Y'),), 1, ('Y',), 1)
    require(state_tuple(s) == wanted and s.x == prompt + ('Y',), 'terminal prefill transition')
    holders = set(second.holders)
    require(rt.complete(second, cert2, {key: 'other'}) == 'duplicate_noop', 'sample repeat')
    require(state_tuple(s) == wanted and second.holders == holders, 'sample replay side effects')
    cid = next(iter(holders))
    rt.consumer_done(second, cid)
    epoch = rt.epoch
    require(not rt.consumer_done(second, cid), 'duplicate consumer accepted')
    require(second.release_count == 1 and rt.epoch == epoch == 2, 'repeat release')
    audit_pages(s, rt.pool)
    require(len(s.kv) == 256, 'full prompt page coverage')
    for handle in first_handles:
        require(not rt.pool.entry(handle).execution, 'second history execution ref leaked')
        require(rt.pool.entry(handle).owners == {key}, 'second finalize removed long-lived owner')
    require(rt.complete(first, cert1) == 'duplicate_noop', 'old view after slot reuse')
    require(state_tuple(s) == wanted and rt.epoch == 2, 'late old view mutation')

    def faulty_partial():
        bad = State(('partial-mutant', 1), 1, prompt, 9)
        model = Runtime([bad])
        rec = launch(model, 'partial-mutant-view', [((bad.r, 1), 64)])
        model.complete(rec, model.certificate(rec), faulty_kv_emit=True)
        require(state_tuple(bad) == (64, 128, (), 0, (), 0), 'intermediate sampling detected')
    expect_rejected(faulty_partial, 'pure_KV_forced_output', observations)
    return [[64, 128, 0, 0], [128, 129, 1, 1]]


def sibling_consumers(observations, faulty=False):
    r = ('siblings', 1)
    left = State(r, 1, ('X',), 21)
    right = State(r, 2, ('X',), 22)
    rt = Runtime([left, right])
    rec = launch(rt, 'siblings-view', [((r, 1), 1), ((r, 2), 1)], bad_cid=faulty)
    cert = rt.certificate(rec)
    rt.complete(rec, cert, {(r, 1): 'same', (r, 2): 'same'})
    require(left.events == right.events == ((0, 'TOKEN', 'same'),), 'sibling output loss')
    require(left.rng == right.rng == 1, 'sibling RNG loss')
    cid1, cid2 = rec.planned[(r, 1)], rec.planned[(r, 2)]
    rt.consumer_done(rec, cid1)
    require(not rec.released and rt.owner is not None, 'right consumer still outstanding')
    require(cid2 in rec.holders, 'left completion cleared right responsibility')
    require(not rt.consumer_done(rec, cid1), 'duplicate left consumed another responsibility')
    require(rec.holders == {cid2} and not rec.released, 'duplicate left changed holders')
    require(rt.complete(rec, cert) == 'duplicate_noop', 'duplicate device event')
    require(rec.holders == {cid2}, 'device repeat cleared consumer')
    rt.consumer_done(rec, cid2)
    require(rec.released and rec.release_count == 1 and rt.epoch == 1, 'last consumer release')
    require(not rt.consumer_done(rec, cid2) and rt.epoch == 1, 'second release')
    audit_pages(left, rt.pool)
    audit_pages(right, rt.pool)
    if not faulty:
        expect_rejected(lambda: sibling_consumers(observations, faulty=True),
                        'consumer_id_view_and_ordinal_only', observations)


def active_binding_check(observations):
    def run(omit):
        s = State(('binding', 1), 1, ('a', 'b'), 1)
        rt = Runtime([s])
        key = (s.r, 1)
        rec = rt.build('binding-view', [(key, 1)], omit_active=omit)
        require(rec.steps[:2] == ['draft_checked', 'published_E_owner_bindings_and_references'],
                'publication order changed')
        rt.prepare_done(rec)
        result = rt.submit(rec)
        if result == 'submitted':
            rt.complete(rec, rt.certificate(rec))
        else:
            require(result == 'aborted' and s.k == 0, 'missing binding advanced state')
        # Fixed, independently stated legal-work expectation from k=0,n=2.
        require(state_tuple(s) == (1, 2, (), 0, (), 0), 'legal completion did not advance')
        require(rec.release_count == 1 and s.active is None, 'binding cleanup')
        audit_pages(s, rt.pool)
    run(False)
    expect_rejected(lambda: run(True), 'active_view_registration_omitted', observations)


def ownership_paths(observations):
    for mode in ('KV_ONLY', 'SAMPLE', 'SAMPLE_FAIL'):
        def run(omit_owner=False, mode=mode):
            s = State(('owner-' + mode, 1), 1, ('a', 'b'), 1)
            rt = Runtime([s])
            key = (s.r, 1)
            q = 1 if mode == 'KV_ONLY' else 2
            rec = launch(rt, 'owner-view-' + mode, [(key, q)])
            proposals = {} if mode == 'KV_ONLY' else {key: '<FAIL>' if mode == 'SAMPLE_FAIL' else 'Y'}
            rt.complete(rec, rt.certificate(rec), proposals, omit_owner=omit_owner)
            finish_consumers(rt, rec)
            wanted = ((1, 2, (), 0, (), 0) if mode == 'KV_ONLY' else
                      (2, 2, (), 0, (), 0) if mode == 'SAMPLE_FAIL' else
                      (2, 3, ((0, 'TOKEN', 'Y'),), 1, ('Y',), 1))
            require(state_tuple(s) == wanted, 'ownership path semantic boundary')
            require(rec.released and s.active is None, 'ownership test still relies on E')
            require(all(not rt.pool.entry(h).execution for h in s.kv.values()),
                    'execution reference masked missing owner')
            audit_pages(s, rt.pool)
            if mode == 'SAMPLE_FAIL':
                require(not rt.sample_commits and not rt.output_commits and not rec.started,
                        'sampling failure committed output')
                require(not s.live and s.v == 1, 'sampling failure revision/termination')
        run()
        expect_rejected(lambda run=run: run(True), 'long_owner_omitted_' + mode, observations)


def preparation_submission_edges():
    for mode in ('prepare_fail', 'known_rejected', 'cancel_preparing', 'cancel_ready', 'unknown'):
        s = State(('prepare-' + mode, 1), 1, ('a', 'b'), 1)
        rt = Runtime([s])
        key = (s.r, 1)
        rec = rt.build('prepare-view-' + mode, [(key, 1)])
        require(s.active == rec.view_id and not rec.business_done, 'publish did not bind')
        require(not rec.claimed and s.k == 0, 'publication committed computation')
        if mode == 'prepare_fail':
            rt.prepare_done(rec, success=False)
        elif mode == 'cancel_preparing':
            s.live = False
            rt.abort_unsubmitted(rec)
            require(rec.business_done and not rec.released and s.active == rec.view_id,
                    'cancel ignored outstanding preparation')
            require(('prep', rec.view_id) in rec.holders, 'cancel fabricated preparation end')
            rt.prepare_done(rec)
        else:
            rt.prepare_done(rec)
            if mode == 'cancel_ready':
                s.live = False
                require(rt.submit(rec) == 'aborted', 'canceled ready view submitted')
            elif mode == 'known_rejected':
                require(rt.submit(rec, 'rejected') == 'rejected', 'known rejection')
            else:
                require(rt.submit(rec, 'unknown') == 'unknown', 'unknown status lost')
                require(not rec.released and s.active == rec.view_id,
                        'unknown submission cleared binding')
                require(('device', rec.view_id) in rec.holders, 'unknown lost device responsibility')
                s.live = False
                rt.complete(rec, rt.certificate(rec))
        require(state_tuple(s) == (0, 2, (), 0, (), 0), 'abort changed semantic state')
        require(rec.business_done and rec.release_count == 1 and s.active is None,
                'abort did not resolve and release once')
        require(all(rt.pool.reusable(h) for h in rec.page_refs), 'discarded reservation not reusable')
        require(rt.prepare_done(rec) == 'duplicate_noop' and rt.epoch == 1, 'prepare replay released twice')

    # A real allocator reuses only an unowned, fully drained symbolic cell.
    s = State(('reuse-pages', 1), 1, ('a', 'b'), 1)
    rt = Runtime([s])
    key = (s.r, 1)
    old = rt.build('reuse-old', [(key, 1)])
    rt.prepare_done(old)
    rt.submit(old, 'rejected')
    handles = set(old.page_refs)
    new = rt.build('reuse-new', [(key, 1)])
    require({h[0] for h in new.page_refs} == {h[0] for h in handles}, 'free pages not reused')
    require(all(h[1] == 1 for h in new.page_refs), 'page reuse epoch not advanced')
    rt.prepare_done(new)
    rt.submit(new)
    rt.complete(new, rt.certificate(new))
    audit_pages(s, rt.pool)


def invalidation_and_edges():
    require(Runtime([]).build('empty', []) is None, 'empty work launched')
    for q in (0, -1, 2):
        s = State(('edge', 1), 1, ('X',), 1)
        rt = Runtime([s])
        try:
            rt.build('illegal', [((s.r, 1), q)])
        except ValueError:
            require(rt.owner is None and s.active is None and not rt.pool.pages,
                    'illegal plan leaked reservation')
        else:
            raise AssertionError('illegal q accepted')
    s = State(('duplicate', 1), 1, ('X',), 1)
    rt = Runtime([s])
    try:
        rt.build('duplicate-work', [((s.r, 1), 1), ((s.r, 1), 1)])
    except ValueError:
        require(rt.owner is None and not rt.pool.pages, 'duplicate plan leaked resources')
    else:
        raise AssertionError('duplicate branch accepted')
    for mode in ('cancel', 'old_attempt', 'old_lifetime', 'missing_layer', 'wrong_certificate'):
        s = State(('invalid', 1), 1, ('X',), 1)
        rt = Runtime([s])
        key = (s.r, 1)
        rec = launch(rt, 'invalid-' + mode, [(key, 1)])
        cert = rt.certificate(rec)
        if mode == 'cancel':
            s.live = False
        elif mode == 'old_attempt':
            s.g += 1
        elif mode == 'old_lifetime':
            replacement = State(('invalid', 2), 1, ('new',), 1)
            s.live = False
            rt.states[(replacement.r, 1)] = replacement
        elif mode == 'missing_layer':
            cert['writes'] = tuple(w for w in cert['writes'] if w[0] != LAYERS[0])
        else:
            wrong = dict(cert)
            wrong['epoch'] += 1
            require(rt.complete(rec, wrong, {key: 'bad'}) == 'identity_mismatch', 'wrong epoch accepted')
            require(not rec.claimed and not rec.released, 'wrong evidence consumed responsibility')
            s.live = False
        rt.complete(rec, cert, {key: 'bad'})
        require(state_tuple(s) == (0, 1, (), 0, (), 0), 'invalid result changed state')
        require(not rec.started and rec.release_count == 1 and s.active is None,
                'discard left phantom responsibility or binding')
        require(all(rt.pool.reusable(h) for h in rec.new_pages[key].values()),
                'discard retained uncommitted pages')
        require(rt.complete(rec, cert, {key: 'bad'}) == 'duplicate_noop', 'invalid replay')
        require(rt.epoch == 1, 'invalid duplicate released twice')
        if mode == 'old_lifetime':
            require(state_tuple(replacement) == (0, 1, (), 0, (), 0), 'new lifetime polluted')
    s = State(('cancel-second', 1), 1, ('a', 'b'), 1)
    rt = Runtime([s])
    key = (s.r, 1)
    first = launch(rt, 'cancel-second-1', [(key, 1)])
    rt.complete(first, rt.certificate(first))
    old_handles = set(s.kv.values())
    second = launch(rt, 'cancel-second-2', [(key, 1)])
    s.live = False
    rt.complete(second, rt.certificate(second), {key: 'bad'})
    require(state_tuple(s) == (1, 2, (), 0, (), 0), 'cancel lost prior KV boundary')
    require(set(s.kv.values()) == old_handles, 'discard changed old page mapping')
    audit_pages(s, rt.pool)
    require(all(rt.pool.reusable(h) for h in second.new_pages[key].values()),
            'discarded second block still reserved')
    s = State(('EOS', 1), 1, ('X',), 1)
    rt = Runtime([s])
    key = (s.r, 1)
    rec = launch(rt, 'EOS-view', [(key, 1)])
    rt.complete(rec, rt.certificate(rec), {key: '<EOS>'})
    require(state_tuple(s) == (1, 1, ((0, 'STOP', '<EOS>'),), 1, ('<EOS>',), 1), 'EOS contract')
    require(not s.live, 'EOS did not terminate')
    rt.consumer_done(rec, rec.planned[key])
    require(rec.release_count == 1, 'EOS resource completion')
    audit_pages(s, rt.pool)


def layer_certificate_edges():
    for mode in ('one_position_missing', 'logits_only', 'logits_missing', 'not_visible'):
        s = State(('layer-' + mode, 1), 1, ('a', 'b'), 1)
        rt = Runtime([s])
        key = (s.r, 1)
        rec = launch(rt, 'layer-view-' + mode, [(key, 2)])
        cert = rt.certificate(rec)
        require(len(cert['writes']) == 4, 'certificate domain should be two layers by two positions')
        if mode == 'one_position_missing':
            cert['writes'] = cert['writes'][:-1]
            require(len({layer for layer, full in cert['writes']}) == 2,
                    'negative case must still mention every layer')
        elif mode == 'logits_only':
            cert['writes'] = ()
        elif mode == 'logits_missing':
            cert['logits'] = ()
        else:
            cert['visible'] = False
        rt.complete(rec, cert, {key: 'must-not-commit'})
        require(state_tuple(s) == (0, 2, (), 0, (), 0), 'incomplete layer/result evidence committed')
        require(rec.release_count == 1 and not rec.started, 'drained failure cleanup')
        require(all(rt.pool.reusable(h) for h in rec.page_refs), 'failed writes became long-lived')


def main():
    negatives = []
    mixed_requests(negatives)
    permutation_count = permutations_and_reference(negatives)
    trajectory = prefill_trace(negatives)
    sibling_consumers(negatives)
    active_binding_check(negatives)
    ownership_paths(negatives)
    preparation_submission_edges()
    invalidation_and_edges()
    layer_certificate_edges()
    expected_negatives = {
        'placement_key_without_request_id', 'derived_parameter_array',
        'row_and_arrays_consistently_wrong', 'pure_KV_forced_output',
        'consumer_id_view_and_ordinal_only', 'active_view_registration_omitted',
        'long_owner_omitted_KV_ONLY', 'long_owner_omitted_SAMPLE',
        'long_owner_omitted_SAMPLE_FAIL'
    }
    require(set(negatives) == expected_negatives and len(negatives) == len(expected_negatives),
            'negative-control coverage mismatch')
    print(json.dumps({
        'scope': 'CPU deterministic symbolic identity/state/page-reference model; no GPU, numerical KV/logits, sampling distribution, network or performance measurement',
        'cross_request_equal_local_ids': 'two distinct rows, targets and logical output commits',
        'row_permutations_checked': permutation_count,
        'prefill_steps_k_n_outputs_rng': trajectory,
        'prefill_history_after_first_finalize': '128 symbolic layer-position pages retain owners; none reusable',
        'retained_KV_paths': ['KV_ONLY', 'SAMPLE', 'SAMPLE_FAIL'],
        'publication': 'draft validation, atomic publication/binding, preparation, submission, completion, final release',
        'sibling_same_ordinal_consumers': 'first and duplicate first preserve second; last releases once',
        'edges': ['empty plan', 'illegal q', 'duplicate branch', 'cancel', 'old attempt',
                  'old lifetime', 'wrong certificate', 'EOS', 'old view after slot reuse',
                  'prepare failure', 'known rejection', 'cancel during preparation',
                  'cancel before submission', 'unknown submission then drain', 'free-page epoch reuse'],
        'completion_evidence_rejected': ['whole layer missing', 'one position missing while both layers listed',
                                         'logits only', 'required logits missing', 'not visible'],
        'negative_controls_detected': negatives,
        'result': 'PASS'
    }, ensure_ascii=True, sort_keys=True))


if __name__ == '__main__':
    main()
