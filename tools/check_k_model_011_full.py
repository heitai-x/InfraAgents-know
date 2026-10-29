#!/usr/bin/env python3
'''Deterministic, offline, read-only CPU model for K-MODEL-011.
Only Python stdlib. No files, processes, network, project imports or devices.
Samples are canonical indices. Token choices replace learned logits.
'''
from fractions import Fraction


class Mismatch(AssertionError):
    pass


class HarnessError(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise Mismatch(message)


def negative(label, check):
    try:
        check()
    except Mismatch as error:
        print('negative detected: ' + label + ': ' + str(error))
    else:
        raise HarnessError('negative control accepted: ' + label)


def boundary(frame, fs, fa):
    return frame * fs // fa


class Speech:
    '''Synthetic acoustic adapter, one pending block and bounded queue.
    DATA uses two bytes/sample; END uses one byte and one event slot.
    The final block holds one sample until two resumable flush units.
    External feed/seal are used only by the acoustic-adapter fixtures.
    GeneratedSpeech below obtains code/EOS through its TalkerStep chain.
    '''
    def __init__(self, bugs=(), lookahead=1, end_policy='pad_crop',
                 startup=True, cap=16, slots=1, bad_length=False):
        self.fs, self.fa, self.nf, self.codebooks = 5, 3, 2, 2
        self.lookback, self.lookahead = 2, lookahead
        self.end_policy, self.startup = end_policy, startup
        self.cap, self.slots = cap, slots
        self.bugs, self.bad_length = set(bugs), bad_length
        self.epoch = self.current_epoch = 1
        self.cancelled = False
        self.eos_known, self.final_frames = False, None
        self.ready = [set() for _ in range(self.codebooks)]
        self.phase = 'RUN'
        self.next_decode = self.emit = self.overlap = 0
        self.pending = None
        self.flush_stage, self.flush_work_done = 0, False
        self.acoustic_calls = 0
        self.queue, self.trace = [], []
        self.event('INIT')

    def event(self, kind, **fields):
        self.trace.append(dict(kind=kind, next=self.next_decode,
                               emit=self.emit, overlap=self.overlap,
                               phase=self.phase,
                               qbytes=sum(x[1] for x in self.queue),
                               qsize=len(self.queue), **fields))

    def provide(self, c, frames):
        frames = tuple(frames)
        if self.eos_known or not 0 <= c < self.codebooks or any(j < 0 for j in frames):
            raise HarnessError('invalid input operation')
        self.ready[c].update(frames)
        self.event('CODE', c=c, frames=frames)

    def feed(self, count):
        for c in range(self.codebooks):
            self.provide(c, range(count))

    def seal(self, frames):
        if self.eos_known or frames < 0:
            raise HarnessError('EOS must be sealed once')
        if any(r != set(range(frames)) for r in self.ready):
            raise HarnessError('EOS fixture contains an incomplete/extra frame')
        if boundary(frames, self.fs, self.fa) < self.emit:
            raise HarnessError('EOS retracts committed samples')
        self.eos_known, self.final_frames = True, frames
        self.event('EOS', frames=frames)

    def cancel(self):
        self.cancelled = True
        self.event('CANCEL')

    def reuse_slot(self):
        self.current_epoch += 1
        self.event('REUSE', epoch=self.current_epoch)

    def nblocks(self):
        if not self.eos_known:
            raise HarnessError('unknown final block count')
        return (self.final_frames + self.nf - 1) // self.nf

    def active(self):
        return not self.cancelled and self.current_epoch == self.epoch

    def block_frames(self, b):
        stop = (b + 1) * self.nf
        if self.eos_known:
            stop = min(stop, self.final_frames)
        return range(b * self.nf, stop)

    def complete(self, b):
        if b < 0 or (self.eos_known and b >= self.nblocks()):
            return False
        books = range(1) if 'one_book' in self.bugs else range(self.codebooks)
        return all(j in self.ready[c] for c in books for j in self.block_frames(b))

    def tail_ok(self):
        return self.end_policy == 'pad_crop' or 'implicit_tail' in self.bugs

    def reason(self, b):
        if not self.complete(b):
            return 'WAIT_CODE'
        if b - self.lookback < 0 and not self.startup:
            return 'FAILED_START_POLICY'
        for k in range(b - self.lookback, b + self.lookahead + 1):
            if k < 0:
                continue
            if k > b and 'ignore_lookahead' in self.bugs:
                continue
            if self.eos_known and k >= self.nblocks():
                if not self.tail_ok():
                    return 'FAILED_END_POLICY'
                continue
            if not self.complete(k):
                return 'WAIT_CONTEXT'
            if (self.eos_known and (k + 1) * self.nf > self.final_frames
                    and not self.tail_ok()):
                return 'FAILED_END_POLICY'
        return 'READY'

    def build(self, b):
        lo = boundary(b * self.nf, self.fs, self.fa)
        end_frame = (b + 1) * self.nf
        if self.eos_known:
            end_frame = min(end_frame, self.final_frames)
        hi = boundary(end_frame, self.fs, self.fa)
        origin = max(0, lo - 2)
        raw = list(range(origin, hi + 1))
        wave = [v for i, v in enumerate(raw) if lo <= origin + i < hi]
        if self.bad_length:
            wave = wave[:-1]
        if 'overlap' in self.bugs:
            wave = raw
        final = self.eos_known and b == self.nblocks() - 1
        held = wave[-1:] if final else []
        self.pending = dict(b=b, start=lo, end=hi, epoch=self.epoch,
                            base=self.overlap, final=final,
                            wave=wave[:-1] if final else wave, held=held)
        self.acoustic_calls += 1
        if final:
            self.phase = 'FLUSH'
        self.event('BUILD', block=b)

    def stop(self):
        self.pending = None
        self.phase = 'CANCELLED'
        self.event('STOP')
        return 'CANCELLED'

    def publish(self):
        p = self.pending
        if p is None:
            raise HarnessError('publish without pending')
        if not self.active() and 'unguarded' not in self.bugs:
            return self.stop()
        if len(p['wave']) != p['end'] - p['start'] and 'overlap' not in self.bugs:
            if 'advance_failure' in self.bugs:
                self.next_decode += 1
            self.phase = 'FAILED'
            self.event('REJECT', reason='BLOCK_LENGTH_MISMATCH')
            return 'BLOCK_LENGTH_MISMATCH'
        if (p['b'] != self.next_decode or p['start'] != self.emit
                or p['base'] != self.overlap):
            self.phase = 'FAILED'
            self.event('REJECT', reason='OFFSET_OR_STATE')
            return 'OFFSET_OR_STATE'
        size = 2 * len(p['wave'])
        if (size > self.cap or self.slots < 1) and 'no_credit' not in self.bugs:
            self.phase = 'FAILED'
            self.event('REJECT', reason='FAILED_QUEUE_CONFIG')
            return 'FAILED_QUEUE_CONFIG'
        if (len(self.queue) >= self.slots
                or sum(x[1] for x in self.queue) + size > self.cap):
            if 'no_credit' not in self.bugs:
                self.event('WAIT', reason='WAIT_SEND')
                return 'WAIT_SEND'
        self.queue.append(('DATA', size))
        self.emit, self.next_decode = p['end'], p['b'] + 1
        self.overlap += 1
        self.pending = None
        self.phase = 'END' if p['final'] else 'RUN'
        self.event('COMMIT', block=p['b'], start=p['start'], end=p['end'],
                   epoch=p['epoch'], wave=tuple(p['wave']))
        return 'COMMITTED'

    def advance(self, after_compute=None):
        if self.phase in ('DONE', 'FAILED', 'CANCELLED'):
            return self.phase
        if not self.active():
            return self.stop()
        if self.phase == 'FLUSH':
            if not self.flush_work_done:
                self.flush_stage += 1
                self.event('FLUSH_STEP', stage=self.flush_stage)
                if self.flush_stage < 2:
                    return 'PAUSED_BUDGET'
                if self.pending is not None:
                    self.pending['wave'].extend(self.pending['held'])
                    self.pending['held'] = []
                self.flush_work_done = True
                self.event('FLUSH_DONE')
                return 'PROGRESS'
            if self.pending is not None:
                return self.publish()
            if self.emit != boundary(self.final_frames, self.fs, self.fa):
                self.phase = 'FAILED'
                self.event('REJECT', reason='FLUSH_LENGTH')
                return 'FLUSH_LENGTH'
            self.phase = 'END'
            self.event('PHASE')
            return 'PROGRESS'
        if self.phase == 'END':
            if not self.active():
                return self.stop()
            if (not self.eos_known or self.next_decode != self.nblocks()
                    or self.emit != boundary(self.final_frames, self.fs, self.fa)
                    or not self.flush_work_done or self.pending is not None):
                raise HarnessError('END entered before logical completion')
            if self.slots < 1 or self.cap < 1:
                self.phase = 'FAILED'
                self.event('REJECT', reason='FAILED_QUEUE_CONFIG')
                return 'FAILED_QUEUE_CONFIG'
            if len(self.queue) >= self.slots or sum(x[1] for x in self.queue) + 1 > self.cap:
                if 'no_credit' not in self.bugs:
                    self.event('WAIT', reason='WAIT_SEND')
                    return 'WAIT_SEND'
            self.queue.append(('END', 1))
            self.phase = 'DONE'
            self.event('END', epoch=self.epoch)
            return 'DONE'
        if self.pending is not None:
            return self.publish()
        if 'wait_before_end' in self.bugs and not self.complete(self.next_decode):
            self.event('WAIT', reason='WAIT_CODE')
            return 'WAIT_CODE'
        if self.eos_known and self.next_decode == self.nblocks():
            self.phase = 'FLUSH'
            self.event('PHASE')
            return 'PROGRESS'
        reason = self.reason(self.next_decode)
        if reason != 'READY':
            if reason.startswith('FAILED'):
                self.phase = 'FAILED'
                self.event('REJECT', reason=reason)
            else:
                self.event('WAIT', reason=reason)
            return reason
        self.build(self.next_decode)
        if after_compute is not None:
            after_compute(self)
        if self.phase == 'FLUSH':
            return 'PROGRESS'
        return self.publish()

    def drain(self):
        if not self.queue:
            raise HarnessError('draining an empty queue')
        kind, size = self.queue.pop(0)
        self.event('DRAIN', item=kind, size=size)


def validate(m, expect_done=False):
    '''Independent observable-event replay; never calls candidate methods.'''
    epoch, cancelled, final = 1, False, None
    ready = [set() for _ in range(m.codebooks)]
    blocks, samples, queue = [], [], []
    flush_steps, flush_completions, ends = 0, 0, 0
    for e in m.trace:
        kind = e['kind']
        if kind == 'CODE':
            ready[e['c']].update(e['frames'])
        elif kind == 'EOS':
            need(final is None, 'duplicate EOS')
            final = e['frames']
            need(all(r == set(range(final)) for r in ready), 'EOS has incomplete frames')
            need(len(samples) <= boundary(final, m.fs, m.fa), 'EOS retracts output')
        elif kind == 'CANCEL':
            cancelled = True
        elif kind == 'REUSE':
            epoch = e['epoch']
        elif kind == 'FLUSH_STEP':
            flush_steps += 1
            need(final is not None and e['stage'] == flush_steps, 'wrong flush resume point')
            need(flush_steps <= 2, 'flush repeated after pause/backpressure')
        elif kind == 'FLUSH_DONE':
            flush_completions += 1
            need(final is not None and flush_steps == 2, 'unjustified flush completion')
            need(flush_completions == 1, 'duplicate flush completion')
        elif kind == 'COMMIT':
            need(not cancelled and e['epoch'] == epoch, 'commit after cancellation/slot reuse')
            b = len(blocks)
            need(e['block'] == b, 'skipped or repeated block')
            real_end = (b + 1) * m.nf if final is None else min((b + 1) * m.nf, final)
            lo = boundary(b * m.nf, m.fs, m.fa)
            hi = boundary(real_end, m.fs, m.fa)
            need(e['start'] == lo == len(samples) and e['end'] == hi, 'sample boundary error')
            need(e['wave'] == tuple(range(lo, hi)), 'payload gap/overlap/padded sample')
            if b - m.lookback < 0:
                need(m.startup, 'undeclared startup')
            n = None if final is None else (final + m.nf - 1) // m.nf
            for k in range(b - m.lookback, b + m.lookahead + 1):
                if k < 0:
                    continue
                if n is not None and k >= n:
                    need(m.end_policy == 'pad_crop', 'implicit zero tail accepted')
                    continue
                stop = (k + 1) * m.nf if final is None else min((k + 1) * m.nf, final)
                for c in range(m.codebooks):
                    for j in range(k * m.nf, stop):
                        need(j in ready[c], 'missing context/codebook frame committed')
                if final is not None and (k + 1) * m.nf > final:
                    need(m.end_policy == 'pad_crop', 'partial tail has no legal policy')
            if final is not None and b == n - 1:
                need(flush_completions == 1, 'final block before stable flush')
            samples.extend(e['wave'])
            blocks.append(b)
            queue.append(('DATA', 2 * len(e['wave'])))
        elif kind == 'END':
            need(not cancelled and e['epoch'] == epoch, 'stale/cancelled END')
            need(final is not None and flush_completions == 1, 'END before EOS/flush')
            need(len(samples) == boundary(final, m.fs, m.fa), 'END loses samples')
            need(len(blocks) == (final + m.nf - 1) // m.nf, 'END skips block')
            ends += 1
            need(ends == 1, 'duplicate END')
            queue.append(('END', 1))
        elif kind == 'DRAIN':
            need(bool(queue), 'drain without committed event')
            need(queue.pop(0) == (e['item'], e['size']), 'send order/accounting mismatch')
        need(e['next'] == len(blocks), 'cursor advanced without successful publication')
        need(e['emit'] == len(samples), 'emit differs from committed sample prefix')
        need(e['overlap'] == len(blocks), 'tentative overlap became committed')
        need(e['qsize'] == len(queue), 'queue event count mismatch')
        need(e['qbytes'] == sum(x[1] for x in queue), 'queue bytes mismatch')
        need(len(queue) <= m.slots and sum(x[1] for x in queue) <= m.cap, 'queue over credit')
        if e['phase'] == 'DONE':
            need(ends == 1, 'DONE without END transaction')
    if expect_done:
        need(m.phase == 'DONE' and ends == 1, 'finite legal input failed to reach DONE')
    return tuple(samples)


def drive(m):
    statuses = []
    for _ in range(80):
        status = m.advance()
        statuses.append(status)
        if m.phase in ('DONE', 'FAILED', 'CANCELLED'):
            return statuses
        if status == 'WAIT_SEND':
            if not m.queue:
                return statuses
            m.drain()
        elif status in ('WAIT_CODE', 'WAIT_CONTEXT', 'WAIT_CONDITION'):
            return statuses
    raise HarnessError('acoustic fixture exceeded bounded steps')


def ready_request(frames, **kwargs):
    m = Speech(**kwargs)
    m.feed(frames)
    m.seal(frames)
    return m


class GeneratedSpeech(Speech):
    '''Actual synthetic TalkerStep -> AdvanceSpeech result-dispatch chain.
    Each consume models an atomic completed position, without network math.
    Each choice is an input tuple: CODE/book/frame/value or EOS.
    This class does not use feed, provide or seal to generate its fixtures.
    '''
    def __init__(self, choices, omit_dispatch=False, **kwargs):
        super().__init__(**kwargs)
        self.choices = tuple(tuple(x) for x in choices)
        self.choice_i = 0
        self.known = [('BOS',)]
        self.n = 0
        self.logits_cursor = None
        self.rng = 0
        self.talker_calls = 0
        self.condition_ready = True
        self.cancel_on_call = None
        self.eligible, self.quarantined = True, False
        self.failure_reason = None
        self.omit_dispatch = omit_dispatch
        self.slice_i = 0
        self.control = []

    def active(self):
        return super().active() and self.eligible

    def log(self, kind, **fields):
        self.control.append(dict(kind=kind, slice=self.slice_i,
                                 phase=self.phase, reason=self.failure_reason,
                                 known=tuple(self.known), n=self.n, rng=self.rng,
                                 calls=self.talker_calls, eligible=self.eligible,
                                 quarantined=self.quarantined,
                                 ready=tuple(tuple(sorted(x)) for x in self.ready),
                                 eos=self.eos_known, final=self.final_frames,
                                 next=self.next_decode, emit=self.emit,
                                 overlap=self.overlap, audio_mark=len(self.trace),
                                 **fields))

    def result(self, tag, detail=None):
        self.log('RESULT', tag=tag, detail=detail)
        return tag, detail

    def terminate(self, phase, reason):
        if self.phase not in ('DONE', 'FAILED', 'CANCELLED'):
            self.failure_reason = reason
            self.eligible = False
            self.quarantined = True
            self.pending = None
            self.phase = phase
            self.event('CONTROL_STOP', reason=reason)
            self.log('TERMINAL')
        return self.phase

    def talker_step(self):
        self.talker_calls += 1
        self.log('CALL')
        if self.cancel_on_call == self.talker_calls:
            self.cancel()
        if not self.active():
            return self.result('CANCELLED', 'CANCELLED')
        if self.eos_known:
            return self.result('FAILED', 'FAILED_AFTER_EOS')
        if self.n < len(self.known):
            if not self.condition_ready:
                return self.result('WAIT', 'WAIT_CONDITION')
            i = self.n
            token = self.known[i]
            self.n += 1
            self.logits_cursor = self.n if self.n == len(self.known) else None
            self.log('CONSUME', index=i, token=token)
            return self.result('COMPUTED')
        if self.logits_cursor != self.n:
            return self.result('FAILED', 'FAILED_MISSING_LOGITS')
        if self.choice_i >= len(self.choices):
            return self.result('FAILED', 'FAILED_NO_CHOICE')
        y = self.choices[self.choice_i]
        self.choice_i += 1
        self.known.append(y)
        self.rng += 1
        self.logits_cursor = None
        self.log('SELECT', token=y)
        if y[0] == 'CODE':
            _, c, j, value = y
            if (not 0 <= c < self.codebooks or not 0 <= value < 8
                    or j != len(self.ready[c])):
                return self.result('FAILED', 'FAILED_CODE_RANGE_OR_POSITION')
            self.ready[c].add(j)
            self.event('CODE', c=c, frames=(j,))
            return self.result('SELECTED')
        if y == ('EOS',):
            counts = [len(r) for r in self.ready]
            if len(set(counts)) != 1:
                return self.result('FAILED', 'FAILED_INCOMPLETE_FRAME')
            frames = counts[0]
            if any(r != set(range(frames)) for r in self.ready):
                return self.result('FAILED', 'FAILED_INCOMPLETE_FRAME')
            if boundary(frames, self.fs, self.fa) < self.emit:
                return self.result('FAILED', 'FAILED_PREFIX_RETRACTION')
            self.eos_known, self.final_frames = True, frames
            self.event('EOS', frames=frames)
            return self.result('EOS')
        return self.result('FAILED', 'FAILED_TOKEN_KIND')

    def advance(self, budget=4):
        if not isinstance(budget, int) or budget < 0:
            raise HarnessError('budget must be a nonnegative integer')
        self.slice_i += 1
        self.log('ENTER', budget=budget)
        used = 0

        def finish(status):
            self.log('RETURN', status=status)
            return status

        while True:
            if self.phase in ('DONE', 'FAILED', 'CANCELLED'):
                return finish(self.phase)
            if not self.active():
                return finish(self.terminate('CANCELLED', 'CANCELLED'))
            # Re-evaluate semantic dependencies before considering another unit.
            self.log('RECHECK')
            if self.phase != 'RUN' or self.pending is not None:
                if used >= budget:
                    return finish('PAUSED_BUDGET')
                status = super().advance()
                self.log('DOWNSTREAM', status=status)
                return finish(status)
            if self.eos_known and self.next_decode == self.nblocks():
                self.phase = 'FLUSH'
                self.event('PHASE')
                continue
            reason = self.reason(self.next_decode)
            if reason.startswith('FAILED'):
                return finish(self.terminate('FAILED', reason))
            if reason == 'READY':
                if used >= budget:
                    return finish('PAUSED_BUDGET')
                status = super().advance()
                self.log('DOWNSTREAM', status=status)
                return finish(status)
            if self.eos_known:
                return finish(reason)
            if used >= budget:
                return finish('PAUSED_BUDGET')
            tag, detail = self.talker_step()
            used += 1
            if self.omit_dispatch:
                # Fault: the chapter's former unconditional continuation.
                continue
            if tag in ('COMPUTED', 'SELECTED'):
                continue
            if tag == 'EOS':
                if not self.eos_known or self.final_frames is None:
                    return finish(self.terminate('FAILED', 'FAILED_EOS_RESULT'))
                continue
            if tag == 'WAIT':
                return finish(detail)
            if tag == 'CANCELLED':
                return finish(self.terminate('CANCELLED', detail))
            if tag == 'FAILED':
                return finish(self.terminate('FAILED', detail))
            return finish(self.terminate('FAILED', 'FAILED_TALKER_RESULT'))


def control_signature(e):
    return tuple(e[k] for k in ('known', 'n', 'rng', 'calls', 'ready',
                                'eos', 'final', 'next', 'emit', 'overlap'))


def validate_control(m, expect_done=False):
    '''Independent caller-contract oracle from observed input/output events.
    Reconstructs known/consumed/selected positions; never calls TalkerStep,
    AdvanceSpeech, result dispatch, readiness or termination implementations.
    Terminal obligations derive from the returned result type, not phase.
    '''
    known, n, selected, calls = [('BOS',)], 0, 0, 0
    terminal, frozen, fatal_mark = None, None, None
    wait = None
    recheck_due = False
    accepted_eos = False
    tags = set()
    selected_codes = [set() for _ in range(m.codebooks)]
    for e in m.control:
        kind = e['kind']
        if terminal is not None:
            need(kind in ('TERMINAL', 'RETURN', 'ENTER'),
                 'work or loop re-entry after terminal Talker result')
        if wait is not None:
            need(kind == 'RETURN', 'WAIT result retried instead of returned')
        if recheck_due:
            need(kind == 'RECHECK', 'progress/EOS result skipped dependency recheck')
        if kind == 'CALL':
            need(not accepted_eos, 'Talker called after accepted EOS')
            calls += 1
        elif kind == 'CONSUME':
            need(e['index'] == n and n < len(known), 'wrong consumed position')
            need(e['token'] == known[n], 'consumed token differs from selected input')
            n += 1
        elif kind == 'SELECT':
            need(n == len(known), 'selected before consuming known input')
            need(selected < len(m.choices) and e['token'] == m.choices[selected],
                 'selection differs from deterministic input sequence')
            selected += 1
            known.append(e['token'])
            if e['token'][0] == 'CODE':
                _, c, j, value = e['token']
                if 0 <= c < m.codebooks and 0 <= value < 8:
                    selected_codes[c].add(j)
        elif kind == 'RESULT':
            tag, detail = e['tag'], e['detail']
            tags.add(tag)
            if tag in ('FAILED', 'CANCELLED'):
                terminal = ('FAILED' if tag == 'FAILED' else 'CANCELLED', detail)
                frozen, fatal_mark = control_signature(e), e['audio_mark']
                if detail == 'FAILED_INCOMPLETE_FRAME':
                    need(len({tuple(sorted(x)) for x in selected_codes}) > 1,
                         'half-frame fixture did not actually create unequal codebooks')
                    need(not e['eos'] and e['final'] is None,
                         'invalid EOS installed a final boundary')
            elif tag == 'WAIT':
                wait = detail
            elif tag in ('COMPUTED', 'SELECTED', 'EOS'):
                recheck_due = True
                if tag == 'EOS':
                    need(e['eos'] and e['final'] is not None, 'EOS lacks final boundary')
                    need(all(x == set(range(e['final'])) for x in selected_codes),
                         'EOS accepted incomplete selected codebooks')
                    accepted_eos = True
            else:
                raise Mismatch('unexpected Talker result in fixture')
        elif kind == 'RECHECK':
            recheck_due = False
        elif kind == 'RETURN':
            if terminal is not None:
                phase, reason = terminal
                need(e['status'] == phase and e['phase'] == phase,
                     'terminal result hidden by pause/wait/progress')
                need(e['reason'] == reason, 'terminal reason lost or replaced')
                need(not e['eligible'] and e['quarantined'], 'failed suffix remains eligible')
            elif wait is not None:
                need(e['status'] == wait, 'actual wait reason was replaced')
                need(e['phase'] == 'RUN', 'temporary condition wait became terminal')
                wait = None
        if terminal is not None and kind != 'RESULT':
            need(control_signature(e) == frozen, 'token/code or committed prefix changed after failure')
        need(e['known'] == tuple(known) and e['n'] == n,
             'known/calculated frontiers disagree with event replay')
        need(e['rng'] == selected and e['calls'] == calls,
             'selection/RNG/call accounting disagrees with replay')
    need(wait is None and not recheck_due, 'control obligation left unresolved')
    if fatal_mark is not None:
        need(not any(e['kind'] in ('COMMIT', 'END', 'BUILD')
                     for e in m.trace[fatal_mark:]),
             'new DATA/END/acoustic work after terminal Talker result')
    samples = validate(m, expect_done)
    return samples, tags


def drive_generated(m, budget=4):
    statuses = []
    for _ in range(240):
        status = m.advance(budget)
        statuses.append(status)
        if m.phase in ('DONE', 'FAILED', 'CANCELLED'):
            return statuses
        if status == 'WAIT_SEND':
            if not m.queue:
                raise HarnessError('WAIT_SEND without a drainable fixture event')
            m.drain()
        elif status.startswith('WAIT'):
            return statuses
    raise HarnessError('generated fixture exceeded bounded steps')


def codes(frames):
    return [('CODE', c, j, (j + c) % 8)
            for j in range(frames) for c in range(2)]


def half_frame_fixture(omit_dispatch=False):
    # Preparation itself traverses AdvanceSpeech -> TalkerStep.
    # BOS consumed, first code selected, then that code consumed.
    # The next invocation's FIRST TalkerStep selects the invalid EOS.
    choices = [('CODE', 0, 0, 1), ('EOS',),
               ('CODE', 1, 0, 2), ('CODE', 0, 1, 3), ('CODE', 1, 1, 4), ('EOS',)]
    m = GeneratedSpeech(choices, omit_dispatch=omit_dispatch, lookahead=0)
    need(m.advance(3) == 'PAUSED_BUDGET', 'half-frame preparation did not use three units')
    need(m.talker_calls == 3 and m.n == len(m.known), 'half-frame preparation frontiers')
    need(m.ready == [{0}, set()] and not m.eos_known, 'half-frame preparation input')
    validate_control(m)
    return m


def talker_checks():
    observed_tags = set()
    for budget in (1, 4):
        m = half_frame_fixture()
        before = (m.talker_calls, m.emit, m.next_decode, m.overlap)
        need(m.advance(budget) == 'FAILED', 'half-frame EOS was hidden by budget')
        need(m.failure_reason == 'FAILED_INCOMPLETE_FRAME', 'failure reason not preserved')
        need(m.talker_calls == before[0] + 1, 'Talker called again after failing first step')
        need(m.known[-1] == ('EOS',) and m.n == len(m.known) - 1,
             'fixture did not retain an appended, unconsumed invalid EOS')
        for _ in range(3):
            need(m.advance(10) == 'FAILED', 'failed request revived on re-entry')
        need((m.talker_calls, m.emit, m.next_decode, m.overlap)
             == (before[0] + 1, before[1], before[2], before[3]),
             'failed call changed output prefix or performed more generation')
        samples, tags = validate_control(m)
        observed_tags.update(tags)
        need(samples == () and not m.queue and m.acoustic_calls == 0,
             'half-frame failure published DATA/END or started acoustic work')
        print('Talker failure: budget=' + str(budget)
              + ', first step FAILED_INCOMPLETE_FRAME, one call, no DATA/END, terminal re-entry stable')

        bad = half_frame_fixture(omit_dispatch=True)
        bad.advance(budget)
        negative('Talker result dispatch removed, budget=' + str(budget),
                 lambda req=bad: validate_control(req))

    # The same omitted-dispatch mutant can consume its appended EOS and
    # complete code with later choices. Its actual trace uses the same oracle.
    bad = half_frame_fixture(omit_dispatch=True)
    bad.advance(20)
    negative('ignored failure continued consuming/selecting into acoustic output',
             lambda: validate_control(bad))

    # A previously committed prefix remains immutable after a later half frame.
    prefix = GeneratedSpeech(codes(2) + [('CODE', 0, 2, 5), ('EOS',)], lookahead=0)
    for _ in range(60):
        if prefix.choice_i == 5 and prefix.n == len(prefix.known):
            break
        status = prefix.advance(1)
        if status == 'WAIT_SEND':
            prefix.drain()
        need(prefix.phase == 'RUN', 'prefix preparation unexpectedly terminated')
    else:
        raise HarnessError('prefix preparation exceeded bound')
    need((prefix.emit, prefix.next_decode, prefix.overlap) == (3, 1, 1),
         'prefix fixture did not commit its first block')
    need(prefix.advance(1) == 'FAILED', 'later half-frame EOS not propagated')
    samples, tags = validate_control(prefix)
    observed_tags.update(tags)
    need(samples == (0, 1, 2), 'failure rewrote a legal committed prefix')
    frozen = (prefix.emit, prefix.next_decode, prefix.overlap, prefix.talker_calls)
    prefix.advance(20)
    validate_control(prefix)
    need(frozen == (prefix.emit, prefix.next_decode, prefix.overlap, prefix.talker_calls),
         'prefix changed on terminal re-entry')

    # Success cases traverse consumption/selection/EOS, not external feed/seal.
    for frames in (0, 3, 6):
        m = GeneratedSpeech(codes(frames) + [('EOS',)])
        statuses = drive_generated(m, budget=1)
        samples, tags = validate_control(m, True)
        observed_tags.update(tags)
        need(samples == tuple(range(boundary(frames, 5, 3))), 'generated EOS sample extent')
        need(m.n == len(m.known) - 1 and m.known[-1] == ('EOS',),
             'accepted EOS was consumed for an unnecessary successor prediction')
        need(m.failure_reason is None, 'successful EOS left a failure reason')
        need('PAUSED_BUDGET' in statuses, 'generated success did not test slice recovery')

    waiting = GeneratedSpeech(codes(1) + [('EOS',)])
    waiting.condition_ready = False
    need(waiting.advance(1) == 'WAIT_CONDITION', 'wait hidden by exhausted budget')
    need(waiting.talker_calls == 1 and waiting.n == waiting.rng == 0, 'wait consumed state')
    _, tags = validate_control(waiting)
    observed_tags.update(tags)
    waiting.condition_ready = True
    drive_generated(waiting, budget=2)
    samples, tags = validate_control(waiting, True)
    observed_tags.update(tags)
    need(samples == (0,), 'condition wake did not resume legal generation')

    cancelled = GeneratedSpeech(codes(1) + [('EOS',)])
    cancelled.cancel_on_call = 1
    need(cancelled.advance(1) == 'CANCELLED', 'Talker cancellation hidden by budget')
    need(cancelled.advance(20) == 'CANCELLED', 'cancelled request revived')
    samples, tags = validate_control(cancelled)
    observed_tags.update(tags)
    need(samples == () and cancelled.talker_calls == 1 and not cancelled.queue,
         'cancelled Talker produced output or another call')
    need(observed_tags >= {'COMPUTED', 'SELECTED', 'EOS', 'WAIT', 'CANCELLED', 'FAILED'},
         'missing Talker result-type coverage')
    print('Talker chain: progress recheck, empty/partial/full EOS, true condition wait, cancellation,')
    print('unconsumed failed EOS isolation and preserved committed prefix checked by event replay')


def arithmetic_checks():
    for fs, fa, nf, count in ((24000, 50, 10, 5), (48000, 441, 1, 441), (5, 3, 2, 7)):
        candidate = [boundary(k * nf, fs, fa) for k in range(count + 1)]
        exact = [Fraction(k * nf * fs, fa) for k in range(count + 1)]
        for got, ref in zip(candidate, exact):
            need(Fraction(got) <= ref < got + 1, 'individual rational boundary drift')
    offsets = [boundary(k, 48000, 441) for k in range(442)]
    need(offsets[-1] == 48000, 'rational total')
    need(set(b - a for a, b in zip(offsets, offsets[1:])) == {108, 109}, 'rational lengths')
    step = Fraction(48000, 441)
    rounded = (step.numerator + step.denominator - 1) // step.denominator

    def bad_rounding():
        for k in range(442):
            ref = k * step
            need(k * rounded <= ref < k * rounded + 1, 'fixed rounding causes drift')

    negative('fixed per-block rounding', bad_rounding)
    need(boundary(10, 24000, 50) == 4800, 'PCM geometry')
    need(4800 * 2 == 9600 and 24000 * 2 * Fraction(2, 5) == 19200, 'PCM bytes')
    first_ms = Fraction(20 * 1000, 500) + 25 + 5
    need(first_ms == 70 and (Fraction(10 * 1000, 500) + 25 + 5) / 200 == Fraction(1, 4),
         'first packet or serial utilization')
    print('geometry: exact boundaries, 108/109 alternation, 4800 samples and 9600 PCM bytes')
    print('design arithmetic: first arrival 70 ms; serial resource demand 25%, not a measurement')


def threshold_start(arrivals, target_samples, eos_time=None):
    total = 0
    for t, count in arrivals:
        total += count
        if total > 0 and total >= target_samples:
            return t
    if eos_time is not None and total > 0:
        if arrivals and eos_time < arrivals[-1][0]:
            raise HarnessError('ordered END before data')
        return eos_time
    return None


def check_start(arrivals, target, start, eos_time=None):
    times = sorted(set(t for t, _ in arrivals) | ({eos_time} if eos_time is not None else set()))
    eligible = []
    for t in times:
        n = sum(count for at, count in arrivals if at <= t)
        if n > 0 and (n >= target or (eos_time is not None and eos_time <= t)):
            eligible.append(t)
    need(start == (min(eligible) if eligible else None), 'not earliest audio threshold event')


def playout_validate(arrivals, start, fs):
    offset = 0
    for at, length in arrivals:
        for i in range(length):
            need(at <= start + Fraction(offset + i, fs), 'sample unavailable at playout')
        offset += length


def buffer_checks():
    arrivals = [(Fraction(70), 4800), (Fraction(120), 4800), (Fraction(170), 4800)]
    t100 = threshold_start(arrivals, 2400)
    t300 = threshold_start(arrivals, 7200)
    check_start(arrivals, 2400, t100)
    check_start(arrivals, 7200, t300)
    need((t100, t300) == (70, 120), 'whole-block audio threshold')
    wall = arrivals[0][0] + 100
    need(wall == 170, 'fixed wall delay')
    need(Fraction(sum(n for t, n in arrivals if t <= wall), 24000) == Fraction(3, 5),
         'actual audio buffer at wall-time start')
    negative('audio amount added as wall-clock delay', lambda: check_start(arrivals, 2400, wall))
    short = [(Fraction(70), 1200)]
    check_start(short, 2400, threshold_start(short, 2400, Fraction(90)), Fraction(90))
    need(threshold_start(short, 2400) is None, 'short unknown stream invented EOS')
    check_start([], 2400, threshold_start([], 2400, Fraction(90)), Fraction(90))
    playout_validate([(t / 1000, n) for t, n in arrivals], Fraction(7, 100), 24000)
    equal = [(Fraction(7, 100) + k * Fraction(1, 5), 4800) for k in range(3)]
    playout_validate(equal, Fraction(7, 100), 24000)
    late = [(Fraction(7, 100), 4800), (Fraction(1, 2), 4800)]
    negative('late block despite favorable average supply',
             lambda: playout_validate(late, Fraction(7, 100), 24000))
    print('startup: 100/300 ms audio thresholds start at 70/120 ms; fixed wait starts at 170 ms')
    print('startup edges: short stream waits for END; empty stream has no first sample')


def state_checks():
    empty = ready_request(0, end_policy=None)
    drive(empty)
    need(validate(empty, True) == (), 'empty stream emitted samples')
    full = ready_request(6)
    statuses = drive(full)
    need(validate(full, True) == tuple(range(10)), 'full canonical stream')
    need(full.acoustic_calls == 3, 'backpressure caused acoustic recomputation')
    need(statuses.count('WAIT_SEND') >= 3, 'ordinary/final/END waits not exercised')
    need('PAUSED_BUDGET' in statuses and full.flush_stage == 2, 'flush resume not exercised')
    wait_phases = {e['phase'] for e in full.trace
                   if e['kind'] == 'WAIT' and e['reason'] == 'WAIT_SEND'}
    need(wait_phases == {'RUN', 'FLUSH', 'END'}, 'missing backpressure phase')
    partial = ready_request(3)
    drive(partial)
    need(validate(partial, True) == tuple(range(5)), 'partial tail rounded to full block')
    unknown = Speech()
    unknown.feed(2)
    need(unknown.advance() == 'WAIT_CONTEXT', 'unknown future became terminal padding')
    for c in range(unknown.codebooks):
        unknown.provide(c, range(2, 4))
    need(unknown.advance() == 'COMMITTED', 'lookahead did not unblock block 0')
    unknown.seal(4)
    drive(unknown)
    need(validate(unknown, True) == tuple(range(6)), 'unknown EOS recovery')
    causal = Speech(lookahead=0)
    causal.feed(2)
    need(causal.advance() == 'COMMITTED', 'causal block fixture')
    causal.seal(2)
    drive(causal)
    need(validate(causal, True) == tuple(range(3)), 'EOS after complete causal prefix')
    missing = Speech()
    missing.provide(0, range(4))
    missing.provide(1, range(1))
    need(missing.advance() == 'WAIT_CODE', 'incomplete codebook accepted')
    validate(missing)
    for label, kwargs in (('startup', dict(startup=False)), ('tail', dict(end_policy=None))):
        m = ready_request(3, **kwargs)
        drive(m)
        validate(m)
        need(m.phase == 'FAILED', 'missing ' + label + ' policy accepted')
    failed = ready_request(2, bad_length=True)
    drive(failed)
    validate(failed)
    need(failed.phase == 'FAILED' and failed.next_decode == failed.emit == failed.overlap == 0,
         'publication failure changed committed state')
    tiny = ready_request(4, cap=2)
    drive(tiny)
    validate(tiny)
    need(tiny.phase == 'FAILED', 'impossible event size waits forever')
    for action in ('cancel', 'reuse_slot'):
        race = Speech()
        race.feed(4)
        result = race.advance(lambda req, op=action: getattr(req, op)())
        need(result == 'CANCELLED', 'post-compute transaction check absent')
        validate(race)
        need(race.emit == race.next_decode == race.overlap == 0, 'cancelled scratch committed')

    bad = ready_request(0, bugs=('wait_before_end',))
    drive(bad)
    negative('termination check after next-code access', lambda: validate(bad, True))
    bad = ready_request(2, bugs=('advance_failure',), bad_length=True)
    drive(bad)
    negative('advance after final publication failure', lambda: validate(bad))
    for action in ('cancel', 'reuse_slot'):
        bad = Speech(bugs=('unguarded',))
        bad.feed(4)
        bad.advance(lambda req, op=action: getattr(req, op)())
        negative('post-compute ' + action + ' without guard', lambda req=bad: validate(req))
    bad = ready_request(6, bugs=('no_credit',))
    drive(bad)
    negative('publication without queue credit', lambda: validate(bad))
    bad = Speech(bugs=('ignore_lookahead',))
    bad.feed(2)
    bad.advance()
    negative('lookahead omitted', lambda: validate(bad))
    bad = Speech(bugs=('one_book',))
    bad.provide(0, range(4))
    bad.provide(1, range(1))
    bad.advance()
    negative('only first codebook checked', lambda: validate(bad))
    bad = ready_request(3, end_policy=None, bugs=('implicit_tail',))
    drive(bad)
    negative('implicit zero tail without legal end policy', lambda: validate(bad))
    bad = ready_request(4, bugs=('overlap',))
    drive(bad)
    negative('raw overlap published as stable payload', lambda: validate(bad))
    restored = ready_request(3)
    drive(restored)
    need(validate(restored, True) == tuple(range(5)), 'restored implementation rejected')
    print('acoustic states: empty/unknown EOS, causal prefix, partial tail and legal crop checked')
    print('acoustic states: RUN/FLUSH/END backpressure, flush resume and failure preservation checked')
    print('full acoustic trace: ' + ' -> '.join(statuses))


def main():
    arithmetic_checks()
    buffer_checks()
    state_checks()
    talker_checks()
    restored = half_frame_fixture()
    need(restored.advance(1) == 'FAILED', 'restored dispatch did not propagate failure')
    validate_control(restored)
    print('Completed: deterministic CPU arithmetic, synthetic Talker control and acoustic event models')
    print('Not checked: learned network math, GPU visibility/lifetime, hardware timings or speech quality')


if __name__ == '__main__':
    main()
