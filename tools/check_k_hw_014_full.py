import itertools
import json
from fractions import Fraction


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def covered(length, ranges):
    if length < 0:
        raise ValueError('negative length')
    end = 0
    for lo, hi in sorted(ranges):
        if not (0 <= lo < hi <= length):
            raise ValueError('invalid range')
        if lo > end:
            return False
        end = max(end, hi)
    return end == length


def byte_reference(length, ranges):
    actual = set()
    for lo, hi in ranges:
        actual.update(range(lo, hi))
    return actual == set(range(length))


def exact_plan(length, ranges):
    end = 0
    for lo, hi in sorted(ranges):
        if lo != end or not (lo < hi <= length):
            return False
        end = hi
    return end == length


def plan_reference(length, ranges):
    counts = [0] * length
    for lo, hi in ranges:
        if not (0 <= lo < hi <= length):
            return False
        for address in range(lo, hi):
            counts[address] += 1
    return all(count == 1 for count in counts)


def replicas_ready(length, targets, done):
    return all(covered(length, done.get(target, [])) for target in targets)


def transfer_candidate(events, fault=''):
    local_pending = {0, 1}
    remote_pending = {0, 1}
    visible = False
    cancelled = False
    published = False
    for event in events:
        if event in ('s0', 's1'):
            local_pending.discard(int(event[1]))
        elif event in ('r0', 'r1'):
            remote_pending.discard(int(event[1]))
        elif event == 'v':
            if fault == 'early_visible' or not remote_pending:
                visible = True
        elif event == 'c':
            cancelled = True
            if fault == 'cancel_drains':
                local_pending.clear()
                remote_pending.clear()
        elif event == 'p':
            if not cancelled and visible:
                if fault == 'early_visible' or not remote_pending:
                    published = True
    return (not local_pending, not remote_pending, published)


def transfer_reference(events):
    positions = {event: i for i, event in enumerate(events)}
    def pos(event):
        return positions.get(event, float('inf'))
    return (
        's0' in positions and 's1' in positions,
        'r0' in positions and 'r1' in positions,
        pos('r0') < pos('v') and pos('r1') < pos('v')
        and pos('v') < pos('p') and pos('p') < pos('c'),
    )


def convert_chunks(tokens, heads, channels, chunk, fault=''):
    source = list(range(1, tokens * heads * channels + 1))
    result = [None] * len(source)
    for lo in range(0, tokens, chunk):
        hi = min(tokens, lo + chunk)
        wire = source[lo * heads * channels:hi * heads * channels]
        for i, value in enumerate(wire):
            t = lo + i // (heads * channels)
            h = (i // channels) % heads
            c = i % channels
            address = (h * tokens + t) * channels + c
            if fault == 'no_transpose':
                address = lo * heads * channels + i
            result[address] = value
    return result


def layout_reference(tokens, heads, channels):
    source = list(range(1, tokens * heads * channels + 1))
    return [source[(t * heads + h) * channels + c]
            for h in range(heads)
            for t in range(tokens)
            for c in range(channels)]


def loses_wakeup(actions, arrival_position):
    cq = 0
    armed = False
    events = 0
    blocked = False
    for i in range(len(actions) + 1):
        if i == arrival_position:
            cq += 1
            if armed:
                events += 1
                armed = False
            if blocked and events:
                events -= 1
                blocked = False
        if i == len(actions):
            break
        action = actions[i]
        if action == 'ARM':
            armed = True
        elif action == 'POLL':
            cq = 0
        elif action == 'WAIT':
            if events:
                events -= 1
            else:
                blocked = True
        else:
            raise ValueError('unknown action')
    return cq > 0 and blocked and events == 0


def pipeline(count, transfer, conversion, slots):
    if count < 0 or transfer < 0 or conversion < 0 or slots <= 0:
        raise ValueError('invalid pipeline parameters')
    net_free = Fraction(0)
    convert_free = Fraction(0)
    slot_free = [Fraction(0)] * slots
    lifetimes = []
    for i in range(count):
        slot = i % slots
        start = max(net_free, slot_free[slot])
        arrived = start + transfer
        finish = max(arrived, convert_free) + conversion
        net_free = arrived
        convert_free = finish
        slot_free[slot] = finish
        lifetimes.append((start, finish, slot))
    return convert_free, lifetimes


def main():
    observations = {}
    negatives = []
    coverage_cases = 0
    for length in range(9):
        for chunk in range(1, 5):
            plan = [(lo, min(lo + chunk, length))
                    for lo in range(0, length, chunk)]
            check(exact_plan(length, plan) == plan_reference(length, plan),
                  'plan/reference disagreement')
            for mask in range(1 << len(plan)):
                completed = [part for i, part in enumerate(plan)
                             if mask & (1 << i)]
                check(covered(length, completed) == byte_reference(length, completed),
                      'coverage/reference disagreement')
                coverage_cases += 1
    observations['coverage_cases'] = coverage_cases
    check(covered(0, []) and exact_plan(0, []), 'empty domain')
    check(not covered(5, [(0, 2), (3, 5)]), 'hole not detected')
    overlap = [(0, 2), (0, 2)]
    check(not covered(4, overlap), 'overlap cannot fill missing bytes')
    faulty_byte_total = sum(hi - lo for lo, hi in overlap) == 4
    check(faulty_byte_total != byte_reference(4, overlap), 'byte-total negative missed')
    negatives.append('byte_total_instead_of_range_coverage')
    check(not exact_plan(4, overlap) and not plan_reference(4, overlap),
          'overlapping distinct planned ranges accepted')
    targets = [('B', 1), ('C', 1)]
    done = {('B', 1): [(0, 3)]}
    independent_replica_truth = all(
        byte_reference(3, done.get(target, [])) for target in targets)
    check(replicas_ready(3, targets, done) == independent_replica_truth,
          'replica reference disagreement')
    faulty_collapsed_target = covered(3, done[('B', 1)])
    check(faulty_collapsed_target != independent_replica_truth, 'replica negative missed')
    negatives.append('collapse_distinct_destination_replicas')
    done[('C', 1)] = [(0, 3)]
    check(replicas_ready(3, targets, done), 'repaired replica plan rejected')

    histories = 0
    for ordering in itertools.permutations(('s0', 's1', 'r0', 'r1', 'v', 'p', 'c')):
        for size in range(len(ordering) + 1):
            prefix = ordering[:size]
            check(transfer_candidate(prefix) == transfer_reference(prefix),
                  'event history disagreement: ' + repr(prefix))
            histories += 1
    observations['event_prefix_comparisons'] = histories
    for fault, trace in (('early_visible', ('v', 'p')),
                         ('cancel_drains', ('c',))):
        check(transfer_candidate(trace, fault) != transfer_reference(trace),
              'event negative missed: ' + fault)
        check(transfer_candidate(trace) == transfer_reference(trace),
              'repaired event candidate rejected')
        negatives.append(fault)
    check(transfer_candidate(('s0', 's1')) == (True, False, False),
          'local completion became remote readiness')
    check(transfer_candidate(('r0', 'v', 'p'))[2] is False,
          'partial receive published')
    check(transfer_candidate(('r0_error', 'r1', 'v', 'p'))[2] is False,
          'error completion became arrival')
    check(transfer_candidate(('old_generation', 'p')) == (False, False, False),
          'stale completion changed current attempt')

    layout_cases = 0
    for tokens in range(6):
        for heads in range(1, 4):
            for channels in range(1, 5):
                for chunk in range(1, 4):
                    check(convert_chunks(tokens, heads, channels, chunk)
                          == layout_reference(tokens, heads, channels),
                          'chunk layout/reference disagreement')
                    layout_cases += 1
    observations['layout_cases'] = layout_cases
    check(convert_chunks(5, 2, 3, 2, 'no_transpose')
          != layout_reference(5, 2, 3), 'transpose negative missed')
    check(convert_chunks(5, 2, 3, 2) == layout_reference(5, 2, 3),
          'repaired transpose rejected')
    negatives.append('omit_NHD_to_HND_permutation')
    check([min(2, 5 - lo) * 2 * 3 * 2 for lo in range(0, 5, 2)]
          == [24, 24, 12], 'tail payload size')

    owned = {('slot', 2, 'a'), ('slot', 2, 'b')}
    completions = [('slot', 1, 'a'), ('slot', 2, 'a'), ('slot', 2, 'a')]
    active = set(owned)
    for ticket in completions:
        active.discard(ticket)
    independent_active = owned.difference(set(completions))
    check(active == independent_active == {('slot', 2, 'b')},
          'duplicate or stale completion removed another holder')
    faulty_count = len(owned)
    for ticket in completions:
        if ticket in owned:
            faulty_count -= 1
    check(faulty_count != len(independent_active), 'duplicate-count negative missed')
    negatives.append('duplicate_completion_decrements_other_holder')
    active.discard(('slot', 2, 'b'))
    check(not active, 'last actual holder did not drain')

    correct_actions = ('ARM', 'POLL', 'WAIT')
    for at in range(4):
        check(not loses_wakeup(correct_actions, at), 'correct notification order lost wakeup')
    check(loses_wakeup(('POLL', 'ARM', 'WAIT'), 1), 'lost-wakeup negative missed')
    negatives.append('poll_before_rearm_then_wait')
    observations['notification_arrival_positions'] = 4
    # An event can remain after its CQE was polled; it must not settle another WR.
    pending_wr = {'another_wr'}
    queued_event = 1
    queued_cqe = []
    queued_event -= 1
    for wr in queued_cqe:
        pending_wr.discard(wr)
    check(queued_event == 0 and pending_wr == {'another_wr'}, 'empty event became completion')

    transfer = Fraction(32, 1024 * 20) * 1000 + Fraction(5, 1000)
    conversion = Fraction(4, 5)
    expected = transfer + conversion + 15 * max(transfer, conversion)
    two_slots, lifetimes = pipeline(16, transfer, conversion, 2)
    one_slot, _ = pipeline(16, transfer, conversion, 1)
    check(two_slots == expected == Fraction(2588, 100), 'two-slot pipeline cost')
    check(one_slot == 16 * (transfer + conversion) == Fraction(3788, 100),
          'one-slot serial cost')
    check(pipeline(0, transfer, conversion, 2)[0] == 0, 'empty pipeline')
    for slot in range(2):
        intervals = [(start, finish) for start, finish, owner in lifetimes if owner == slot]
        check(all(intervals[i][1] <= intervals[i + 1][0]
                  for i in range(len(intervals) - 1)), 'slot reused before final reader')
    endpoints = sorted({point for start, finish, _ in lifetimes for point in (start, finish)})
    peak = max(sum(start <= (lo + hi) / 2 < finish
                   for start, finish, _ in lifetimes)
               for lo, hi in zip(endpoints, endpoints[1:]))
    check(peak == 2, 'independent live-interval peak')
    check(512 + 2 * 32 == 576 and 512 + 576 == 1088, 'payload capacity accounting')
    observations['pipeline_ms'] = {'two_slots': str(two_slots), 'one_slot': str(one_slot)}
    observations['negative_controls_detected'] = negatives
    observations['scope'] = (
        'Deterministic CPU scalar/event model only. Hardware completion and CUDA ordering '
        'are assumed input events. No RDMA, GPU, DMA, registration, driver recovery, '
        'device visibility, performance measurement, or Agent effectiveness is tested.'
    )
    print(json.dumps(observations, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
