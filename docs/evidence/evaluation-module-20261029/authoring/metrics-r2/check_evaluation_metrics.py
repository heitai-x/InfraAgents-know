#!/usr/bin/env python3
"""Finite synthetic metric/event checks; no service, GPU or probability claim."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction as F
from itertools import product
import json
import hashlib


@dataclass(frozen=True)
class Request:
    name: str
    planned: int
    ready: int
    sent: int
    decision: tuple[int, bool] | None
    content: tuple[tuple[int, int], ...]
    mode: str
    terminal: tuple[int, str] | None
    quality: tuple[int, bool] | None
    prompt: int
    prefill_work: int
    output_version: str = 'v1'
    certificate_version: str = 'v1'
    quality_events: tuple[tuple[str, int, bool, str, str], ...] = ()
    material_ready_at: int | None = None
    material_token_count: int | None = None


ROWS = (
    Request('A', 0, 0, 0, (1, True), ((10, 1), (20, 1), (30, 1)), 'TOKEN', (45, 'FINAL'), (180, True), 100, 20, material_ready_at=30, material_token_count=3),
    Request('B', 10, 10, 30, (31, True), ((40, 1), (50, 1), (60, 1)), 'TOKEN', (65, 'FINAL'), (75, True), 200, 200, material_ready_at=60, material_token_count=3),
    Request('C', 20, 20, 20, (21, True), ((50, 3),), 'CHUNK', (55, 'FINAL'), (70, True), 300, 100, material_ready_at=50, material_token_count=3),
    Request('D', 30, 30, 30, (35, False), (), 'TOKEN', (35, 'REJECT'), None, 400, 0),
    Request('E', 40, 40, 40, (41, True), ((60, 1),), 'TOKEN', (100, 'TIMEOUT'), None, 500, 500),
    Request('F', 50, 50, 50, (51, True), ((70, 1), (80, 1), (90, 1)), 'TOKEN', (170, 'FINAL'), None, 600, 600, material_ready_at=90, material_token_count=3),
    Request('G', 60, 60, 60, (90, True), (), 'TOKEN', (80, 'CANCEL'), None, 700, 0),
    Request('H', 70, 70, 70, (71, True), ((100, 1),), 'TOKEN', (110, 'FINAL'), (120, True), 800, 800, material_ready_at=100, material_token_count=1),
)


def validate(row: Request) -> None:
    if not (row.planned <= row.ready <= row.sent) or row.mode not in {'TOKEN', 'CHUNK'}:
        raise ValueError('request timing/mode')
    if row.prompt < 0 or row.prefill_work < 0:
        raise ValueError('negative count')
    if row.decision is not None:
        when, accepted = row.decision
        if when < row.sent or type(accepted) is not bool:
            raise ValueError('decision')
    previous = row.sent
    for when, size in row.content:
        if type(when) is not int or type(size) is not int or size <= 0 or when < previous:
            raise ValueError('content order/count')
        if row.mode == 'TOKEN' and size != 1:
            raise ValueError('TOKEN with grouped count')
        if row.decision is None or not row.decision[1] or when < row.decision[0]:
            raise ValueError('content without earlier admission')
        previous = when
    if row.terminal is not None:
        when, reason = row.terminal
        if when < previous or reason not in {'FINAL', 'REJECT', 'TIMEOUT', 'CANCEL', 'ERROR'}:
            raise ValueError('terminal')
        if reason == 'FINAL' and (row.decision is None or not row.decision[1] or row.decision[0] > when):
            raise ValueError('normal completion without effective admission')
    if (row.material_ready_at is None) != (row.material_token_count is None):
        raise ValueError('incomplete immutable-output descriptor')
    if row.material_ready_at is not None:
        if type(row.material_ready_at) is not int or type(row.material_token_count) is not int or row.material_token_count != sum(size for _, size in row.content):
            raise ValueError('immutable-output descriptor count/type')
    if row.quality is not None:
        if len(row.quality) != 2 or type(row.quality[0]) is not int:
            raise ValueError('quality shape/time')
        if row.material_ready_at is not None and row.quality[0] < row.material_ready_at:
            raise ValueError('quality before immutable material readiness')
        if row.output_version != row.certificate_version or type(row.quality[1]) is not bool:
            raise ValueError('quality identity/value')


def metrics(row: Request, horizon: int, fault: str = '') -> dict:
    validate(row)
    first = last = previous = None
    count = intervals = 0
    interval_sum = 0
    exact = True
    for when, size in row.content:
        if when > horizon:
            continue
        iterations = size if fault == 'invent_chunk_tokens' else 1
        for _ in range(iterations):
            if first is None:
                first = when
            if previous is not None:
                interval_sum += when - previous
                intervals += 1
            previous = last = when
        count += size
        exact = exact and (size == 1 or fault == 'invent_chunk_tokens')
    terminal = row.terminal if row.terminal and row.terminal[0] <= horizon else None
    completed = bool(terminal and terminal[1] == 'FINAL')
    start = row.planned if fault == 'planned_as_client_send' else row.sent
    tpot = F(interval_sum, intervals) if count > 1 and exact else None
    if fault == 'protocol_tpot' and completed and count > 1 and exact:
        tpot = F(terminal[0] - first, count - 1)
    if fault == 'single_token_zero' and count == 1:
        tpot = F(0)
    return {'n': count, 'completed': completed,
            'ttft_client': None if first is None else F(first - start),
            'ttft_offered': None if first is None else F(first - row.planned),
            'last_content_latency': None if last is None else F(last - row.sent),
            'protocol_latency': F(terminal[0] - row.sent) if completed else None,
            'tpot': tpot,
            'per_user_rate': F(1000 * (count - 1), last - first) if count > 1 and exact and last > first else None,
            'exact': exact, 'terminal': terminal}


def quality_status(row: Request, evidence: int) -> bool | None:
    events = list(row.quality_events)
    if row.quality:
        events.append(('primary', row.quality[0], row.quality[1], row.certificate_version, 'task-quality-v1'))
    seen = {}
    result = None
    for identity, when, verdict, output_version, rule in events:
        if type(when) is not int or type(verdict) is not bool:
            raise ValueError('certificate type')
        if when > evidence:
            continue
        payload = (when, verdict, output_version, rule)
        if identity in seen:
            if seen[identity] != payload:
                raise ValueError('certificate identity conflict')
            continue
        seen[identity] = payload
        if (row.material_ready_at is not None and when < row.material_ready_at) or output_version != row.output_version or rule != 'task-quality-v1':
            raise ValueError('certificate binding/content boundary')
        if result is not None and result != verdict:
            raise ValueError('contradictory fixed-rule verdict')
        result = verdict
    return result if row.material_ready_at is not None else None


def report(rows: tuple[Request, ...], horizon: int = 160, evidence: int = 200,
           window: tuple[int, int] = (0, 100), fault: str = '', cohort_key: str = 'planned') -> dict:
    lo, hi = window
    if not lo < hi <= horizon <= evidence or cohort_key not in {'planned', 'sent'}:
        raise ValueError('window/evidence/cohort')
    if len({row.name for row in rows}) != len(rows):
        raise ValueError('duplicate request identity')
    used_horizon = evidence if fault == 'judge_as_service_end' else horizon
    result = dict(offered=0, confirmed_accepts=0, effective_admitted=0, rejected=0,
                  preterminated=0, pending=0, normal=0, timeout=0, incomplete=0,
                  known_good=0, unknown_good=0, known_good_tokens=0,
                  cohort_output=0, completion_window_output=0, completion_window_normal=0,
                  emitted_window=0, offered_input=0, prefill_work=0,
                  tpot_eligible=0, tpot_observed=0, single_token=0)
    observed_tpots = []
    for row in rows:
        validate(row)
        if not lo <= getattr(row, cohort_key) < hi:
            continue
        result['offered'] += 1
        result['offered_input'] += row.prompt
        result['prefill_work'] += row.prompt if fault == 'offered_input_as_work' else row.prefill_work
        m = metrics(row, used_horizon, fault)
        decision = row.decision if row.decision and row.decision[0] <= used_horizon else None
        terminal = m['terminal']
        if decision and decision[1]:
            result['confirmed_accepts'] += 1
        preterminated = bool(terminal and terminal[1] not in {'FINAL', 'REJECT'} and
                             (decision is None or terminal[0] < decision[0]))
        if preterminated and fault != 'late_admission_is_effective':
            result['preterminated'] += 1
        elif decision is None:
            result['pending'] += 1
        elif not decision[1]:
            result['rejected'] += 1
        else:
            result['effective_admitted'] += 1
            if terminal is None:
                result['incomplete'] += 1
            elif terminal[1] == 'FINAL':
                result['normal'] += 1
            elif terminal[1] == 'TIMEOUT':
                result['timeout'] += 1
        if fault == 'completed_cohort_as_emission':
            if m['completed']:
                result['emitted_window'] += m['n']
        else:
            for when, size in row.content:
                inside = lo <= when <= hi if fault == 'closed_emission_window' else lo <= when < hi
                if inside:
                    result['emitted_window'] += size
        if not m['completed']:
            continue
        result['cohort_output'] += m['n']
        if lo <= terminal[0] < hi:
            result['completion_window_normal'] += 1
            result['completion_window_output'] += m['n']
        if m['n'] == 1:
            result['single_token'] += 1
        if m['n'] > 1:
            result['tpot_eligible'] += 1
            result['tpot_observed'] += m['tpot'] is not None
            if m['tpot'] is not None:
                observed_tpots.append(m['tpot'])
        latency = False if m['n'] == 0 or m['ttft_client'] > 35 else (
            True if m['n'] == 1 else None if m['tpot'] is None else m['tpot'] <= 15)
        quality = quality_status(row, evidence)
        if latency is False or quality is False:
            continue
        if latency is None or quality is None:
            if fault == 'unknown_as_pass':
                result['known_good'] += 1
                result['known_good_tokens'] += m['n']
            elif fault != 'unknown_as_failure':
                result['unknown_good'] += 1
        else:
            result['known_good'] += 1
            result['known_good_tokens'] += m['n']
    divisor = hi - lo if fault == 'arrival_window_as_goodput_duration' else used_horizon - lo
    result['request_throughput_cohort'] = F(1000 * result['normal'], used_horizon - lo)
    result['output_throughput_cohort'] = F(1000 * result['cohort_output'], used_horizon - lo)
    result['request_throughput_completion_window'] = F(1000 * result['completion_window_normal'], hi - lo)
    result['output_throughput_completion_window'] = F(1000 * result['completion_window_output'], hi - lo)
    result['output_throughput_emission_window'] = F(1000 * result['emitted_window'], hi - lo)
    result['goodput_lower'] = F(1000 * result['known_good'], divisor)
    result['goodput_upper'] = F(1000 * (result['known_good'] + result['unknown_good']), divisor)
    result['token_goodput_lower'] = F(1000 * result['known_good_tokens'], divisor)
    result['input_offered_rate'] = F(1000 * result['offered_input'], hi - lo)
    result['prefill_position_rate'] = F(1000 * result['prefill_work'], used_horizon - lo)
    result['qualified_fraction_lower'] = F(result['known_good'], result['offered']) if result['offered'] else None
    result['qualified_fraction_upper'] = F(result['known_good'] + result['unknown_good'], result['offered']) if result['offered'] else None
    result['tpot_coverage'] = F(result['tpot_observed'], result['tpot_eligible']) if result['tpot_eligible'] else None
    result['observed_mean_tpot_ms'] = sum(observed_tpots, F(0)) / len(observed_tpots) if observed_tpots else None
    return result


def direct_endpoint_reference(row: Request, horizon: int) -> tuple:
    events = [(when, size) for when, size in row.content if when <= horizon]
    if not events:
        return None, None, None
    n = sum(size for _, size in events)
    tpot = F(events[-1][0] - events[0][0], n - 1) if n > 1 and all(size == 1 for _, size in events) else None
    return F(events[0][0] - row.sent), F(events[0][0] - row.planned), tpot


class CheckError(AssertionError):
    pass


def same(actual, expected, label: str) -> None:
    if actual != expected:
        raise CheckError(f'{label}: {actual!r} != {expected!r}')


COMMON_FIXTURE_SHA256 = '4b1d581c771b2a5bbfeac81999f25445f43ab76fcf5e486c8274e97ab80306ce'


def main() -> None:
    encoded_fixture = json.dumps([asdict(row) for row in ROWS], sort_keys=True, separators=(',', ':')).encode('utf-8')
    same(hashlib.sha256(encoded_fixture).hexdigest(), COMMON_FIXTURE_SHA256, 'common fixture fingerprint')
    checks = 1
    for row in ROWS:
        for horizon in [0, 10, 50, 100, 160, 200]:
            m = metrics(row, horizon)
            same((m['ttft_client'], m['ttft_offered'], m['tpot']),
                 direct_endpoint_reference(row, horizon), f'endpoint {row.name}/{horizon}')
            checks += 1
    for n in [0, 1, 2, 3, 11]:
        for step in [0, 1, 7]:
            content = tuple((10 + k * step, 1) for k in range(n))
            end = 20 + max([when for when, _ in content], default=0)
            row = Request('generated', 0, 0, 2, (3, True), content, 'TOKEN', (end, 'FINAL'), (end + 1, True), 5, 5)
            m = metrics(row, end)
            same((m['ttft_client'], m['ttft_offered'], m['tpot']), direct_endpoint_reference(row, end), f'generated {n}/{step}')
            same(m['per_user_rate'], F(1000, step) if n > 1 and step > 0 else None, f'per user {n}/{step}')
            checks += 1
    expected_counts = dict(offered=8, confirmed_accepts=7, effective_admitted=6,
                           rejected=1, preterminated=1, pending=0, normal=4, timeout=1,
                           incomplete=1, known_good=3, unknown_good=1, known_good_tokens=7,
                           cohort_output=10, completion_window_output=9,
                           completion_window_normal=3, emitted_window=13,
                           offered_input=3600, prefill_work=2220,
                           tpot_eligible=3, tpot_observed=2, single_token=1)
    expected_rates = dict(request_throughput_cohort=F(25), output_throughput_cohort=F(125, 2),
                          request_throughput_completion_window=F(30), output_throughput_completion_window=F(90),
                          output_throughput_emission_window=F(130), goodput_lower=F(75, 4),
                          goodput_upper=F(25), token_goodput_lower=F(175, 4),
                          input_offered_rate=F(36000), prefill_position_rate=F(13875),
                          qualified_fraction_lower=F(3, 8), qualified_fraction_upper=F(1, 2),
                          tpot_coverage=F(2, 3), observed_mean_tpot_ms=F(10))
    expected = {**expected_counts, **expected_rates}
    observed = report(ROWS)
    same(observed, expected, 'hand reconstructed census/rates')
    checks += 1
    early = report(ROWS, evidence=160)
    same((early['known_good'], early['unknown_good'], early['goodput_lower'], early['goodput_upper']),
         (2, 2, F(25, 2), F(25)), 'early evidence')
    # Enumerate all admissible unresolved labels, independently of the report's bounds.
    for known, unresolved, label in [(2, 2, 'early'), (3, 1, 'late')]:
        possible = [known + sum(bits) for bits in product((0, 1), repeat=unresolved)]
        values = early if label == 'early' else observed
        same((values['goodput_lower'], values['goodput_upper']),
             (F(1000 * min(possible), 160), F(1000 * max(possible), 160)), 'enumerated bounds ' + label)
        checks += 1
    same((early['normal'], early['cohort_output'], early['request_throughput_cohort']),
         (observed['normal'], observed['cohort_output'], observed['request_throughput_cohort']), 'evidence does not move service time')
    checks += 1
    same(metrics(ROWS[0], 160)['tpot'], F(10), 'A exact TPOT')
    same(metrics(ROWS[0], 160, 'protocol_tpot')['tpot'], F(35, 2), 'A wrong protocol TPOT')
    checks += 2
    # Resource-side position identities are specified independently of the aggregate count fields.
    work_ranges = [('A', 80, 100), ('B', 0, 200), ('C', 200, 300),
                   ('E', 0, 500), ('F', 0, 600), ('H', 0, 800)]
    physical_positions = [(name, position) for name, start, stop in work_ranges for position in range(start, stop)]
    same(len(physical_positions), observed['prefill_work'], 'resource-side position ledger')
    work_events = [('draft', i) for i in range(24)] + [('target_verify', i) for i in range(30)]
    work_events += [('visible', row.name, offset) for row in ROWS for offset in range(sum(size for when, size in row.content if when <= 160))]
    same(tuple(sum(event[0] == category for event in work_events) for category in ['draft', 'target_verify', 'visible']),
         (24, 30, 14), 'speculative roles remain separate')
    # A real wrong reducer merges draft proposals into visible-output work.
    wrong_visible = sum(event[0] in {'draft', 'visible'} for event in work_events)
    if wrong_visible == 14:
        raise CheckError('draft/output role mutant survived')
    checks += 3
    per_user_mean = (F(1000, 10) + F(1000, 30)) / 2
    reciprocal_mean = F(1000) / ((F(10) + F(30)) / 2)
    same((per_user_mean, reciprocal_mean), (F(200, 3), F(50)), 'aggregation and inversion differ')
    checks += 1
    shifted = replace(ROWS[1], sent=130, decision=(131, True),
                      content=((140, 1), (150, 1), (160, 1)), terminal=(165, 'FINAL'), quality=(175, True), material_ready_at=160)
    changed = tuple(shifted if row.name == 'B' else row for row in ROWS)
    same(report(changed, horizon=260, evidence=300, cohort_key='planned')['offered'], 8, 'planned cohort keeps late send')
    same(report(changed, horizon=260, evidence=300, cohort_key='sent')['offered'], 7, 'sent cohort is a different population')
    same(metrics(shifted, 260)['ttft_client'], F(10), 'late send client TTFT')
    same(metrics(shifted, 260)['ttft_offered'], F(130), 'late send offered wait')
    checks += 4
    # Keep output/intervals fixed while moving only protocol completion.
    changed_final = replace(ROWS[0], terminal=(145, 'FINAL'), quality=(180, True))
    same(metrics(changed_final, 160)['tpot'], metrics(ROWS[0], 160)['tpot'], 'protocol-tail intervention')
    same(metrics(changed_final, 160)['protocol_latency'] - metrics(ROWS[0], 160)['protocol_latency'], F(100), 'protocol-tail effect')
    checks += 2
    early_quality = replace(ROWS[0], quality=(35, True))
    early_quality_rows = tuple(early_quality if row.name == 'A' else row for row in ROWS)
    same(report(early_quality_rows, evidence=160), observed, 'quality may precede protocol FINAL')
    same(metrics(early_quality, 160), metrics(ROWS[0], 160), 'judge timing does not change service metrics')
    same(quality_status(replace(ROWS[0], quality_events=(('primary', 180, True, 'v1', 'task-quality-v1'),)), 200),
         True, 'identical certificate delivery is idempotent')
    conflict = replace(ROWS[0], quality_events=(('later', 190, False, 'v1', 'task-quality-v1'),))
    same(quality_status(conflict, 185), True, 'future contradiction does not alter older evidence cutoff')
    checks += 4
    producer_early = replace(ROWS[0], material_ready_at=25, quality=(28, True))
    producer_early_rows = tuple(producer_early if row.name == 'A' else row for row in ROWS)
    same(report(producer_early_rows, evidence=160), observed, 'quality may precede client last content after producer SEAL')
    same(metrics(producer_early, 160), metrics(ROWS[0], 160), 'producer-side quality never changes client visibility')
    missing_descriptor = replace(ROWS[0], material_ready_at=None, material_token_count=None)
    missing_rows = tuple(missing_descriptor if row.name == 'A' else row for row in ROWS)
    same((report(missing_rows)['known_good'], report(missing_rows)['unknown_good']), (2, 2), 'no immutable output proof stays unknown')
    checks += 3
    rejected = []
    for fault in ['protocol_tpot', 'planned_as_client_send', 'invent_chunk_tokens',
                  'single_token_zero', 'late_admission_is_effective', 'completed_cohort_as_emission',
                  'closed_emission_window', 'judge_as_service_end', 'unknown_as_pass',
                  'unknown_as_failure', 'arrival_window_as_goodput_duration', 'offered_input_as_work']:
        try:
            if fault in {'protocol_tpot', 'planned_as_client_send', 'single_token_zero'}:
                for row in ROWS:
                    m = metrics(row, 160, fault)
                    same((m['ttft_client'], m['ttft_offered'], m['tpot']), direct_endpoint_reference(row, 160), fault)
            else:
                same(report(ROWS, fault=fault), expected, fault)
        except CheckError:
            rejected.append(fault)
        else:
            raise CheckError(f'mutant survived: {fault}')
    checks += len(rejected)
    for invalid in [replace(ROWS[0], certificate_version='other-output'),
                    replace(ROWS[2], mode='TOKEN'),
                    replace(ROWS[0], terminal=(25, 'CANCEL')),
                    replace(ROWS[0], quality=(25, True)),
                    conflict,
                    replace(ROWS[0], quality_events=(('primary', 181, True, 'v1', 'task-quality-v1'),))]:
        try:
            report(tuple(invalid if row.name == invalid.name else row for row in ROWS))
        except ValueError:
            checks += 1
        else:
            raise CheckError('invalid record admitted')
    empty = report(())
    same((empty['offered'], empty['normal'], empty['goodput_lower'], empty['goodput_upper']), (0, 0, F(0), F(0)), 'empty population rates')
    same((empty['qualified_fraction_lower'], empty['qualified_fraction_upper'], empty['tpot_coverage'], empty['observed_mean_tpot_ms']),
         (None, None, None, None), 'empty statistics stay undefined')
    checks += 1
    summary_keys = ['offered', 'confirmed_accepts', 'effective_admitted', 'normal', 'known_good', 'unknown_good', 'cohort_output', 'goodput_lower', 'goodput_upper', 'token_goodput_lower']
    common_summary = {key: str(observed[key]) if isinstance(observed[key], F) else observed[key] for key in summary_keys}
    print(json.dumps({'result': 'PASS', 'checks': checks,
                      'common_fixture_sha256': COMMON_FIXTURE_SHA256, 'common_fixture_summary': common_summary,
                      'mutants_rejected': rejected + ['draft_candidates_as_visible_tokens'],
                      'scope': 'Finite synthetic client metric, cohort, evidence and work-count models only; no production service, GPU, tokenizer, load generator or statistical coverage result.'}, sort_keys=True))


if __name__ == '__main__':
    main()
