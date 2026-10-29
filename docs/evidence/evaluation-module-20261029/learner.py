"""A fresh DS reasoning exercise, separate from authoring and semantic review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
KB = RUN.parents[2]
WORKSPACE = KB.parents[1]
sys.path.insert(0, str(WORKSPACE / 'knowledge-workflow' / 'src'))
from knowledge_workflow.config import load_config
from knowledge_workflow.model_clients import ModelClient

CONFIG = load_config(WORKSPACE / 'knowledge-workflow' / 'workflow.evaluation.local.toml')
DIRECTORY = RUN / 'learner'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Retain the prior artifact: ' + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    paths = ['knowledge/workloads/metric-events.md', 'knowledge/workloads/measurement-comparability.md',
             'knowledge/tools/statistical-experiments.md']
    materials = [{'path': name, 'sha256': hashlib.sha256((KB / name).read_bytes()).hexdigest(),
                  'content': (KB / name).read_text(encoding='utf-8-sig'), 'reading_scope': 'full'} for name in paths]
    task = {
        'participant': 'ds-learner:evaluation-module:initial',
        'task': 'Use only the supplied knowledge and this new scenario. Audit metrics and propose a defensible experiment. No tools, implementation source, author checks or review reports are provided. Return one strict JSON object; do not invent measurements.',
        'contract': {'units': 'milliseconds unless reporting rates per second; all listed times have an already valid common mapping',
            'arrival_cohort': 'actual sends in [0,100)', 'service_cutoff_H': 150,
            'quality_evidence_cutoffs_J': [150, 180], 'completion_and_emission_window': '[0,100)',
            'primary_success': 'normal FINAL by H, output quality PASS by J, client SEND-to-first-content <=35ms, exact token TPOT <=15ms; unobserved quantities remain unknown',
            'quality_identity': 'Each listed quality result binds that request complete immutable output. No other quality evidence exists.',
            'effective_admission': 'A request already terminated before an acceptance decision is not revived. Keep the actual server decision separately.'},
        'requests': [
            {'id': 'a', 'planned': 0, 'ready': 5, 'send': 20, 'accepted': 25,
             'token_times': [50, 60, 70], 'final': 80, 'quality': {'time': 120, 'value': 'PASS'}},
            {'id': 'b', 'planned': 10, 'ready': 10, 'send': 10, 'accepted': 12,
             'chunks': [{'time': 40, 'tokens': 2}], 'final': 130, 'quality': {'time': 160, 'value': 'PASS'}},
            {'id': 'c', 'planned': 90, 'ready': 90, 'send': 95, 'timeout': 140,
             'late_actual_acceptance': 145, 'output_tokens': 0},
        ],
        'questions': [
            'For a compute actual-send TTFT, planned-origin TTFT, exact TPOT and the distinct (protocol latency minus TTFT)/(N-1) value. State what can be recovered for b.',
            'Compute emission-window output tokens/s, completion-window output tokens/s, and cohort output tokens/s using elapsed H-0. Keep goodput separate.',
            'At each J, classify good, failed and unknown requests. Give certified goodput and identification bounds on possible goodput; these bounds are not confidence intervals.',
            'Count actual acceptance decisions and effective business admissions by H. Explain why late acceptance does not repair the timeout or imply resource completion.',
            'A new 2x2 table has latency means low-load baseline=12, low-load change=9, high-load baseline=15, high-load change=17ms. Derive both conditional effects and interaction, and decide whether the change is uniformly faster.',
            'A report has two process runs with 100 requests each, all baseline runs before all changed runs. It labels a 200-request pooled p99 as independent evidence. Explain what is and is not supported and redesign assignment, repetitions, warmup and normal versus diagnostic timing.',
            'Under a separate stipulated iid fixed-sample Bernoulli model only, 200 trials show zero SLO violations. Compute the 95% one-sided upper bound and assess a below-1% claim. Then discuss why two correlated service runs do not inherit that model. Give one valid planned-looks or independent-confirmation design and its resource/decision boundaries.',
        ],
        'answer_schema': {'metrics': {}, 'admission_accounts': {}, 'statistical_judgment': {},
                          'experiment_plan': {}, 'decision': '', 'limits': []},
    }
    save(DIRECTORY / 'task.json', task)
    save(DIRECTORY / 'materials.json', {'files': materials, 'scope': 'Only current chapter prose; no check code or feedback.'})
    record = ModelClient(CONFIG.adapters)('learner', json.dumps({'task': task, 'materials': materials}, ensure_ascii=False),
        DIRECTORY / 'initial-call', engine='ds', effort=CONFIG.models['effort'], timeout=CONFIG.execution['model_timeout'])
    if record.get('status') != 'completed' or not isinstance(record.get('parsed'), dict):
        raise RuntimeError('Learner call did not complete strict JSON; retained record is authoritative')
    save(DIRECTORY / 'initial-answer.json', record['parsed'])
    print(json.dumps({'status': record['status'], 'engine': record['engine'],
                      'requested_model': record['requested_model'], 'elapsed_seconds': record['elapsed_seconds']}, ensure_ascii=False))


def revision():
    paths = ['knowledge/workloads/metric-events.md', 'knowledge/workloads/measurement-comparability.md',
             'knowledge/tools/statistical-experiments.md', 'knowledge/workloads/slo-goodput.md']
    materials = [{'path': name, 'sha256': hashlib.sha256((KB / name).read_bytes()).hexdigest(),
                  'content': (KB / name).read_text(encoding='utf-8-sig'), 'reading_scope': 'full'} for name in paths]
    feedback = json.loads((DIRECTORY / 'feedback.json').read_text(encoding='utf-8'))
    initial = json.loads((DIRECTORY / 'initial-answer.json').read_text(encoding='utf-8'))
    save(DIRECTORY / 'revision-materials.json', {'files': materials,
        'scope': 'Current repaired chapter prose plus explicit feedback; no check code or semantic-review reports.'})
    prompt = {'task': 'A fresh response applying explicit feedback to a preserved prior answer. Keep the initial artifact unchanged. Solve the revision_task under the supplied current contracts. Return one strict JSON object with the requested fields. No tools or invented experiments.',
              'initial_answer': initial, 'feedback': feedback, 'materials': materials}
    record = ModelClient(CONFIG.adapters)('learner-revision', json.dumps(prompt, ensure_ascii=False),
        DIRECTORY / 'revision-call', engine='ds', effort=CONFIG.models['effort'], timeout=CONFIG.execution['model_timeout'])
    if record.get('status') != 'completed' or not isinstance(record.get('parsed'), dict):
        raise RuntimeError('Revision did not complete strict JSON; retain original records')
    save(DIRECTORY / 'revision-answer.json', record['parsed'])
    print(json.dumps({'status': record['status'], 'engine': record['engine'],
                      'requested_model': record['requested_model'], 'elapsed_seconds': record['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'revision':
        revision()
    else:
        main()
