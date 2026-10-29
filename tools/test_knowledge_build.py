"""Isolated behavioral forward tests of the authoring workflow.

Temporary fixtures exercise real subprocess checks. Release is stubbed only in
tests of the coordinator's pre-release guards; release behavior has its own suite.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import knowledge_build as workflow

CHAPTER = "knowledge/algorithms/first.md"
SECOND = "knowledge/algorithms/second.md"
SOURCE = "knowledge/sources/S-ONE.md"
FIRST = "K-ALG-001"
NEXT = "K-ALG-002"
QUOTE = "The published prefix remains immutable until its last reader completes."


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="knowledge-workflow-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "kb"
        self.root.mkdir()
        self.run = self.root / "docs/evidence/batch-one"
        self.put(CHAPTER, "# K-ALG-001: First\n\n" + QUOTE + "\n")
        self.put(SECOND, "# K-ALG-002: Second\n\nA different chapter is still waiting for substantive review.\n")
        self.put(SOURCE, "# S-ONE\nA local prose source about publication and readers.\n")
        self.put("tools/check_fixture.py", "assert sum(range(5)) == 10\nprint('executed arithmetic reference: 10')\n")
        workflow.write(self.root / "knowledge/catalog.json", {"entries": [
            {"id": FIRST, "path": "algorithms/first.md", "revision": 2, "sha256": workflow.digest(self.root / CHAPTER),
             "source_ids": ["S-ONE"], "related": [NEXT]},
            {"id": NEXT, "path": "algorithms/second.md", "revision": 1, "sha256": workflow.digest(self.root / SECOND),
             "source_ids": [], "related": [FIRST]}]})
        workflow.write(self.root / "knowledge/sources.json", {"sources": [
            {"id": "S-ONE", "local_path": SOURCE, "local_sha256": workflow.digest(self.root / SOURCE)}]})
        workflow.write(self.root / "knowledge/temporal-policy.json", {"profile_id": "fixture-v1"})
        workflow.write(self.root / "VERSION.json", {"version_id": "fixture-v1"})
        workflow.write(self.root / "docs/expansion-status.json", {"entries": [
            {"id": FIRST, "review": "navigation_registered_deep_review_pending"},
            {"id": NEXT, "review": "focused_review_with_limits"}]})

    def put(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def state(self):
        return workflow.read(self.run / "workflow.json")

    def job(self):
        return self.state()["jobs"][FIRST]

    def start(self, max_rounds=3):
        workflow.initialize(self.root, self.run, max_rounds=max_rounds)
        workflow.claim(self.run, "/author", entry_id=FIRST)

    def review(self, decision="accept", **updates):
        criteria = {name: {"assessment": "sufficient", "reason": "The fixture declaration establishes this scoped condition.",
                           "evidence": [{"path": CHAPTER, "quote": QUOTE}]}
                    for name in workflow.QUESTIONS}
        result = {"entry_id": FIRST, "body_sha256": workflow.digest(self.root / CHAPTER),
                  "reviewer": "/reviewer", "decision": decision,
                  "documents": {CHAPTER: workflow.digest(self.root / CHAPTER)},
                  "criteria": criteria, "findings": [], "limitations": ["Fixture review does not establish technical content quality."],
                  "checks": [{"script": "tools/check_fixture.py", "args": []}]}
        result.update(updates)
        return result

    def submit(self, review=None, name="review-input.json"):
        path = self.root / name
        workflow.write(path, self.review() if review is None else review)
        return workflow.record_review(self.root, self.run, FIRST, path)

    def accept_and_check(self):
        self.submit()
        return workflow.execute_checks(self.root, self.run, FIRST)

    def test_initialize_preserves_exact_baseline_and_pending_is_not_done(self):
        result = workflow.initialize(self.root, self.run)
        self.assertEqual(result["counts"], {"queued": 2})
        self.assertFalse(result["all_content_accepted"])
        snapshot = workflow.read(self.run / "before/snapshot.json")
        self.assertEqual((self.run / "before" / CHAPTER).read_bytes(), (self.root / CHAPTER).read_bytes())
        self.assertEqual(snapshot[CHAPTER], workflow.digest(self.root / CHAPTER))
        self.assertEqual(self.state()["jobs"][FIRST]["baseline_revision"], 2)

    def test_pending_selection_uses_existing_deep_review_accounting(self):
        result = workflow.initialize(self.root, self.run, "pending")
        self.assertEqual(result["total"], 1)
        self.assertEqual(set(self.state()["jobs"]), {FIRST})

    def test_unregistered_chapter_or_source_changes_refuse_snapshot(self):
        self.put(CHAPTER, (self.root / CHAPTER).read_text(encoding="utf-8") + "unregistered\n")
        with self.assertRaisesRegex(ValueError, "Unregistered chapter"):
            workflow.initialize(self.root, self.run)
        self.assertFalse((self.run / "before").exists())
        catalog = workflow.read(self.root / "knowledge/catalog.json")
        catalog["entries"][0]["sha256"] = workflow.digest(self.root / CHAPTER)
        workflow.write(self.root / "knowledge/catalog.json", catalog)
        self.put(SOURCE, "An unregistered source-note modification.\n")
        with self.assertRaisesRegex(ValueError, "Unregistered source-note"):
            workflow.initialize(self.root, self.run)
        self.assertFalse((self.run / "before").exists())

    def test_reinitialization_never_overwrites_original_baseline(self):
        workflow.initialize(self.root, self.run)
        before = workflow.digest(self.run / "before/snapshot.json")
        with self.assertRaisesRegex(ValueError, "already exists"):
            workflow.initialize(self.root, self.run)
        self.assertEqual(workflow.digest(self.run / "before/snapshot.json"), before)

    def test_claim_assigns_author_without_reclaiming_working_job(self):
        self.start()
        self.assertEqual(self.job()["round"], 1)
        self.assertEqual(self.job()["author"], "/author")
        self.assertEqual(workflow.claim(self.run, "/other", entry_id=FIRST)["claimed"], [])
        self.assertEqual(self.job()["author"], "/author")
        prompt = workflow.make_prompt(self.root, self.run, FIRST, "author")
        self.assertIn(SOURCE, prompt)
        self.assertIn(NEXT, prompt)

    def test_review_rejects_same_author_and_unbound_chapter(self):
        self.start()
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.submit(self.review(reviewer="/author"))
        with self.assertRaisesRegex(ValueError, "current chapter bytes"):
            self.submit(self.review(body_sha256="0" * 64))
        self.assertEqual(self.job()["state"], "working")
        self.assertEqual(self.job()["reviews"], [])

    def test_review_requires_six_actual_scoped_assessments(self):
        self.start()
        result = self.review()
        del result["criteria"]["cost"]
        with self.assertRaisesRegex(ValueError, "six criteria"):
            self.submit(result)
        result = self.review()
        result["criteria"]["cost"]["reason"] = ""
        with self.assertRaisesRegex(ValueError, "Missing scoped"):
            self.submit(result)

    def test_review_rejects_fabricated_quote_and_stale_source_read(self):
        self.start()
        result = self.review()
        result["criteria"]["algorithm"]["evidence"][0]["quote"] = "This invented claim never appears in the reviewed chapter."
        with self.assertRaisesRegex(ValueError, "Evidence is absent"):
            self.submit(result)
        result = self.review()
        result["documents"][SOURCE] = workflow.digest(self.root / SOURCE)
        self.put(SOURCE, "A revised source note makes the old reading stale.\n")
        with self.assertRaisesRegex(ValueError, "Stale reviewed file"):
            self.submit(result)

    def test_partial_or_blocking_findings_cannot_be_accepted(self):
        self.start()
        result = self.review()
        result["criteria"]["cost"]["assessment"] = "partial"
        with self.assertRaisesRegex(ValueError, "Unresolved content"):
            self.submit(result)
        with self.assertRaisesRegex(ValueError, "Blocking findings"):
            self.submit(self.review(findings=[{"id": "missing-cost", "blocking": True}]))

    def test_empty_checks_require_specific_reason_but_do_not_invent_execution(self):
        self.start()
        with self.assertRaisesRegex(ValueError, "meaningful checks"):
            self.submit(self.review(checks=[]))
        self.submit(self.review(checks=[], no_checks_reason="This fixture is a prose-only identity explanation."))
        result = workflow.execute_checks(self.root, self.run, FIRST)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["checks"], [])

    def test_actual_check_execution_retains_stdout_exit_and_bound_script(self):
        self.start()
        result = self.accept_and_check()
        self.assertEqual(result["state"], "ready")
        record = workflow.read(self.root / result["checks"][0]["path"])
        self.assertEqual(record["exit_code"], 0)
        self.assertIn("executed arithmetic reference: 10", record["stdout"])
        self.assertEqual(record["script_sha256"], workflow.digest(self.root / "tools/check_fixture.py"))
        self.assertFalse(workflow.summary(self.state())["all_content_accepted"])

    def test_injected_runner_records_real_backend_and_script_identity(self):
        self.start()
        self.submit()
        observed = []

        def isolated_runner(root, script, args, output_dir, *, timeout):
            observed.append((root, script, args, output_dir, timeout))
            return {"returncode": 0, "stdout": "isolated reference passed", "stderr": "",
                    "execution": {"backend": "fixture_isolated_runner", "actual_argv": ["isolated-python", script],
                                  "script_sha256": workflow.digest(root / script)}}

        with patch.object(workflow.subprocess, "run", side_effect=AssertionError("local execution is forbidden")):
            result = workflow.execute_checks(self.root, self.run, FIRST, runner=isolated_runner)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(observed[0][1], "tools/check_fixture.py")
        log = workflow.read(self.root / result["checks"][0]["path"])
        self.assertEqual(log["script_path"], "tools/check_fixture.py")
        self.assertEqual(log["execution"]["backend"], "fixture_isolated_runner")
        self.assertEqual(log["stdout"], "isolated reference passed")

    def test_injected_runner_cannot_substitute_a_different_script(self):
        self.start()
        self.submit()

        def wrong_runner(*args, **kwargs):
            return {"returncode": 0, "stdout": "PASS", "stderr": "",
                    "execution": {"backend": "fixture", "script_sha256": "0" * 64}}

        with self.assertRaisesRegex(ValueError, "does not match"):
            workflow.execute_checks(self.root, self.run, FIRST, runner=wrong_runner)
        self.assertEqual(self.job()["state"], "reviewed")
        self.assertEqual(self.job()["checks"], [])

    def test_cli_completes_review_and_check_without_treating_remaining_queue_as_done(self):
        command = [sys.executable, "-X", "utf8", str(Path(workflow.__file__).resolve()),
                   "--root", str(self.root), "--run", "batch-one"]

        def invoke(*arguments):
            result = subprocess.run(command + list(arguments), capture_output=True,
                                    text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

        self.assertEqual(invoke("init")["counts"], {"queued": 2})
        self.assertEqual(invoke("next", "--owner", "/author", "--entry", FIRST)["claimed"], [FIRST])
        workflow.write(self.root / "cli-review.json", self.review())
        self.assertEqual(invoke("review", "--entry", FIRST, "--file", "cli-review.json")["state"], "reviewed")
        self.assertEqual(invoke("check", "--entry", FIRST)["state"], "ready")
        status = invoke("status")
        self.assertEqual(status["counts"], {"ready": 1, "queued": 1})
        self.assertFalse(status["all_content_accepted"])

    def test_failing_check_enters_revision_and_preserves_failed_round(self):
        self.start()
        self.put("tools/check_fixture.py", "print('induced failure before repair')\nraise SystemExit(7)\n")
        self.submit()
        first = workflow.execute_checks(self.root, self.run, FIRST)
        self.assertEqual(first["state"], "checks_failed")
        log_path = self.root / first["checks"][0]["path"]
        before = log_path.read_bytes()
        self.assertEqual(workflow.read(log_path)["exit_code"], 7)
        workflow.claim(self.run, "/author", entry_id=FIRST)
        self.assertEqual(self.job()["round"], 2)
        self.put("tools/check_fixture.py", "assert 2 + 2 == 4\nprint('repaired reference')\n")
        self.submit(name="second-review.json")
        second = workflow.execute_checks(self.root, self.run, FIRST)
        self.assertEqual(second["state"], "ready")
        self.assertEqual(log_path.read_bytes(), before)
        self.assertEqual(len(self.job()["reviews"]), 2)
        self.assertNotEqual(first["checks"][0]["path"], second["checks"][0]["path"])

    def test_revise_decision_and_reopen_keep_previous_review_bytes(self):
        self.start()
        first_review = self.review("revise", findings=[{"id": "cost-1", "blocking": True, "criterion": "cost"}])
        self.submit(first_review)
        first_path = self.root / self.job()["reviews"][-1]["path"]
        before = first_path.read_bytes()
        workflow.claim(self.run, "/author", entry_id=FIRST)
        self.accept_and_check()
        workflow.reopen(self.run, FIRST, "A newly checked assumption invalidated the previous candidate.")
        self.assertEqual(self.job()["state"], "revise")
        self.assertEqual(first_path.read_bytes(), before)
        self.assertEqual(self.job()["revision_reasons"][-1]["round"], 2)

    def test_review_overwrite_is_refused_without_new_round(self):
        self.start()
        self.submit()
        original = (self.root / self.job()["reviews"][-1]["path"]).read_bytes()
        with self.assertRaisesRegex(ValueError, "already retained"):
            self.submit(self.review(limitations=["A later reviewer changed the conclusion."]), name="different-review.json")
        self.assertEqual((self.root / self.job()["reviews"][-1]["path"]).read_bytes(), original)

    def test_failed_round_cannot_be_reset_by_resubmitting_identical_review(self):
        self.start(max_rounds=1)
        self.put("tools/check_fixture.py", "raise SystemExit(9)\n")
        review = self.review()
        self.submit(review)
        failed = workflow.execute_checks(self.root, self.run, FIRST)
        log = self.root / failed["checks"][0]["path"]
        original = log.read_bytes()
        try:
            self.submit(review, name="same-review-again.json")
        except ValueError:
            pass
        self.assertEqual(self.job()["state"], "checks_failed")
        self.assertEqual(log.read_bytes(), original)

    def test_chapter_change_after_review_invalidates_check(self):
        self.start()
        self.submit()
        self.put(CHAPTER, (self.root / CHAPTER).read_text(encoding="utf-8") + "A changed algorithm.\n")
        with self.assertRaisesRegex(ValueError, "current chapter bytes"):
            workflow.execute_checks(self.root, self.run, FIRST)
        self.assertNotEqual(self.job()["state"], "ready")

    def test_changed_check_script_prevents_registration(self):
        self.start()
        self.accept_and_check()
        self.put("tools/check_fixture.py", "raise AssertionError('changed after successful execution')\n")
        stub = types.SimpleNamespace(release=lambda *args, **kwargs: self.fail("release guard was bypassed"))
        with patch.dict(sys.modules, {"knowledge_release": stub}):
            with self.assertRaisesRegex(ValueError, "script changed after"):
                workflow.register(self.root, self.run, "fixture-v2")

    def test_self_modifying_check_is_not_recorded_as_success(self):
        self.start()
        self.put("tools/check_fixture.py", "from pathlib import Path\np=Path(__file__)\np.write_text('# changed during check\\n', encoding='utf-8')\n")
        self.submit()
        result = workflow.execute_checks(self.root, self.run, FIRST)
        self.assertEqual(result["state"], "checks_failed")
        record = workflow.read(self.root / result["checks"][0]["path"])
        self.assertEqual(record["exit_code"], -2)
        self.assertIn("changed during", record["stderr"])

    def test_timeout_is_real_failure_with_retained_diagnostic(self):
        self.start()
        self.put("tools/check_fixture.py", "import time\ntime.sleep(2)\n")
        self.submit()
        result = workflow.execute_checks(self.root, self.run, FIRST, timeout=0.02)
        self.assertEqual(result["state"], "checks_failed")
        record = workflow.read(self.root / result["checks"][0]["path"])
        self.assertEqual(record["exit_code"], -1)
        self.assertIn("timed out", record["stderr"])

    def test_path_escape_is_rejected_before_snapshot_or_check(self):
        with self.assertRaisesRegex(ValueError, "outside knowledge"):
            workflow.inside(self.root, "../outside.json")
        with self.assertRaisesRegex(ValueError, "under docs/evidence"):
            workflow.initialize(self.root, self.root.parent / "outside-run")
        self.start()
        self.submit(self.review(checks=[{"script": "../outside.py", "args": []}]))
        with self.assertRaisesRegex(ValueError, "outside knowledge"):
            workflow.execute_checks(self.root, self.run, FIRST)

    def test_round_budget_stops_one_job_and_other_queue_work_remains_claimable(self):
        self.start(max_rounds=1)
        workflow.reopen(self.run, FIRST, "Unresolved mathematical condition needs more evidence.")
        outcome = workflow.claim(self.run, "/other", limit=1)
        self.assertEqual(outcome["claimed"], [NEXT])
        self.assertEqual(self.job()["state"], "needs_attention")
        self.assertEqual(outcome["counts"], {"needs_attention": 1, "working": 1})
        self.assertFalse(outcome["all_content_accepted"])

    def test_registered_is_not_verified_and_cannot_advance_baseline(self):
        self.start()
        self.accept_and_check()
        stub = types.SimpleNamespace(release=lambda *args, **kwargs: {
            "version_id": "fixture-v2", "selected_ids": [FIRST], "revised_ids": [],
            "source_note_changes": [], "already_registered": False})
        with patch.dict(sys.modules, {"knowledge_release": stub}):
            workflow.register(self.root, self.run, "fixture-v2")
        self.assertEqual(self.job()["state"], "registered")
        self.assertFalse(workflow.summary(self.state())["all_content_accepted"])
        with self.assertRaisesRegex(ValueError, "active jobs"):
            workflow.continue_run(self.root, self.run, "batch-two")

    def test_continue_preserves_remaining_revision_state_and_reports_actual_counts(self):
        self.start()
        state = self.state()
        state["registered_version"] = "fixture-v2"
        state["jobs"][FIRST]["state"] = "verified"
        state["jobs"][NEXT].update(state="revise", round=2, issues=[{"id": "state-gap", "blocking": True}])
        workflow.write(self.run / "workflow.json", state)
        workflow.write(self.root / "VERSION.json", {"version_id": "fixture-v2"})
        result = workflow.continue_run(self.root, self.run, "batch-two")
        successor = workflow.read(self.root / "docs/evidence/batch-two/workflow.json")
        self.assertEqual(set(successor["jobs"]), {NEXT})
        self.assertEqual(successor["jobs"][NEXT]["round"], 2)
        self.assertEqual(successor["jobs"][NEXT]["issues"], state["jobs"][NEXT]["issues"])
        self.assertEqual(result["counts"], workflow.summary(successor)["counts"])
        self.assertEqual(self.state()["successor_run"], "docs/evidence/batch-two")

    def test_exercise_requires_current_separately_authored_bound_artifacts(self):
        self.start()
        documents = {CHAPTER: workflow.digest(self.root / CHAPTER)}
        artifact = {"entry_id": FIRST, "body_sha256": documents[CHAPTER], "participant": "/learner",
                    "limitations": ["Synthetic isolated fixture, not an Agent ability study."],
                    "observed_outcome": "The fixture records a bounded exercise, not a causal benefit."}
        for key in ["task", "initial_answer", "feedback", "revision"]:
            name = "docs/evidence/batch-one/exercise/" + key + ".md"
            self.put(name, "Retained " + key + " artifact.\n")
            artifact[key] = name
            documents[name] = workflow.digest(self.root / name)
        artifact["documents"] = documents
        path = self.root / "exercise.json"
        self_author = deepcopy(artifact)
        self_author["participant"] = "/author"
        workflow.write(path, self_author)
        with self.assertRaisesRegex(ValueError, "differ from the author"):
            workflow.record_exercise(self.root, self.run, FIRST, path)
        workflow.write(path, artifact)
        workflow.record_exercise(self.root, self.run, FIRST, path)
        self.assertEqual(self.job()["agent_use"], "bounded_exercise_recorded")
        self.put(artifact["initial_answer"], "Changed after feedback.\n")
        with self.assertRaisesRegex(ValueError, "Stale exercise artifact"):
            workflow.record_exercise(self.root, self.run, FIRST, path)


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({"test": test.id(), "result": "PASS"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.outcomes.append({"test": test.id(), "result": "FAIL", "detail": self._exc_info_to_string(error, test)})

    def addError(self, test, error):
        super().addError(test, error)
        self.outcomes.append({"test": test.id(), "result": "ERROR", "detail": self._exc_info_to_string(error, test)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WorkflowTests)
    outcome = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    if args.report:
        tool = Path(workflow.__file__).resolve()
        report = {"result": "PASS" if outcome.wasSuccessful() else "FAIL", "tests_run": outcome.testsRun,
                  "failures": len(outcome.failures), "errors": len(outcome.errors), "tests": outcome.outcomes,
                  "workflow_sha256": workflow.digest(tool), "test_sha256": workflow.digest(Path(__file__)),
                  "implementation_bindings": {"tools/knowledge_build.py": workflow.digest(tool),
                                              "tools/knowledge_release.py": workflow.digest(tool.with_name("knowledge_release.py"))},
                  "scope": "Isolated tempfile behavioral tests of queue, reviews, real subprocess checks, guards and continuation; release implementation is not assessed.",
                  "limits": ["Technical review fields are deliberate fixtures and do not certify chapter semantics.",
                             "No target GPU, model invocation, real queue mutation, or causal knowledge-benefit experiment.",
                             "Tester inherited parent conversation and could inspect implementation; this is not blind evaluation."]}
        workflow.write(args.report, report)
    raise SystemExit(0 if outcome.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
