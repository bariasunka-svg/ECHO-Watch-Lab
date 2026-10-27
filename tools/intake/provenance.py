# SPDX-License-Identifier: AGPL-3.0-only
"""Source provenance for EW-LAB-01: every Lab source/test/tool file with its origin and exact hashes.

Hashes of donor files come from the EXACT archives (vendor/donors/*.zip, verified against their pinned SHA-256), never
from a working tree or Git HEAD. For each Lab file: archive member + archive hash (when adopted), current hash, whether
it changed, and the recorded change note. Lab-new files are marked as such.

  env\\ui\\python.exe -I tools\\intake\\provenance.py --out docs\\SOURCE_PROVENANCE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE_ZIP = ROOT / "vendor/donors/echo-watch-native-0.2.0-foundation-3d14d6502c9e-source.zip"
NATIVE_SHA = "29332384727d47cb09386778f5e0e3e24c78a780390ac8f657b116daec759f97"
NATIVE_PREFIX = "echo-watch-native-0.2.0-foundation/"
PS_ZIP = ROOT / "vendor/donors/Patsight_PS27D_C3_reference_source.zip"
PS_SHA = "72efc34a1852bf9beced72edd26119d6f5f8bc2295efa5e35740c66f371703ea"

NOTES = {
    "src/echo_watch/config.py": "Lab paths: env/ui coordinator, env/gpu worker only, no CPU interpreter; data/operator; run evidence cap",
    "src/echo_watch/__init__.py": "Lab package description",
    "src/echo_watch/__main__.py": "manual launch entry -> app.launch.main (Native CLI not adopted)",
    "src/echo_watch/item_policy.py": "item-memory-v3: absolute admission gate on acquisition AND ownership, bounded unresolved hold, weak "
                                     "evidence never absence, low-confidence resembling proposals block absence, clone() for fork/commit, "
                                     "candidate rows for trace (defect P1 tablet-as-phone)",
    "src/echo_watch/matching.py": "DESCRIPTOR_VERSION, AdmissionGate (P/N/M rule), frozen calibration loader",
    "src/echo_watch/engine/evaluator.py": "fork()/adopt() isolated speculative state; candidates in Evaluation",
    "src/echo_watch/engine/session.py": "ambiguity carried across Resume (Q3 repair); defect A: fork evaluation, age recheck at commit under the final lock, version check; prior "
                                        "states/candidates/negative in trace; gate in trace+manifest; run-level item events; live_events; EW-LAB-05: review "
                                        "allowance parameter + saved-setup binding/metadata only",
    "src/echo_watch/engine/contracts.py": "ItemObservation.admission; projection carries admission/alternatives",
    "src/echo_watch/engine/board.py": "passes decision admission into observations",
    "src/echo_watch/engine/store.py": "corrections + item_names tables, activity indexes",
    "src/echo_watch/engine/library.py": "manual candidate (recovers a Scan miss), rename (append-only), profiles(); EW-LAB-05: enroll "
                                        "(new + explicit reuse, one idempotent command), geometry veto removed, compat-scoped hints/"
                                        "replacement, resolve_refs for Start",
    "src/echo_watch/engine/api.py": "GPU interpreter only; EW-LAB-05: Start boundary re-validates exact references and setup binding",
    "src/echo_watch/engine/render.py": "branding string",
    "src/echo_watch/guards.py": "branding strings",
    "src/echo_watch/metrics.py": "ResourceSampler gpu_fn (worker-reported VRAM in the UI process)",
    "src/echo_watch/worker/client.py": "last_vram from replies; EW-LAB-06 R1: per-attempt reply queue/stderr capture "
                                       "(bounded, truncation marked), structured failures on fatal/EOF/exit/deadline",
    "tests/test_item_policy.py": "fragmented-track move dx=120 -> -60 (old move placed the speaker under the watch box)",
    "tools/qualify/run_session.py": "Lab scenarios, admission child, corrected scorers, worker-VRAM guard",
    "tools/qualify/evaluate.py": "rewritten: defects B (endpoint/gaps/pauses/failed) and C (complete pairing evidence)",
    "src/echo_watch/app/display_schedule.py": "EW-LAB-02 A: latest-completed paired display with generation validity",
    "src/echo_watch/sources/fingerprint.py": "EW-LAB-02 B: content scope file-sha256, chunked cancellable hashing, stability",
    "tests/test_display_schedule.py": "EW-LAB-02 A regressions",
    "src/echo_watch/app/presentation.py": "EW-LAB-03 R3: one paired presentation for image, boxes, tray and inspector",
    "src/echo_watch/app/correction_flow.py": "EW-LAB-03 R1/R2: truthful save/retry/reconcile UI state, group description",
    "tests/test_paired_presentation.py": "EW-LAB-03 R3 regressions (real view methods, no Tk)",
    "tests/test_correction_truth.py": "EW-LAB-03 R1/R2 regressions",
    "tests/test_outcome_and_inspector.py": "EW-LAB-04 C1/C2 regressions (real renderers into non-widget sinks)",
    "tests/test_corrections_durable.py": "EW-LAB-02 B/C regressions",
    "tools/qualify/evaluate_native_v0.py": "PRESERVED adopted scorer (header comment only) for baseline/counterexamples",
    "src/echo_watch/engine/compat.py": "EW-LAB-05: the one enrollment compatibility predicate (content scope, size, model)",
    "src/echo_watch/engine/setups.py": "EW-LAB-05: versioned, validated saved-setup JSON artifacts",
    "src/echo_watch/app/views.py": "EW-LAB-05: Tk-free projections of the four-step Display Assets screens and layout policy",
    "src/echo_watch/app/controller.py": "EW-LAB-05: guided setup step model, enrollment command (new/reuse/replace), saved setups, "
                                        "exact Start binding, source-change invalidation, finish actions; EW-LAB-06: GPU owner "
                                        "check before Scan/Start and idle-worker handover before the worker starts",
    "src/echo_watch/app/tk_app.py": "EW-LAB-05: Discover / Select / Configure / Monitor workbench after the owner's references; "
                                    "Lab-03/04 paired renderers and Review/Activity/correction paths retained; EW-LAB-06: "
                                    "YOLOE rail entry/page, header Pause/Stop follow the active mode, exit stops both",
    "src/echo_watch/app/launch.py": "EW-LAB-06: builds the analysis owner and the YOLOE controller (no model, no process)",
    "tests/lab05_support.py": "EW-LAB-05 test support (synthetic stores, codec stubs, fake worker/session, real Start boundary)",
    "tests/lab05_view_harness.py": "EW-LAB-05 GUI-free harness for the production renderers (recording sinks, Tk guard)",
    "tests/test_lab05_enrollment.py": "EW-LAB-05 matrix 1-3: repeat-run enrollment, source isolation, immutable commands",
    "tests/test_lab05_setup_config.py": "EW-LAB-05 matrix 5: setup artifacts, exact binding, review allowance",
    "tests/test_lab05_views.py": "EW-LAB-05 matrix 4/7: four-step screens, gating, states, shared presentation, finish/Review",
    "tests/test_lab05_monitor_motion.py": "EW-LAB-05 matrix 6: moving supported observation, uncertainty, old-basis/stale",
    "src/echo_watch/yoloe/__init__.py": "EW-LAB-06: YOLOE prompt-free detection + BoT-SORT tracking workspace package",
    "src/echo_watch/yoloe/profile.py": "EW-LAB-06: the one frozen detector/tracker profile, flat tracker-config reader, pins",
    "src/echo_watch/yoloe/botsort.yaml": "EW-LAB-06: values copied from installed Ultralytics 8.4.160 cfg/trackers/botsort.yaml "
                                         "(SHA-256 8cd88930...; AGPL-3.0); ReID off, sparseOptFlow; provenance header",
    "src/echo_watch/yoloe/association.py": "EW-LAB-06: immutable detections, join by tracker-returned index with checks",
    "src/echo_watch/yoloe/tracker.py": "EW-LAB-06: adapter over the INSTALLED BOTSORT (snapshot/restore, tentative tracks)",
    "src/echo_watch/yoloe/worker.py": "EW-LAB-06: PF-only worker process, one predict + explicit tracker update per frame; "
                                      "protocol/guards pattern of worker/main.py (that file unchanged); R1: startup via "
                                      "bootstrap.start, quantize=None, RuntimeCapabilities effect boundary",
    "src/echo_watch/yoloe/bootstrap.py": "EW-LAB-06 R1: staged startup, structured fatal, requested/effective/observed receipt "
                                         "against the pinned Ultralytics 8.4.160 schema (quantize, not half)",
    "tests/pinned_ultralytics.py": "EW-LAB-06 R1: AST-extracted, hash-pinned installed Ultralytics config code for schema tests",
    "tests/test_lab06_r1_startup.py": "EW-LAB-06 R1: pinned-schema counterexample + production bootstrap with effects intercepted",
    "tests/test_lab06_r1_transport.py": "EW-LAB-06 R1: client diagnostic transport with the pipe boundary mocked",
    "tests/test_lab06_r1_failure_readback.py": "EW-LAB-06 R1: failure into outcome/manifest/trace/reopen/page renderers",
    "src/echo_watch/yoloe/client.py": "EW-LAB-06: WorkerClient subclass (command line + typed commands); client.py unchanged",
    "src/echo_watch/yoloe/source.py": "EW-LAB-06: ordered passthrough FFmpeg source with back-pressure, PTS mapping, media clock",
    "src/echo_watch/yoloe/session.py": "EW-LAB-06: YOLOE run driver, commit-time validity, settlement, evidence",
    "src/echo_watch/yoloe/evidence.py": "EW-LAB-06: atomic manifest + bounded background JSONL trace writer",
    "src/echo_watch/yoloe/review.py": "EW-LAB-06: saved-test listing, offset paging, source verification, exact-frame adapter",
    "src/echo_watch/app/resources.py": "EW-LAB-06: one application owner of GPU analysis (Display Assets or YOLOE)",
    "src/echo_watch/app/yoloe_controller.py": "EW-LAB-06: GUI-free YOLOE page state and commands",
    "src/echo_watch/app/yoloe_views.py": "EW-LAB-06: Tk-free YOLOE projections (labels, colours, counts, authority, layout)",
    "src/echo_watch/app/yoloe_page.py": "EW-LAB-06: YOLOE Tk page (fixed geometry, paired painting, results panel)",
    "requirements/gpu-tracker.lock": "EW-LAB-06: hash-locked lap 0.5.13 (owner-approved exact wheel)",
    "tools/provision/lap_wheel.py": "EW-LAB-06: bounded official fetch + offline inspection of the pinned lap wheel",
    "tools/lab06/tracker_synthetic_check.py": "EW-LAB-06: permitted synthetic CPU BoT-SORT check (GPU runtime, no model/CUDA)",
    "tools/lab06/run_tracker_check.py": "EW-LAB-06: windowless bounded launcher of the synthetic tracker check",
    "tests/lab06_support.py": "EW-LAB-06 test support (spy detector, scripted tracker, in-process worker core, scripted source)",
    "tests/test_lab06_detection_association.py": "EW-LAB-06 matrix 1-2 (UI runtime): association, detector call wiring, pins",
    "tests/test_lab06_source.py": "EW-LAB-06 matrix 3: ordered source, PTS/VFR/gaps, back-pressure, pacing",
    "tests/test_lab06_session.py": "EW-LAB-06 matrix 3/6: session lifecycle, settlement, epochs, evidence",
    "tests/test_lab06_owner_controller.py": "EW-LAB-06 matrix 4/6: single owner, handover, controller, results without inference",
    "tests/test_lab06_review.py": "EW-LAB-06 matrix 6: listing, paging, source verification, exact-frame selection",
    "tests/test_lab06_presentation.py": "EW-LAB-06 matrix 5: projections and production page renderers against sinks",
    "tests/test_import_closure.py": "EW-LAB-06: torch/ultralytics also confined to yoloe/worker.py and yoloe/tracker.py",
}
PS_ADOPTED = {
    "src/echo_watch/app/canvas_geometry.py": ("src/patronsight/app/canvas_geometry.py", "body unchanged; header added"),
    "src/echo_watch/review/frames.py": ("src/patronsight/app/review_player.py", "LatestWorker body unchanged; FFmpeg decoder new"),
    "src/echo_watch/app/tk_theme.py": ("src/patronsight/app/dashboard_theme.py", "token structure from the donor; EW-LAB-05 values "
                                       "follow the owner's Display Assets references; styles new"),
}
PS_PATTERNS = {
    "src/echo_watch/app/tk_app.py": ["src/patronsight/app/live_view.py", "src/patronsight/app/visitor_profile.py",
                                     "src/patronsight/app/visitor_history_page.py", "src/patronsight/app/session_review.py",
                                     "src/patronsight/app/settings_workspace.py"],
    "src/echo_watch/app/launch.py": ["src/patronsight/app/application.py"],
    "src/echo_watch/app/tasks.py": ["src/patronsight/app/review_player.py"],
    "src/echo_watch/review/corrections.py": ["src/patronsight/visitor_corrections.py"],
    "src/echo_watch/review/activity.py": ["src/patronsight/visitor_activity.py", "src/patronsight/app/visitor_history_page.py"],
    "src/echo_watch/review/runs.py": ["src/patronsight/app/session_review.py", "src/patronsight/app/session_review_state.py"],
    "src/echo_watch/review/export.py": ["src/patronsight/app/review_export.py"],
}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert sha_bytes(NATIVE_ZIP.read_bytes()) == NATIVE_SHA and sha_bytes(PS_ZIP.read_bytes()) == PS_SHA
    nz, pz = zipfile.ZipFile(NATIVE_ZIP), zipfile.ZipFile(PS_ZIP)
    native = {n[len(NATIVE_PREFIX):]: sha_bytes(nz.read(n)) for n in nz.namelist() if not n.endswith("/")}
    ps = {n: sha_bytes(pz.read(n)) for n in pz.namelist() if not n.endswith("/")}
    files = []
    for base in ("src", "tests", "tools", "requirements"):
        for p in sorted((ROOT / base).rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            rel = p.relative_to(ROOT).as_posix()
            cur = sha_bytes(p.read_bytes())
            rec = {"path": rel, "sha256": cur}
            if rel in native:
                rec.update(origin="native-archive", archive_member=NATIVE_PREFIX + rel, archive_sha256=native[rel],
                           changed=native[rel] != cur, note=NOTES.get(rel, "unchanged" if native[rel] == cur else "see diff"))
            elif rel in PS_ADOPTED:
                m, note = PS_ADOPTED[rel]
                rec.update(origin="patronsight-archive (selected code)", archive_member=m, archive_sha256=ps[m], changed=True, note=note)
            elif rel == "requirements/gpu-worker.lock":
                rec.update(origin="native-archive", archive_member=NATIVE_PREFIX + rel, archive_sha256=native[rel],
                           changed=native[rel] != cur, note="exact copy")
            else:
                rec.update(origin="lab-new", note=NOTES.get(rel, ""))
                if rel in PS_PATTERNS:
                    rec["patterns_from_patronsight_archive"] = {m: ps[m] for m in PS_PATTERNS[rel]}
            files.append(rec)
    doc = {"native_archive": {"path": str(NATIVE_ZIP.relative_to(ROOT)), "sha256": NATIVE_SHA,
                              "tree": "3d14d6502c9e31af5853e1b3aaea3c8e1d048edbe975d21f99962fd02cfee014"},
           "patronsight_archive": {"path": str(PS_ZIP.relative_to(ROOT)), "sha256": PS_SHA},
           "not_adopted_from_native": sorted(k for k in native if k.startswith(("src/", "tests/", "tools/"))
                                            and not (ROOT / k).exists()),
           "files": files,
           "src_tree_sha256": sha_bytes("".join(f"{r['path']}\0{r['sha256']}\n" for r in files
                                                if r["path"].startswith("src/") and r["path"].endswith(".py")).encode())}
    out = ROOT / a.out
    out.write_text(json.dumps(doc, indent=1), "utf-8")
    counts = {}
    for r in files:
        k = r["origin"] + (":changed" if r.get("changed") else (":unchanged" if "changed" in r else ""))
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps({"files": len(files), "by_origin": counts, "src_tree_sha256": doc["src_tree_sha256"],
                      "not_adopted": doc["not_adopted_from_native"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
