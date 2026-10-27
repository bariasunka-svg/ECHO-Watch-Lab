# SPDX-License-Identifier: AGPL-3.0-only
"""Render RESULTS.md from the per-session result.json / evaluation.json files (numbers are copied, never retyped)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLOSEOUT = ROOT / "closeouts" / "patsight-parity-foundation-20261025-193700"

# label -> (class, note). Classes: QUALIFICATION (the contract route), DIAGNOSTIC (real models, not the contract route),
# SELFTEST (stand-in worker replaying truth: harness validation only, never recognition evidence)
CLASSES = {
    "ST1-standin-scene3": ("SELFTEST", "stand-in worker replays truth; validates harness/evaluator/orchestration only"),
    "D1-cpu-gemini-one": ("DIAGNOSTIC", "CPU worker (8 torch threads: pre-pin build); real models"),
    "D2-cpu-scene-three": ("DIAGNOSTIC", "CPU worker (8 torch threads: pre-pin build); real models"),
    "D3-cpu-controls": ("DIAGNOSTIC", "CPU worker (8 torch threads: pre-pin build); real models"),
    "D4-cpu-scene-one": ("DIAGNOSTIC", "CPU worker, 4 torch threads; real models"),
    "D5-cpu-gemini-three": ("DIAGNOSTIC", "CPU worker, 4 torch threads; real models"),
    "F1-cpu-scene-three": ("DIAGNOSTIC", "FINAL source, CPU worker, 4 torch threads, explicit CPU backend (not the GPU contract route); real models"),
    "F2-cpu-scene-one": ("DIAGNOSTIC", "FINAL source, CPU worker, 4 torch threads, explicit CPU backend (not the GPU contract route); real models"),
    "F3-cpu-controls": ("DIAGNOSTIC", "FINAL source, CPU worker, 4 torch threads, explicit CPU backend (not the GPU contract route); real models"),
    "F4-cpu-gemini-three": ("DIAGNOSTIC", "FINAL source, CPU worker, 4 torch threads, explicit CPU backend (not the GPU contract route); real models"),
    "F5-cpu-gemini-one": ("DIAGNOSTIC", "FINAL source, CPU worker, 4 torch threads, explicit CPU backend (not the GPU contract route); real models"),
    "S1-gpu-scene-three": ("QUALIFICATION-REFUSED", "GPU admission refused (NVML free minimum below 2200 MiB); no model process started"),
    "S1r-gpu-scene-three": ("QUALIFICATION-FAILED", "GPU admission passed; the GPU worker died at import (builder defect: missing USERNAME in the child environment; repaired afterwards)"),
    "S1-gpu-scene-three-b": ("QUALIFICATION-REFUSED", "GPU admission refused again (NVML free minimum below 2200 MiB); no model process started"),
    "S1-gpu-scene-three-c": ("QUALIFICATION-REFUSED", "third GPU admission refusal; GPU work ended under the authorised terms; no model process started"),
    "S2-gpu-scene-three": ("QUALIFICATION", "GPU contract route; fresh 10 s admission passed; owner authorised this run (25 Oct 2026, after the S1 refusals)"),
    "S2-gpu-controls": ("QUALIFICATION", "GPU contract route; fresh 10 s admission passed; owner-authorised run"),
    "S2-gpu-gemini-three": ("QUALIFICATION", "GPU contract route; fresh 10 s admission passed; owner-authorised run (10 s clip, performance check, no truth scoring)"),
    "S2-gpu-scene-one": ("QUALIFICATION", "GPU contract route; fresh 10 s admission passed; owner-authorised run"),
    "S2-gpu-gemini-one": ("QUALIFICATION", "GPU contract route; fresh 10 s admission passed; owner-authorised run (10 s clip, performance check, no truth scoring)"),
}


def load(p: Path):
    return json.loads(p.read_text("utf-8")) if p.is_file() else None


def f(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def section(label: str, cls: str, note: str) -> str:
    d = CLOSEOUT / "sessions" / label
    res, ev, adm = load(d / "result.json"), load(d / "evaluation.json"), load(d / "admission.json")
    out = [f"### {label} - {cls}", f"_{note}_", ""]
    if res is None:
        return "\n".join(out + ["no result file", ""])
    oc = res.get("outcome") or {}
    out += [f"* status **{res.get('status')}**; scenario `{res.get('scenario')}`, items `{res.get('items_case')}`, device `{res.get('device')}`; "
            f"wall {f(res.get('total_wall_s'), 1)} s (worker process {f(res.get('worker_process_wall_s'), 1)} s; limit 240 s); outcome `{oc.get('kind')}` / "
            f"ended by `{oc.get('ended_by')}` / complete `{oc.get('complete')}`; committed results {oc.get('committed')}; discards {oc.get('discards')}",
            f"* worker start {f(res.get('worker_start_s'), 1)} s; scan {f((res.get('scan') or {}).get('duration_s'), 3)} s "
            f"({(res.get('scan') or {}).get('proposals')} proposals, infer {(res.get('scan') or {}).get('infer_s')} s); "
            f"requested -> active {f((ev or {}).get('throughput', {}).get('requested_to_active_s'), 2)} s; leftovers after shutdown: "
            f"{res.get('leftover_owned_processes_after_shutdown')}"]
    wr = res.get("worker_receipt") or {}
    if wr:
        out.append(f"* worker receipt: device `{wr.get('device')}` ({wr.get('gpu_name', 'cpu')}), torch `{wr.get('torch')}`, CUDA `{wr.get('cuda')}`, precision `{wr.get('precision')}`, "
                   f"threads {wr.get('threads')}/{wr.get('threads_requested', '?')} requested, models `{str(wr.get('model_sha256'))[:12]}` / `{str(wr.get('vp_model_sha256'))[:12]}`, load {wr.get('load_s')} s")
    if res.get("selection"):
        out.append("* selection among real proposals: " + "; ".join(
            f"{s['item']} -> `{s['label']}` conf {f(s['confidence'])} IoU-with-hint {f(s['iou_with_hint'])}" for s in res["selection"]))
    if adm:
        out.append(f"* admission: admitted **{adm['admitted']}**, mean CPU {adm['machine_cpu_mean']}%, p95 {adm['machine_cpu_p95']}%, NVML free min {adm['nvml_free_mib_min']} MiB of {adm['nvml_total_mib']}"
                   + (f" - refused: {adm['refusal_reasons']}" if not adm['admitted'] else ""))
    if ev:
        t = ev["throughput"]
        out += ["", "| throughput (steady window, first 5 s after active excluded) | value | criterion | |", "|---|---|---|---|",
                f"| distinct complete results / s (mean) | {f(t.get('mean_results_per_s'), 2)} ({t.get('steady_results')} in {f(t.get('steady_window_s'), 1)} s) | >= 8 | {'PASS' if t.get('criteria', {}).get('mean_ge_8') else 'FAIL'} |",
                f"| 1 s bins with >= 5 results | {t.get('bins_with_ge5')}/{t.get('bins')} = {f(t.get('fraction_bins_ge5'), 3)} (min bin {t.get('bin_counts_min')}) | >= 95% | {'PASS' if t.get('criteria', {}).get('bins_ge5_in_95pct') else 'FAIL'} |",
                f"| capture -> result p50 / p95 / max (ms) | {t.get('capture_to_result_ms_p50')} / {t.get('capture_to_result_ms_p95')} / {t.get('capture_to_result_ms_max')} | p95 <= 500 | {'PASS' if t.get('criteria', {}).get('p95_latency_le_500ms') else 'FAIL'} |",
                f"| max steady gap (s) | {t.get('max_steady_gap_s')} (long gaps: {len(t.get('long_steady_gaps') or [])}) | <= 1.0 | {'PASS' if t.get('criteria', {}).get('max_gap_le_1s') else 'FAIL'} |",
                f"| source ingress gaps > 250 ms | {t.get('source_ingress_gaps_count')} | reported | - |"]
        sv, sc = ev.get("scene_verdicts"), ev.get("scene_scoring")
        if sv:
            out += ["", "| item | window | eligible | Visible IoU>=.5 | wrong box | checking | uncertain | not visible | cannot assess | coverage | >=95% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in sv["coverage_rows"]:
                out.append(f"| {r['item']} | {r['window']}{' (moved interval, separate)' if r['moved_interval'] else ''} | {r['eligible']} | {r['visible_iou_ge_0.5']} | {r['visible_wrong_box']} | {r['checking']} | {r['uncertain']} | {r['not_seen']} | {r['cannot_assess']} | {f(r['coverage'], 4)} | {'PASS' if r['pass_ge_95'] else 'FAIL'} |")
            out += ["", f"* ordinary windows >= 95%: **{sv['ordinary_windows_coverage_ge_95']}**; zero wrong-target (Visible) **{sv['zero_wrong_target']}** ({len(sc['wrong_target'])}; tentative boxes on another item {len(sc.get('tentative_box_on_other_item', []))}); zero box on an absent target **{sv['zero_box_on_absent_target']}** ({len(sc['box_on_absent_target'])}); zero unsupported box **{sv['zero_unsupported_box']}**"]
            if "speaker_not_visible_latency_s" in sv:
                out.append(f"* speaker removed at 20.0 s: first Not visible after {sv['speaker_not_visible_latency_s']} s (within 3 s: **{sv['speaker_not_visible_within_3s']}**, negative evidence assessable: {sv['speaker_not_visible_negative_ok']}); returned at 35.0 s: Visible after {sv['speaker_visible_latency_s']} s (within 3 s: **{sv['speaker_visible_within_3s']}**)")
        ct = ev.get("controls")
        if ct:
            out += ["", f"* controls **{ct['status']}**: {json.dumps(ct['criteria'])}; speaker states while obscured {ct['state_counts']['speaker']['obscured']}; around pause {ct['state_counts']['speaker']['pause']}; "
                        f"phone_a states {ct['state_counts']['phone_a']['all']}; Visible wrong-target {ct['wrong_target_count']} results in {ct['wrong_target_intervals']}; "
                        f"box while phone_a absent {ct['box_on_absent_target_count']} results in {ct['box_on_absent_target_intervals']}; tentative boxes on another item {ct['tentative_box_on_other_item_count']}"]
        p = ev["pairing"]
        out.append(f"* frame/result pairing against the source file: checked {p['checked']}, mismatched **{p['mismatched']}** ({p['status']})")
        rec = ev["recordings"]
        for k in ("clean", "annotated"):
            r = rec.get(k) or {}
            if r.get("frames_decoded") is not None:
                extra = f"; fresh {r.get('fresh')} unique {r.get('fresh_results_unique')}, held {r.get('held')}, committed results never shown {r.get('results_never_shown')}" if k == "annotated" else ""
                out.append(f"* {k}: {r['frames_decoded']} frames decoded = {r['frames_mapped']} mapped (count match {r['count_match']}), content match **{r['content_match']}** (mean |diff| {r['mean_abs_diff']}), queue drops {r.get('frames_dropped_queue')}, capture gaps {r.get('gaps')} (largest {r.get('largest_gap_s')} s){extra}")
        tr = rec.get("trace") or {}
        out.append(f"* decision trace: {tr.get('state')} - {tr.get('records')} records, {tr.get('dropped')} dropped")
        rs_ = ev.get("recording_stage") or {}
        if rs_.get("samples"):
            def rr(k):
                v = rs_.get(k)
                return "-" if not v else f"{v['mean']} (p05 {v['p05']})"
            pm = rs_.get("pump") or {}
            out.append(f"* distinct rates (1 Hz metrics, active phase, {rs_['samples']} samples): capture {rr('capture_fps')} /s; analysed {rr('analyzed_fps')} /s; "
                       f"clean encode {rr('clean_encode_fps')} /s; annotated encode {rr('annotated_encode_fps')} /s; frames replaced before analysis {rs_.get('replaced_before_analysis')}; "
                       f"annotated render p95 {f(pm.get('render_ms_p95'))} ms; pump fresh {pm.get('fresh_results_rendered')} / held {pm.get('held_frames_offered')} / skipped overdue ticks {pm.get('skipped_overdue_ticks')}")
        sc_ = ev.get("stage_costs_ms") or {}
        keys = ["queue_wait", "worker_round_trip", "worker_infer_ms", "worker_post_ms", "worker_ipc_out_ms", "worker_ipc_back_ms", "evaluate", "eval_quality",
                "eval_assessability_geometry", "eval_matching", "eval_context_features", "handoff_wait", "commit"]
        out += ["", "| stage (ms) | p50 | p95 | max |", "|---|---|---|---|"] + [f"| {k} | {sc_[k]['p50']} | {sc_[k]['p95']} | {sc_[k]['max']} |" for k in keys if k in sc_]
        rs = ev.get("resource") or {}
        if rs.get("samples"):
            def s(k, key="mean"):
                return f((rs.get(k) or {}).get(key), 1)
            out += ["", f"* resources (1 Hz, {rs['samples']} samples, guard aborted: {rs.get('aborted')}): machine CPU mean {s('machine_cpu')} / p95 {s('machine_cpu', 'p95')} / max {s('machine_cpu', 'max')} %; "
                        f"owned family CPU mean {s('family_cpu')} % (sum of cores), RSS mean {s('family_rss_mib')} / max {s('family_rss_mib', 'max')} MiB; "
                        f"GPU used mean {s('gpu_used_mib', 'mean')} MiB, min free {s('gpu_free_mib', 'min')} MiB, util mean {s('gpu_util')} %"]
    return "\n".join(out + [""])


def main() -> int:
    labels = sys.argv[1:] or list(CLASSES)
    parts = ["# Session results (generated by tools/qualify/report.py from result.json / evaluation.json)", ""]
    for lab in labels:
        cls, note = CLASSES.get(lab, ("QUALIFICATION", ""))
        parts.append(section(lab, cls, note))
    (CLOSEOUT / "RESULTS.md").write_text("\n".join(parts), "utf-8")
    print(f"wrote {CLOSEOUT / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
