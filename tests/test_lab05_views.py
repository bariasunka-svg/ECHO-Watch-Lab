# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-05 matrix 4 and 7: the four-step screens, their gating, states and the shared presentation (GUI-free).

A real controller (synthetic store, fake worker/session) drives app.views projections and the PRODUCTION renderer
methods of tk_app.WorkbenchWindow, extracted by AST and executed against recording sinks (lab05_view_harness). No Tk,
Tcl, widget, root, image codec or process is created. Rendered appearance and real-video behaviour are not covered.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from echo_watch.app import views as V
from echo_watch.app.canvas_geometry import source_box_to_canvas
from lab05_support import Rig, settle
from lab05_view_harness import build


@pytest.fixture()
def rig(monkeypatch):
    r = Rig("lab05-views", monkeypatch)
    yield r
    r.close()


def ev(x, y, shift=False):
    return types.SimpleNamespace(x=x, y=y, state=0x0001 if shift else 0)


# ---- step model ---------------------------------------------------------------------------------------------------
def test_step_gating_back_continue_and_navigation_never_starts_or_stops(rig):
    c = rig.ctl
    assert c.view_step == "discover" and not c.go("select") and not c.go("configure")
    bar = V.action_bar(c)
    assert bar["primary"][:2] == ("continue", "Continue to selection") and bar["primary"][2] is False
    assert bar["hint"] == "Available when discovery finishes" and bar["back"] is None
    rig.discover(rig.a)                                          # rig.discover continues to Select
    assert [s["state"] for s in c.steps()] == ["complete", "current", "upcoming", "upcoming"]
    assert not c.go("configure") and "Remember items in Select first" in c.messages[-1].text
    assert c.back() and c.view_step == "discover" and c.candidates          # Back preserves the scan
    assert c.go("select")
    c.toggle_candidate(rig.cid("phone"), True)
    rig.remember()
    assert c.view_step == "configure" and c.back() and c.view_step == "select"
    assert c.candidate(rig.cid("phone")).checked                 # Back preserves the selection
    assert c.remember() and c.view_step == "configure"           # unchanged confirmed selection: no new operation
    rig.start()
    phase = c.phase
    for step in ("discover", "select", "configure"):
        assert not c.go(step) and c.phase == phase              # navigation cannot stop monitoring...
    bar = V.action_bar(c)
    assert bar["primary"][0] == "stop" and bar["primary"][2] and bar["secondary"][0][0] == "pause"
    h = V.header(c)
    assert h["show_controls"] and h["session"] == ("Monitoring - video replay", "success")
    assert "camera" not in repr(h).lower()


# ---- Discover -----------------------------------------------------------------------------------------------------
def test_discover_loading_failure_retry_and_empty_states_through_the_renderer(rig):
    c = rig.ctl
    v, _ns, _clk = build(c)
    d = V.discover(c)
    assert d["primary"] == ("Discover items", False) and "never starts monitoring" in d["status"]
    assert c.choose_source(str(rig.a))
    v._render_discover()
    assert v.prog.calls and v.prog.calls[0][0] == "start" and "Preparing the video" in v.prog_lbl.cget("text")
    assert settle(c, lambda: c.source_status == "ready")
    rig.eng.worker.fail = "CUDA out of memory"
    assert c.scan() and settle(c, lambda: c.scan_status == "failed")
    v._render_discover()
    assert "Discovery failed: RuntimeError: CUDA out of memory" in v.d_status.cget("text")
    assert v.d_scan.cget("text") == "Discover again" and v.d_scan.cget("state") == "normal"   # retry stays in place
    assert c.setup_frame is not None                            # the setup frame is kept for the retry
    rig.eng.worker.fail, rig.eng.worker.candidates = None, []
    assert c.scan() and settle(c, lambda: c.scan_status == "done")
    d = V.discover(c)
    assert d["empty"] and d["can_continue"] and "found no candidates" in d["status"]
    assert c.go("select")                                       # an empty scan still permits manual recovery
    assert c.add_manual_region((20, 20, 60, 60)) and settle(c, lambda: len(c.candidates) == 1)
    assert c.candidates[0].manual and c.candidates[0].checked and c.candidates[0].name == "Item 01"


# ---- Select -------------------------------------------------------------------------------------------------------
def test_click_toggles_the_smallest_outline_focus_is_separate_and_list_checkbox_agree(rig):
    c = rig.ctl
    rig.discover(rig.a)
    v, _ns, _clk = build(c)
    phone, table = rig.cid("phone"), rig.cid("table")
    assert V.pick_at([(f"cand:{x.candidate_id}", x.box) for x in c.candidates], (120, 150))[:2] == \
        [f"cand:{phone}", f"cand:{table}"]                      # a table-sized outline never hides the phone
    v._canvas_press(ev(120, 150))
    assert c.candidate(phone).checked and not c.candidate(table).checked and c.selected == f"cand:{phone}"
    assert "2 outlines overlap here" in c.messages[-1].text
    v._canvas_press(ev(120, 150, shift=True))                   # Shift+click cycles focus, never toggles
    assert c.selected == f"cand:{table}" and not c.candidate(table).checked
    v._toggle_focused()                                         # Space on the focused suggestion
    assert c.candidate(table).checked
    v._render_select()
    rows = v.s_list.rows
    assert rows[f"cand:{phone}"]["values"][0] == "☑" and rows[f"cand:{table}"]["values"][0] == "☑"
    assert rows[f"cand:{rig.cid('mug')}"]["values"][0] == "☐"
    assert v.s_count.cget("text") == "2 selected · 3 suggestions" and v.s_list.sel == (f"cand:{table}",)
    i_mug = v.s_list.order.index(f"cand:{rig.cid('mug')}")
    assert v._list_click(v.s_list, ev(200, 10 + 20 * i_mug)) is None    # name column: the row's own selection...
    v.s_list.selection_set(f"cand:{rig.cid('mug')}")
    v._list_focus(v.s_list)                                     # ...focuses it (<<TreeviewSelect>>) without toggling
    assert c.selected == f"cand:{rig.cid('mug')}" and not c.candidate(rig.cid("mug")).checked
    v._list_click(v.s_list, ev(10, 10 + 20 * i_mug))            # checkbox column: toggle
    assert c.candidate(rig.cid("mug")).checked
    v._render_action_bar()
    assert v.ab_primary.cget("text") == "Remember & configure 3 items" and v.ab_primary.cget("state") == "normal"
    v._overlay_key = None
    v._painted = types.SimpleNamespace(packet=None)
    v._overlay()
    checks = [k.get("text") for a, k in v.canvas.of("text") if k.get("text") == "✓"]
    assert len(checks) == 3                                     # chosen outlines carry a check mark
    assert "Setup frame - monitoring has not started" in v.canvas.texts()


def test_overlap_with_saved_items_needs_an_explicit_choice_and_shows_the_comparison(rig):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    rig.remember()
    c.back()
    assert c.scan() and settle(c, lambda: c.scan_status == "done") and settle(c, lambda: not c.tasks.busy("saved"))
    assert c.go("select")
    v, _ns, _clk = build(c)
    phone = rig.cid("phone")
    v._canvas_press(ev(120, 150))
    assert c.candidate(phone).disposition == "choose"
    v._render_select()
    v._render_action_bar()
    assert v.ab_primary.cget("state") == "disabled" and "choose Use saved item" in v.ab_hint.cget("text")
    assert "Overlaps saved item Item 01 (reference v1" in v.sd_cmp_lbl.cget("text")
    assert v.sd_use.cget("state") == "normal" and v.sd_new.cget("state") == "normal"
    assert "Overlap is only a suggestion" in v.sd_explain.cget("text")
    v._sd_disposition("new")                                    # Create separate item
    v._render_select()
    v._render_action_bar()
    assert c.candidate(phone).disposition == "new" and v.ab_primary.cget("state") == "normal"
    assert v.ab_hint.cget("text") == "1 new · 0 reused"
    v._sd_disposition("reuse")                                  # Use saved item
    assert c.candidate(phone).disposition == "reuse" and V.select(c)["summary"] == "0 new · 1 reused"


def test_long_names_are_ellipsized_and_the_layout_reflows_at_1280x720_100_125_150(rig):
    assert V.ellipsize("A" * 80, 24) == "A" * 23 + "…" and V.ellipsize("Phone A01", 24) == "Phone A01"
    c = rig.ctl
    rig.discover(rig.a)
    cid = rig.cid("mug")
    assert c.rename_candidate(cid, "Limited edition ceramic coffee mug with an unusually long display name")
    c.toggle_candidate(cid, True)
    v, _ns, _clk = build(c)
    v._render_select()
    assert len(v.s_list.rows[f"cand:{cid}"]["text"]) <= 26
    for scale, tier in ((1.0, "standard"), (1.25, "compact"), (1.5, "compact")):
        lay = V.layout(1280, 720, scale)
        lw, lh = lay["logical"]
        assert lay["tier"] == tier and lay["compact_header"]
        assert lw - lay["rail"] - lay["side"] - 2 * 16 >= 300, (scale, lay)   # the scene keeps a usable width
        assert lay["side_collapsible"] == (lw < 1000)
        assert lay["wrap_px"] > 0 and lay["side_px"] == int(lay["side"] * scale)
    wide = V.layout(1440, 900, 1.0)
    assert wide["tier"] == "wide" and 180 <= wide["rail"] <= 200 and 300 <= wide["side"] <= 340


# ---- Configure ----------------------------------------------------------------------------------------------------
def test_configure_projection_is_factual_and_never_claims_tests_or_readiness(rig):
    c = rig.ctl
    rig.discover(rig.a)
    for lab in ("phone", "mug"):
        c.toggle_candidate(rig.cid(lab), True)
    rig.remember()
    cfg = V.configure(c)
    text = repr(cfg)
    assert "passed" not in text.lower() and "Ready" not in text and "Expected position" not in text
    assert [x["status"] for x in cfg["cards"]] == ["Saved", "Saved"] and cfg["count_text"] == "2 saved references"
    assert all(x["sub"].endswith("reference v1") for x in cfg["cards"])
    motion = next(r for r in cfg["readiness"] if r["label"].startswith("Motion"))
    assert motion["ok"] is None and motion["label"] == "Motion/removal/return not tested by setup"
    assert cfg["allowance_text"] == "Request review after 60 seconds of supported Not visible"
    assert "paused time never counts" in cfg["allowance_note"] and "not proof" in cfg["allowance_note"]
    assert cfg["outputs"] == {"record_annotated": True, "clean_source": True, "decision_trace": True, "repeat": False}
    assert cfg["setup_line"][1] == "neutral" and cfg["primary"] == ("Start monitoring", True)


# ---- Monitor: one presentation authority for counts, tray, inspector and boxes -------------------------------------
def _obs(item, state, box, name="ITEM", t_wall=50.0):
    return types.SimpleNamespace(item_id=item, state=state, reason="look-alike nearby" if state == "uncertain" else None,
                                 name=name, last_supported={"t_wall": t_wall},
                                 box=box if state in ("visible", "checking") else None)


def _packet(seq, basis, t, obs, run="run1"):
    return types.SimpleNamespace(packet_id=f"res_{seq:06d}", seq=seq, run_id=run, frame_t_mono=t, frame_t_wall=t,
                                 size=(640, 360), image=np.zeros((360, 640, 3), np.uint8),
                                 basis=types.SimpleNamespace(basis_id=basis), items=obs)


def test_counts_tray_inspector_and_boxes_follow_the_same_painted_packet(rig):
    c = rig.ctl
    rig.discover(rig.a)
    for lab in ("phone", "mug"):
        c.toggle_candidate(rig.cid(lab), True)
    rig.remember()
    rig.start()
    a, b = c.tray
    v, _ns, clk = build(c)
    v._snapshot = types.SimpleNamespace(run_id="run1", basis_id="b1", items=[])
    pk = _packet(1, "b1", clk.t, [_obs(a, "visible", (110.0, 105.0, 50.0, 110.0)), _obs(b, "uncertain", None)])
    v._painted = types.SimpleNamespace(packet=pk)
    c.select(f"item:{a}")
    rows, auth, _key = v._presentation()
    assert auth == "painted"
    v._render_tray(rows)
    v._render_inspector(rows)
    assert {k: v.tile_lbls[k][0].cget("text") for k in v.tile_lbls} == \
        {"monitored": "2", "visible": "1", "not_visible": "0", "other": "1"}
    assert v.tray.rows[a]["values"][0] == "Visible" and v.tray.rows[a]["tone"] == "success"
    assert v.tray.rows[b]["values"][0] == "Match uncertain" and v.tray.rows[b]["tone"] == "pending"
    assert v.insp_state.cget("text") == "Visible"                # the same shared wording as the tray
    assert v.insp_cap_cur.cget("text").startswith("Current view") and v.insp_current.cget("image")
    v._overlay()
    rects = v.canvas.of("rect")
    want = source_box_to_canvas(v._fit, (110.0, 105.0, 160.0, 215.0))
    assert any(tuple(round(x, 1) for x in r[0]) == tuple(round(x, 1) for x in want) for r in rects)
    assert all("Match uncertain" not in (t or "") for t in v.canvas.texts())   # no box for the uncertain item
    # the same image 1.5 s later: no live authority anywhere, nothing counted as Visible, no box
    clk.t += 1.5
    rows, auth, _ = v._presentation()
    v._render_tray(rows)
    v._render_inspector(rows)
    assert v.tile_lbls["visible"][0].cget("text") == "0" and v.tile_lbls["other"][0].cget("text") == "2"
    assert v.tray.rows[a]["values"][0].startswith("Stale - last image 1.5 s old showed Visible")
    assert v.tray.rows[a]["tone"] == "neutral"
    assert "No current supported view" in v.insp_cap_cur.cget("text") and "(historical)" in v.insp_cap_cur.cget("text")
    v._overlay()
    assert not [r for r in v.canvas.of("rect") if r[1].get("outline") in ("#4ade80", "#fbbf24")]
    assert any("STALE" in (t or "") for t in v.canvas.texts())


# ---- Finish and Review -----------------------------------------------------------------------------------------------
def test_eof_with_an_output_failure_finishes_honestly_and_review_reuse_new_stay_reachable(rig):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    rig.remember()
    rig.start()
    v, _ns, _clk = build(c)
    rig.stop(kind=("finalized", "source_ended"), statuses=("finalized", "failed", "finalized"))
    assert c.tray == [] and c.view_step == "monitor"
    f = V.finish(c)
    assert f["title"] == "Video ended" and "Annotated recording: FAILED" in f["outputs"]
    assert any("not evidence that any item is absent" in n for n in f["notes"]) and not f["complete"]
    v._render_finish()
    assert v.f_title.cget("text") == "Video ended" and "Clean recording: saved" in v.f_outputs.cget("text")
    v._render_action_bar()
    assert [b.cget("text") for b in v.ab_sec] == ["Review saved run", "Reuse this setup"]
    assert v.ab_primary.cget("text") == "New setup" and v.ab_primary.cget("state") == "normal"
    v._action_sec(0)                                            # Review saved run -> Runs & Review (no inference)
    assert v.shown[-1] == "runs" and c.review_request == c.finished["run_id"]
    v._action_sec(1)                                            # Reuse this setup -> Configure, same ids
    assert settle(c, lambda: c.view_step == "configure" and c.tray)
    assert c.finished is None and len(rig.eng.session.started) == 1
    c.select(f"item:{c.tray[0]}")
    v._view_all_activity()
    v._insp_same()
    assert v.shown[-2:] == ["activity", "corrections"]          # Activity and Same item stay reachable


def test_header_and_discover_wording_are_replay_honest(rig):
    c = rig.ctl
    h = V.header(c)
    assert h["source"] == ("No replay source", "neutral") and h["session"] == ("Monitoring not started", "neutral")
    rig.prepare(rig.a)
    assert V.header(c)["source"] == ("Replay source ready", "accent")
    assert V.discover(c)["frame_caption"] == "Setup frame - monitoring has not started"
    assert V.title(c) == {"title": "Set up display monitoring", "subtitle": "Discover items"}


def test_owner_report_two_overlaps_show_a_visible_choice_card_bulk_action_true_count_and_label(rig):
    """Owner's screenshot (26 Oct): two selected suggestions overlap saved items; the button was disabled, the count said
    0 selected and the decision buttons were below the fold. Now the decision is at the top of the panel and in the
    bottom bar, the count includes undecided items, and the button names how many items it will save."""
    c = rig.ctl
    rig.discover(rig.a)
    for lab in ("phone", "mug"):
        c.toggle_candidate(rig.cid(lab), True)
    rig.remember()
    saved = set(c.tray)
    assert c.new_setup()
    rig.discover()
    v, _ns, _clk = build(c)
    v._canvas_press(ev(330, 230))                               # the mug (smallest outline)
    v._canvas_press(ev(120, 150))                               # the phone
    assert len(c.pending_choices()) == 2
    v._render_select()
    v._render_action_bar()
    assert v.s_count.cget("text") == "2 selected · 2 to decide · 3 suggestions"
    assert v.ab_primary.cget("text") == "Remember & configure 2 items" and v.ab_primary.cget("state") == "disabled"
    assert v.s_choice.winfo_ismapped() and v.s_ch_title.cget("text") == "2 selected items need your choice"
    assert "overlaps saved Item 01" in v.s_ch_text.cget("text") and "keeps its ID and history" in v.s_ch_text.cget("text")
    assert v.s_ch_use.cget("text").startswith("Item 0") and "use saved" in v.s_ch_use.cget("text")
    assert [b.cget("text") for b in v.ab_sec][0] == "Use saved items for all" and v.ab_sec[0].cget("state") == "normal"
    assert "2 selected items overlap saved items" in v.ab_hint.cget("text")
    v._action_sec(0)                                            # one click from the always-visible bottom bar
    v._render_select()
    v._render_action_bar()
    assert c.pending_choices() == [] and not v.s_choice.winfo_ismapped()
    assert {x.reuse_item for x in c.candidates if x.checked} == saved
    assert v.ab_primary.cget("state") == "normal" and v.ab_primary.cget("text") == "Remember & configure 2 items"
    assert v.s_count.cget("text") == "2 selected · 3 suggestions" and v.ab_hint.cget("text") == "0 new · 2 reused"
    rig.remember()
    assert c.enroll_state == "confirmed" and set(c.tray) == saved and "0 new, 2 reused" in c.enroll_message


def test_one_at_a_time_choice_and_create_separate_for_all(rig):
    c = rig.ctl
    rig.discover(rig.a)
    for lab in ("phone", "mug"):
        c.toggle_candidate(rig.cid(lab), True)
    rig.remember()
    assert c.new_setup()
    rig.discover()
    v, _ns, _clk = build(c)
    for lab in ("phone", "mug"):
        c.toggle_candidate(rig.cid(lab), True)
    v._render_select()
    first = c.pending_choices()[0]
    v._choice_next("new")                                       # "Item 0x: separate item"
    assert first.disposition == "new" and len(c.pending_choices()) == 1 and c.selected == f"cand:{first.candidate_id}"
    v._render_select()
    v._choices_all("new")                                       # "Create separate items for all"
    v._render_select()
    assert c.pending_choices() == [] and not v.s_choice.winfo_ismapped()
    assert V.select(c)["summary"] == "2 new · 0 reused"


def test_image_labels_take_the_image_size_instead_of_a_text_sized_sliver(rig):
    from lab05_view_harness import Sink
    from echo_watch.app.tk_app import show_image
    lbl = Sink()
    show_image(lbl, "photo", (12, 5))
    assert lbl.cget("image") == "photo" and (lbl.cget("width"), lbl.cget("height")) == (0, 0)
    show_image(lbl, None, (12, 5))
    assert lbl.cget("image") == "" and (lbl.cget("width"), lbl.cget("height")) == (12, 5)
    c = rig.ctl
    rig.discover(rig.a)
    v, _ns, _clk = build(c)
    cid = rig.cid("mug")
    v._cand_photos[(c.scan_id, cid, "detail")] = "mug-photo"
    v._focus(f"cand:{cid}")
    v._render_select()
    assert v.sd_thumb.cget("image") == "mug-photo" and v.sd_thumb.cget("width") == 0 and v.sd_thumb.cget("height") == 0


def test_r3_inspector_current_view_swaps_safely_and_updates_at_most_once_per_second(rig):
    """Owner run 14:30 (log: 911 'image pyimage72 doesn't exist' tick failures): the current-view image was released
    while its label still showed it. The label must receive the NEW image while the old one is still held, and the
    panel changes at most once per second (no per-packet flicker)."""
    from lab05_view_harness import Sink
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    rig.remember()
    rig.start()
    item = c.tray[0]
    c.select(f"item:{item}")
    v, ns, clk = build(c)
    made = []
    ns["_photo"] = lambda root, ppm: made.append(object()) or made[-1]
    swaps = []

    class Label(Sink):
        def configure(self, **kw):
            if kw.get("image"):
                swaps.append((kw["image"], getattr(v, "_insp_cur_photo", None)))
            super().configure(**kw)
    v.insp_current = Label()
    v._snapshot = types.SimpleNamespace(run_id="run1", basis_id="b1", items=[])
    for k in range(40):                                         # 40 packets at 25/s = 1.6 s
        pk = _packet(k + 1, "b1", clk.t, [_obs(item, "visible", (100.0 + k, 100.0, 60.0, 120.0))])
        v._painted = types.SimpleNamespace(packet=pk)
        rows, _a, _k = v._presentation()
        v._render_inspector(rows)
        clk.t += 0.04
    assert len(made) == 2                                       # 0.0 s and >= 1.0 s: two updates, not forty
    assert len(swaps) == 2 and swaps[1][1] is made[0]          # the 2nd swap happened while the 1st image was held
    for new, held in swaps:
        assert held is not new                                  # the label switched while the previous image was held
    assert v.insp_current.cget("image") is v._insp_cur_photo is made[-1]
    assert v.insp_cap_cur.cget("text").startswith("Current view")


def test_r3_scene_size_cannot_follow_changing_text():
    """Structural guard for the owner-reported once-a-second 'zoom' (scene canvas resized by clock-driven text).
    Not rendered verification: it pins the layout decisions that make the canvas size independent of text."""
    import ast
    import inspect
    from echo_watch.app import tk_app
    src = inspect.getsource(tk_app)
    tree = ast.parse(src)
    cls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    fit_src = ast.get_source_segment(src, next(f for f in cls["CardStrip"].body if getattr(f, "name", "") == "_fit"))
    assert "height" not in fit_src                              # the tray strip never resizes to its cards
    build_card = ast.get_source_segment(src, next(f for f in cls["CardStrip"].body if getattr(f, "name", "") == "_build"))
    assert "grid_propagate(False)" in build_card and "wraplength" not in build_card
    assert "r.grid_propagate(False)" in src                     # the window never follows its children's size
    assert "self.under.pack_propagate(False)" in src and "msgf.pack_propagate(False)" in src
    assert 'self.scene_time = tk.Label(hd, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, width=9' in src
    assert ".pack_forget()" not in ast.get_source_segment(
        src, next(f for f in cls["WorkbenchWindow"].body if getattr(f, "name", "") == "_render_chrome"))
