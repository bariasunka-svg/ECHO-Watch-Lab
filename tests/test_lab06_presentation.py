# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 matrix 5: the Tk-free YOLOE projections and the PRODUCTION YoloePage renderers driven against recording
sinks. ``tk`` / ``ttk`` / ``filedialog`` in the page module are replaced by a guard that fails on any use and PhotoImage
creation by a marker, so no Tcl interpreter, widget or default root can exist. This checks wiring and invariants
(one packet -> image + boxes + rows + counts + detail; safe image replacement order; detection-only without inference;
stale/other-generation rejection; fixed geometry decisions); it is NOT rendered verification."""
from __future__ import annotations

import ast
import time
from types import SimpleNamespace

import numpy as np
import pytest

import echo_watch.app.yoloe_page as page_mod
from echo_watch.app import yoloe_views as YV
from echo_watch.app.display import prepare
from echo_watch.app.display_schedule import DisplayScheduler, PrepRequest
from echo_watch.config import ROOT
from echo_watch.yoloe.association import join, summarize
from echo_watch.yoloe.profile import VIEW_DETECTIONS, VIEW_TRACKS
from echo_watch.yoloe.session import YoloePacket
from lab05_view_harness import NoTk, Sink


def packet(seq=1, run="yrun_a", epoch="yrun_a/e1", gen=3, dets=None, rows=None, tent=None, img_val=10):
    dets = dets or [{"i": 0, "xyxy": [10, 10, 40, 50], "score": 0.82, "cls": 1, "name": "cup"},
                    {"i": 1, "xyxy": [50, 5, 70, 30], "score": 0.31, "cls": 2, "name": "bottle"},
                    {"i": 2, "xyxy": [5, 60, 30, 90], "score": 0.12, "cls": 3, "name": "laptop"},
                    {"i": 3, "xyxy": [80, 60, 99, 90], "score": 0.55, "cls": 4, "name": "mug"}]
    rows = rows if rows is not None else [[10, 10, 40, 50, 17, 0.82, 1, 0]]
    tent = tent if tent is not None else [(18, 3, [80, 60, 99, 90], 0.55, 4)]
    assoc, _ = join(len(dets), rows, tent)
    img = np.full((100, 120, 3), img_val, np.uint8)
    return YoloePacket(run, epoch, gen, seq, {"out_n": seq - 1, "media_pts": seq / 10}, img, (120, 100), tuple(dets),
                       assoc, (), summarize(len(dets), assoc), False, time.monotonic(), time.monotonic())


# ---- projections -------------------------------------------------------------------------------------------------------
def test_labels_show_real_names_ids_and_untracked_without_inventing_ids():
    ents = {e["i"]: e for e in YV.det_entries(packet(), VIEW_TRACKS, 0.25)}
    assert ents[0]["label"] == "cup · #17 · 0.82" and ents[0]["dash"] == ()
    assert ents[1]["label"] == "bottle · Untracked · 0.31" and ents[1]["color"] == YV.UNTRACKED_COLOR
    assert ents[3]["label"] == "mug · Untracked · 0.55" and ents[3]["tentative"] and ents[3]["track_id"] is None
    assert ents[2]["shown"] is False                                   # 0.12 < 0.25: hidden, still a detection
    det = {e["i"]: e for e in YV.det_entries(packet(), VIEW_DETECTIONS, 0.25)}
    assert det[0]["label"] == "cup · 0.82" and "#" not in "".join(e["label"] for e in det.values())
    assert len(det) == len(ents) == 4                                  # detection-only removes ids, not detections


def test_track_colours_are_stable_per_run_epoch_track_and_not_state_colours():
    a = YV.track_color("r", "r/e1", 17)
    assert a == YV.track_color("r", "r/e1", 17) and a in YV.TRACK_PALETTE
    assert {YV.track_color("r", "r/e1", t) for t in range(1, 40)} > {a}
    from echo_watch.app import tk_theme as T
    assert not set(YV.TRACK_PALETTE) & set(T.STATE_BOX.values())       # never the Visible / Checking state colours
    assert YV.UNTRACKED_COLOR not in YV.TRACK_PALETTE


def test_counts_distinguish_detections_associated_untracked_and_shown():
    pk = packet()
    assert YV.counts(pk, 0.25) == {"detections": 4, "associated": 1, "tentative": 1, "untracked": 3, "shown": 3}
    assert YV.count_texts(pk, 0.6)["detections"] == ("4", "on this image · 1 shown")         # R1 wording
    assert [r[0] for r in YV.list_rows(pk, VIEW_TRACKS, 0.25)] == ["0", "1", "3"]
    assert YV.list_rows(pk, VIEW_DETECTIONS, 0.25)[0][1][2] == "-"


def test_hit_testing_prefers_the_smallest_shown_box():
    ents = YV.det_entries(packet(dets=[{"i": 0, "xyxy": [0, 0, 100, 100], "score": 0.9, "cls": 1, "name": "table"},
                                       {"i": 1, "xyxy": [20, 20, 40, 40], "score": 0.9, "cls": 2, "name": "cup"},
                                       {"i": 2, "xyxy": [25, 25, 35, 35], "score": 0.05, "cls": 3, "name": "x"}],
                                 rows=[], tent=[]), VIEW_TRACKS, 0.25)
    assert YV.hit(ents, 30, 30) == 1 and YV.hit(ents, 5, 5) == 0 and YV.hit(ents, 200, 200) is None


@pytest.mark.parametrize("st,banner", [
    ({"phase": "paused"}, "PAUSED"), ({"phase": "finalizing"}, "FINALIZING"),
    ({"phase": "running", "run_id": "other"}, "PREVIOUS TEST"),
    ({"phase": "running", "epoch": "yrun_a/e2"}, "PREVIOUS STATE"),
    ({"phase": "running", "control_gen": 4}, "PREVIOUS STATE"),
    ({"phase": "running", "inflight_s": 1.6}, "ANALYSIS DELAYED"),
    ({"phase": "idle", "outcome": SimpleNamespace(status="completed")}, "ENDED"),
])
def test_only_the_current_runs_current_state_is_live(st, banner):
    base = {"phase": "running", "run_id": "yrun_a", "epoch": "yrun_a/e1", "control_gen": 3, "inflight_s": None}
    live, text = YV.authority(packet(), {**base, **st}, time.monotonic())
    assert not live and text.startswith(banner)
    assert YV.authority(packet(), base, time.monotonic()) == (True, None)


def test_selection_follows_its_track_or_becomes_historical_text():
    p1 = packet(seq=1)
    sel = YV.select(p1, 0)
    p2 = packet(seq=2, dets=[{"i": 0, "xyxy": [0, 0, 5, 5], "score": 0.9, "cls": 9, "name": "phone"},
                             {"i": 1, "xyxy": [12, 10, 42, 50], "score": 0.8, "cls": 1, "name": "mug"}],
                rows=[[12, 10, 42, 50, 17, 0.8, 1, 1]], tent=[])
    s2, note = YV.follow_selection(sel, p2)
    assert s2["i"] == 1 and s2["key"] == p2.key and note is None          # same track, its NEW index and name
    assert YV.detail_lines(p2, s2, None, VIEW_TRACKS)[0].startswith("mug")
    p3 = packet(seq=3, rows=[], tent=[])
    s3, note3 = YV.follow_selection(s2, p3)
    assert s3 is None and "#17 mug" in note3 and "frame 2" in note3             # its latest name and frame
    assert YV.detail_lines(p3, None, note3, VIEW_TRACKS) == [note3]
    p_other = packet(seq=4, epoch="yrun_a/e2")
    assert YV.follow_selection(s2, p_other)[0] is None                    # ids never join across epochs


def test_footer_reports_actual_rates_and_playback_speed():
    snap = {"phase": "running", "analysed_fps": 11.84, "playback_speed": 0.47, "processing_ms_p50": 83.2}
    assert YV.footer(snap, 9.5) == "Analysed 11.8/s · Displayed 9.5/s · Processing 83 ms · Playback 0.47x source speed"
    assert YV.footer({"phase": "idle"}, 0) == ""
    assert YV.layout(1280, 720, 1.0) == {"side_px": 300, "details_open": False, "list_rows": 6, "compact": True,
                                         "file_chars": 22, "fin_px": 150}
    assert YV.layout(1920, 1080, 1.5)["side_px"] == 450


# ---- the production page renderers against sinks ----------------------------------------------------------------------
class Canvas:
    def __init__(self):
        self.items, self.n, self.log = {}, 0, []

    def _add(self, kind, args, kw):
        self.n += 1
        self.items[self.n] = (kind, args, kw)
        self.log.append(("create", kind, self.n, kw.get("image")))
        return self.n

    def create_image(self, *a, **k):
        return self._add("image", a, k)

    def create_rectangle(self, *a, **k):
        return self._add("rect", a, k)

    def create_text(self, *a, **k):
        return self._add("text", a, k)

    def delete(self, *what):
        for w in what:
            ids = [w] if isinstance(w, int) else [i for i, (_k, _a, kw) in self.items.items() if w in kw.get("tags", ())]
            for i in ids:
                self.items.pop(i, None)
                self.log.append(("delete", i))

    def bbox(self, i):
        x, y = self.items[i][1][:2]
        return (x, y - 6, x + 60, y + 6)

    def tag_lower(self, *a):
        pass

    tag_raise = tag_lower

    def winfo_width(self):
        return 602

    def winfo_height(self):
        return 502

    def of(self, kind, tag="yov"):
        return [(a, kw) for (k, a, kw) in self.items.values() if k == kind and tag in kw.get("tags", ())]


class Tree:
    def __init__(self):
        self.rows, self.order, self.sel, self.ops = {}, [], (), 0

    def get_children(self):
        return tuple(self.order)

    def delete(self, *iids):
        for i in iids:
            self.rows.pop(i)
            self.order.remove(i)
            self.ops += 1

    def insert(self, parent, index, iid, values=(), text=""):
        self.rows[iid] = tuple(values)
        self.order.insert(index if index != "end" else len(self.order), iid)
        self.ops += 1

    def item(self, iid, option=None, **kw):
        if "values" in kw:
            self.rows[iid] = tuple(kw["values"])
            self.ops += 1
            return None
        return self.rows[iid]

    def index(self, iid):
        return self.order.index(iid)

    def move(self, iid, parent, index):
        self.order.remove(iid)
        self.order.insert(index, iid)

    def selection(self):
        return self.sel

    def selection_set(self, items):
        self.sel = tuple(items) if not isinstance(items, str) else (items,)

    def selection_remove(self, *iids):
        self.sel = ()


class FakeCtl:
    def __init__(self, session):
        self.session = session
        self.view, self.shown_conf = VIEW_TRACKS, 0.25
        self.selection, self.selection_note = None, None
        self.start_pending, self.finished, self.phase = False, None, "running"
        self.followed = []

    def select(self, pk, i):
        self.selection, self.selection_note = YV.select(pk, i), None

    def follow(self, pk):
        self.followed.append(pk.seq)
        sel, note = YV.follow_selection(self.selection, pk)
        self.selection = sel
        if note:
            self.selection_note = note


class FakeSession:
    def __init__(self):
        self.painted = []
        self.st = {"phase": "running", "run_id": "yrun_a", "epoch": "yrun_a/e1", "control_gen": 3, "outcome": None,
                   "inflight_s": None}
        self.latest = None

    def note_painted(self, pk, now):
        if pk.key in [p.key for p in self.painted]:
            return False
        self.painted.append(pk)
        return True

    def state(self):
        return dict(self.st)

    def latest_packet(self):
        return self.latest


@pytest.fixture()
def page(monkeypatch):
    guard = NoTk()
    monkeypatch.setattr(page_mod, "tk", guard)
    monkeypatch.setattr(page_mod, "ttk", guard)
    monkeypatch.setattr(page_mod, "filedialog", guard)
    p = page_mod.YoloePage.__new__(page_mod.YoloePage)
    photos = []

    def photo(root, ppm):
        photos.append(object())
        return photos[-1]
    sess = FakeSession()
    p.__dict__.update(root=None, c=FakeCtl(sess), canvas=Canvas(), tree=Tree(), prep=None, rate=page_mod.DisplayedRate(),
                      _photo=None, _img_item=None, _painted=None, _fit=None, _overlay_key=None, _rows_key=None,
                      _placeholder=None, _photo_fn=photo, detail=Sink("detail"), frame_lbl=Sink("frame"),
                      tiles={k: (Sink(k + "n"), Sink(k + "d")) for k in ("detections", "associated", "untracked")})
    p.schedule = DisplayScheduler(SimpleNamespace(submit=lambda tok, req: None))
    p._photos = photos
    return p


def prepared(pk, w=600, h=500):
    gen = ("yoloe", pk.run_id, pk.epoch)
    return prepare(PrepRequest((gen, pk.seq, w, h), gen, pk.seq, pk, pk.image, w, h))


def test_paint_publishes_image_boxes_rows_counts_and_detail_from_one_packet(page):
    pk = packet()
    page.c.select(pk, 0)
    page._paint(prepared(pk))
    page._overlay(time.monotonic(), page.c.session.state())
    assert page._painted.packet is pk and page.c.session.painted == [pk] and page.c.followed == [1]
    assert page.tree.order == ["0", "1", "3"] and page.tree.rows["0"] == ("cup", "0.82", "#17")
    assert page.tree.rows["3"] == ("mug", "0.55", "Untracked") and page.tree.sel == ("0",)
    assert page.tiles["detections"][0].cget("text") == "4" and page.tiles["associated"][0].cget("text") == "1"
    assert page.tiles["untracked"][0].cget("text") == "3" and "3 shown" in page.tiles["detections"][1].cget("text")
    assert page.detail.cget("text").startswith("cup") and "Track #17" in page.detail.cget("text")
    texts = [kw["text"] for _a, kw in page.canvas.of("text")]
    assert sorted(texts) == sorted(["cup · #17 · 0.82", "bottle · Untracked · 0.31", "mug · Untracked · 0.55"])
    rects = page.canvas.of("rect")
    assert len(rects) == 6 and any(kw.get("width") == 3 for _a, kw in rects)     # 3 boxes + 3 label plates, picked


def test_a_new_image_is_on_the_canvas_before_the_old_one_is_removed_and_released(page):
    a, b = packet(seq=1), packet(seq=2, img_val=99)
    page._paint(prepared(a))
    first_item, first_photo = page._img_item, page._photo
    page.canvas.log.clear()
    page._paint(prepared(b))
    creates = [e for e in page.canvas.log if e[0] == "create" and e[1] == "image"]
    deletes = [e for e in page.canvas.log if e[0] == "delete" and e[1] == first_item]
    assert creates and deletes and page.canvas.log.index(creates[0]) < page.canvas.log.index(deletes[0])
    assert creates[0][3] is page._photo and page._photo is not first_photo and first_item not in page.canvas.items


def test_detection_only_and_the_shown_filter_redraw_from_the_same_packet_without_inference(page):
    pk = packet()
    page._paint(prepared(pk))
    st = page.c.session.state()
    page._overlay(time.monotonic(), st)
    page.c.view = VIEW_DETECTIONS
    page._overlay_key = page._rows_key = None
    page._overlay(time.monotonic(), st)
    page._publish(pk)
    texts = [kw["text"] for _a, kw in page.canvas.of("text")]
    assert len(texts) == 3 and not any("#" in t or "Untracked" in t for t in texts)
    assert page.tree.rows["0"][2] == "-" and page._painted.packet is pk and page.c.session.painted == [pk]
    page.c.shown_conf = 0.6
    page._overlay_key = page._rows_key = None
    page._overlay(time.monotonic(), st)
    page._publish(pk)
    assert len(page.canvas.of("text")) == 1 and page.tree.order == ["0"]
    assert page.tiles["detections"][0].cget("text") == "4"                 # available count unchanged


def test_paused_or_previous_state_images_carry_a_banner_and_no_live_claim(page):
    pk = packet()
    page._paint(prepared(pk))
    page._overlay(time.monotonic(), {**page.c.session.state(), "phase": "paused"})
    texts = [kw["text"] for _a, kw in page.canvas.of("text")]
    assert any(t.startswith("PAUSED at frame 1") for t in texts)


def test_completions_of_another_run_or_epoch_are_never_painted(page):
    pk_old = packet(seq=5, epoch="yrun_a/e1")
    pk_new = packet(seq=1, epoch="yrun_a/e2")
    sched = page.schedule
    gen_new = ("yoloe", "yrun_a", "yrun_a/e2")
    res = sched.tick(None, gen_new, [(1, prepared(pk_old))])
    assert res is None and sched.dropped == {"outside_current_generation": 1}
    assert sched.tick(None, gen_new, [(2, prepared(pk_new))]).packet is pk_new
    assert sched.tick(None, gen_new, [(3, prepared(packet(seq=0, epoch="yrun_a/e2")))]) is None   # never regresses


def test_canvas_click_selects_the_smallest_box_and_row_selection_uses_the_painted_packet(page):
    pk = packet()
    page._paint(prepared(pk))
    f = page._fit
    x, y = f.offset_x + 85 * f.scale, f.offset_y + 70 * f.scale
    page._canvas_click(SimpleNamespace(x=x, y=y))
    assert page.c.selection["i"] == 3 and page.tree.sel == ("3",)
    page.tree.sel = ("1",)
    page._syncing = False
    page._row_selected()
    assert page.c.selection["i"] == 1 and page.c.selection["key"] == pk.key
    assert "Untracked" in page.detail.cget("text")


def test_list_updates_are_incremental_when_detections_persist(page):
    page._paint(prepared(packet(seq=1)))
    ops = page.tree.ops
    page._paint(prepared(packet(seq=2)))
    assert page.tree.ops - ops == 0                                          # same rows: no delete/insert churn
    page._paint(prepared(packet(seq=3, dets=[{"i": 0, "xyxy": [1, 1, 9, 9], "score": 0.9, "cls": 5, "name": "phone"}],
                                rows=[], tent=[])))
    assert page.tree.order == ["0"] and page.tree.rows["0"] == ("phone", "0.90", "Untracked")


# ---- fixed geometry and window wiring (structural guards; not rendering) ----------------------------------------------
def _calls(src: str) -> list[str]:
    return [ast.unparse(n) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]


def test_changing_text_cannot_resize_the_yoloe_scene():
    src = (ROOT / "src/echo_watch/app/yoloe_page.py").read_text("utf-8")
    calls = _calls(src)
    for must in ("side.grid_propagate(False)", "side.pack_propagate(False)", "top.pack_propagate(False)",
                 "hd.pack_propagate(False)", "foot.pack_propagate(False)", "fin.pack_propagate(False)",
                 "info.pack_propagate(False)", "left.grid_propagate(False)"):
        assert must in calls, must
    labels = {ast.unparse(n.targets[0]): {k.arg for k in n.value.keywords} for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and len(n.targets) == 1
              and ast.unparse(n.value.func) in ("tk.Label", "tk.Text")}
    for lbl in ("self.file_lbl", "self.frame_lbl"):
        assert "width" in labels[lbl], lbl                                    # fixed width: names cannot push layout
    for lbl in ("self.fin_lbl", "self.detail"):
        assert "height" in labels[lbl], lbl                                   # fixed lines inside fixed cards
    assert "fin.grid_propagate(False)" in calls                              # R1: finish card is a fixed 2x2 grid
    tree = ast.parse(src)
    paint = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_paint")
    body = ast.unparse(paint)
    assert body.index("create_image") < body.index("canvas.delete(self._img_item)") < body.index("= (item, photo)")


def test_yoloe_sits_above_display_assets_and_header_controls_follow_the_active_mode():
    import echo_watch.app.tk_app as tk_app
    assert [k for k, *_ in tk_app.NAV][:2] == ["yoloe", "assets"]
    src = (ROOT / "src/echo_watch/app/tk_app.py").read_text("utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    ns = {}
    exec(compile(ast.Module([fns["_header_stop"], fns["_yoloe_busy"], fns["_pause_resume"]], []), "tk_app", "exec"),
         {"PHASE_PAUSED": "paused"}, ns)
    log = []
    yc = SimpleNamespace(busy=lambda: True, phase="running", stop=lambda: log.append("y_stop"),
                         pause=lambda: log.append("y_pause"), resume=lambda: log.append("y_resume"))
    da = SimpleNamespace(stop=lambda: log.append("da_stop"), phase="active", pause=lambda: log.append("da_pause"),
                         resume=lambda: log.append("da_resume"))
    me = SimpleNamespace(yc=yc, c=da)
    me._yoloe_busy = lambda: ns["_yoloe_busy"](me)
    ns["_header_stop"](me)
    ns["_pause_resume"](me)
    yc.busy = lambda: False
    ns["_header_stop"](me)
    ns["_pause_resume"](me)
    assert log == ["y_stop", "y_pause", "da_stop", "da_pause"]
    show = ast.unparse(fns["show"])
    assert "yc.start" not in show and "session.start" not in show and "choose" not in show   # opening starts nothing
    assert "self.show('assets')" in ast.unparse(fns["_build"])                                # landing unchanged


def test_small_windows_shrink_the_list_and_drop_optional_text_at_a_deliberate_breakpoint(page):
    page.scale, page._lay = 1.0, YV.layout(1440, 900, 1.0)
    page.side, page.tree, page.fin, page.subtitle = Sink("side"), Sink("tree"), Sink("fin"), Sink("subtitle")
    page.fin_lbl, page.detail, page.file_lbl, page._title_lbl = Sink("fin"), Sink("detail"), Sink("file"), Sink("t")
    page.subtitle.mapped = True
    inval = []
    page._invalidate = lambda: inval.append(1)
    page.apply_layout(1280, 720)
    assert page.side.cget("width") == 300 and page.tree.cget("height") == 6 and page.fin.cget("height") == 150
    assert page.file_lbl.cget("width") == 22 and page.subtitle.mapped is False and len(inval) == 1
    page.apply_layout(1280, 720)                                             # same tier: nothing is touched again
    assert len(inval) == 1
    page.apply_layout(1920, 1080)
    assert page.tree.cget("height") == 10 and page.side.cget("width") == 340 and page.subtitle.mapped is True
    assert page.file_lbl.cget("width") == 34


def test_a_yoloe_page_fault_is_contained_and_display_assets_keeps_running():
    src = (ROOT / "src/echo_watch/app/tk_app.py").read_text("utf-8")
    fns = {n.name: n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
    build = ast.unparse(fns["_build_yoloe"])
    assert "except Exception" in build and "self.yoloe_page = None" in build and "Display Assets is" in build
    tick = ast.unparse(fns["_tick"])
    i_y, i_try = tick.index("self.yoloe_page.tick"), tick.rindex("try:", 0, tick.index("self.yoloe_page.tick"))
    assert tick.index("self.c.pump()") < i_try < i_y and "_yoloe_tick_errors" in tick
