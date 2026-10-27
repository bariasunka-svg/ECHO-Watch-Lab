# SPDX-License-Identifier: AGPL-3.0-only
"""GUI-free harness for the EW-LAB-05 production renderers.

The REAL ``WorkbenchWindow`` methods named in ``METHODS`` are extracted from tk_app.py by AST and executed with the
module's own globals, except: ``time`` is a deterministic clock, ``_photo`` returns a marker (no PhotoImage), and
``tk`` / ``ttk`` / ``filedialog`` are replaced by a guard that raises on any use - so an accidental widget, Tcl
interpreter or default root can never be created. Widgets are replaced by recording sinks. This verifies the wiring
from controller state through app.views/app.presentation into what each widget is told to show; it is NOT rendered
verification.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import echo_watch.app.tk_app as tk_mod
from echo_watch.app.canvas_geometry import fit

METHODS = ("_render_action_bar", "_action", "_action_primary", "_action_sec", "_render_counts", "_render_finish",
           "_overlay", "_chip", "_canvas_press", "_hit_boxes", "_focus", "_name_of", "_tray_rows", "_presentation",
           "_render_tray", "_render_inspector", "_render_evidence", "_render_timeline", "_sync_tree_selection",
           "_toggle_focused", "_render_select", "_sync_list", "_render_select_detail", "_sd_save_name", "_sd_match_item",
           "_render_discover", "_progress", "_list_click", "_view_all_activity", "_insp_same", "_finish_review",
           "_sd_disposition", "_render_monitor_side", "_tray_card_click", "_to_src", "_list_focus", "_remember", "_save_config",
           "_start", "_cfg_pull", "_cr_allowance", "_co_changed", "_render_choices", "_choices_all", "_choice_next")


class NoTk:
    """Any attribute access raises: no widget / Tcl / default root can be created through this module."""

    def __getattr__(self, name):
        raise AssertionError(f"GUI construction attempted in a GUI-free test: tk.{name}")


class Sink:
    """A recording stand-in for a label/button/frame/progressbar."""

    def __init__(self, name="sink"):
        self.name, self.opts, self.mapped, self.calls = name, {"text": "", "image": "", "state": "normal", "bg": "",
                                                               "highlightbackground": ""}, False, []

    def configure(self, **kw):
        self.opts.update(kw)

    config = configure

    def cget(self, k):
        return self.opts.get(k, "")

    def grid(self, *a, **k):
        self.mapped = True

    pack = grid

    def grid_remove(self):
        self.mapped = False

    pack_forget = grid_forget = grid_remove

    def winfo_ismapped(self):
        return self.mapped

    def winfo_children(self):
        return []

    def winfo_width(self):
        return 400

    def __getattr__(self, name):           # start/stop/state/current/set/... are recorded no-ops
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k))


class FakeTree:
    def __init__(self):
        self.rows, self.order, self.sel = {}, [], ()

    def get_children(self):
        return tuple(self.order)

    def exists(self, iid):
        return iid in self.rows

    def delete(self, *iids):
        for i in iids:
            self.rows.pop(i, None)
            self.order.remove(i)

    def insert(self, parent, index, iid, text, values, **kw):
        self.rows[iid] = {"text": text, "values": tuple(values), "image": kw.get("image", ""), **kw}
        self.order.insert(index, iid)

    def item(self, iid, option=None, **kw):
        if kw:
            self.rows[iid].update({k: (tuple(v) if k == "values" else v) for k, v in kw.items()})
            return None
        return self.rows[iid].get(option, "")

    def index(self, iid):
        return self.order.index(iid)

    def move(self, iid, parent, index):
        self.order.remove(iid)
        self.order.insert(index, iid)

    def selection(self):
        return self.sel

    def selection_set(self, iid):
        self.sel = (iid,)

    def selection_remove(self, *iids):
        self.sel = ()

    def see(self, iid):
        pass

    def tag_configure(self, *a, **k):
        pass

    def configure(self, **kw):
        pass

    # click geometry for _list_click: row i at y = 10 + 20*i; the checkbox column is x < 40
    def identify_row(self, y):
        i = (y - 10) // 20
        return self.order[i] if 0 <= i < len(self.order) else ""

    def identify_column(self, x):
        return "#1" if x < 40 else "#0"


class CanvasSink:
    def __init__(self):
        self.items, self.n = {}, 0

    def _add(self, kind, args, kw):
        self.n += 1
        self.items[self.n] = (kind, args, kw)
        return self.n

    def create_rectangle(self, *a, **k):
        return self._add("rect", a, k)

    def create_text(self, *a, **k):
        return self._add("text", a, k)

    def create_oval(self, *a, **k):
        return self._add("oval", a, k)

    def create_line(self, *a, **k):
        return self._add("line", a, k)

    def delete(self, *tags):
        for t in tags:
            for i in [i for i, (_k, _a, kw) in self.items.items() if t == "all" or t in (kw.get("tags") or ())]:
                del self.items[i]

    def focus_set(self):
        pass

    def winfo_width(self):
        return 600

    def of(self, kind, tag="ov"):
        return [(a, k) for (kd, a, k) in self.items.values() if kd == kind and tag in (k.get("tags") or ())]

    def texts(self):
        return [k.get("text") for (kd, _a, k) in self.items.values() if kd == "text"]


class Clock:
    t = 1000.0

    def monotonic(self):
        return self.t

    def perf_counter(self):
        return self.t


class View(types.SimpleNamespace):
    """Unknown public widget names resolve to recording sinks; methods are the extracted production code."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        s = Sink(name)
        setattr(self, name, s)
        return s


def build(ctl, clock: Clock | None = None):
    src = Path(tk_mod.__file__).read_text("utf-8")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "WorkbenchWindow")
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert {f.name for f in fns} == set(METHODS), set(METHODS) - {f.name for f in fns}
    clock = clock or Clock()
    ns = dict(vars(tk_mod))
    ns.update(time=clock, tk=NoTk(), ttk=NoTk(), filedialog=NoTk(), _photo=lambda root, ppm: ("photo", len(ppm)))
    exec(compile(ast.Module(body=fns, type_ignores=[]), "tk_app.py", "exec"), ns)
    shown = []
    v = View(c=ctl, root=None, canvas=CanvasSink(), timeline=CanvasSink(), tray=FakeTree(), s_list=FakeTree(),
             s_saved=FakeTree(), _painted=None, _snapshot=None, _hover=None, _overlay_key=None,
             _inspector_for=None, _insp_cur_key=None, _sd_for=None, _published_key=None, _ab_keys={"primary": None,
                                                                                                  "sec": [None, None]},
             _cand_photos={}, _layout={"grid_cols": 2}, _draw_mode=None, shown=shown,
             tile_lbls={k: (Sink(k), Sink(k + "_d")) for k in ("monitored", "visible", "not_visible", "other")},
             ab_sec=[Sink("sec0"), Sink("sec1")], _thumbs_changed=False, _d_setups_sig=None, co_vars={},
             _fit=fit(640, 360, 640, 360),
             cr_name=types.SimpleNamespace(get=lambda: ctl.setup_name, set=lambda x: None),
             cr_allow=types.SimpleNamespace(get=lambda: str(ctl.pending_allowance), set=lambda x: None),
             name_var=types.SimpleNamespace(set=lambda x: None), sd_name=types.SimpleNamespace(get=lambda: "",
                                                                                                 set=lambda x: None))
    v.show = lambda key: shown.append(key)
    v._thumb = lambda *a, **k: None
    v._set_lbl = lambda lbl, text: lbl.configure(text=text)
    v._invalidate = lambda: None
    v._render_discover_grid = lambda d: None
    v._sync_tree_selection_calls = 0
    for name in METHODS:
        setattr(v, name, (lambda n: (lambda *a, **k: ns[n](v, *a, **k)))(name))
    return v, ns, clock
