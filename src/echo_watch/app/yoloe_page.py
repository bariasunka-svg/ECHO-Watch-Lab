# SPDX-License-Identifier: AGPL-3.0-only
"""The YOLOE workspace page of the native workbench (EW-LAB-06): Tk widgets over app.yoloe_controller and the Tk-free
projections in app.yoloe_views. Constructed ONLY by ``WorkbenchWindow`` during the owner's manual launch.

Layout (fixed geometry, after the EW-LAB-05 R3 anti-jitter rules): a top bar (title, file, status, Choose video /
Start / Pause / Stop / View test results), a dominant aspect-preserving video canvas with a presentation switch and a
fixed-height performance footer, and a FIXED-WIDTH side panel (counts, finish card, bounded detection list, selected
detail, optional Details). Every text that changes (clock, counts, names, errors) sits in a fixed-size cell, so it can
never resize the canvas or flash panels. A Results panel (saved tests, paged frame records, the original frame with its
saved boxes) replaces the live panel inside the page.

Painting follows the existing paired rules: a LatestWorker prepares the newest committed packet's OWN image; the tick
paints at most one completion of the current (run, epoch) generation, never regressing; boxes, list rows, counts and
the selected detail are written in the SAME callback from that painted packet. The new PhotoImage is placed on the
canvas before the old item is deleted and the old image released (R3 lifetime rule). No inference, decoding, hashing or
file I/O happens on the Tk thread.

EW-LAB-06 R1: the canvas message, status pill and finish card come from ONE read model (``yoloe_views.page_state``):
a failed start says which stage failed and why, keeps the chosen file and says no frame was analysed (never the
"choose a video" instructions); counters show dashes/"not analysed" before any image; an analysed empty image says so.
The side panel switches between Detections and a steady, scrollable, selectable Details text (bounded technical
failure detail) - no popup. The finish card is a 2x2 grid of actions under a fixed-height message, so no action is cut
off at compact widths; the saved-results panel shows the same details for a reopened test, including zero-frame ones.
"""
from __future__ import annotations

import logging
import time
import tkinter as tk
from types import SimpleNamespace
from tkinter import filedialog, ttk

from ..review.frames import LatestWorker
from ..yoloe.profile import SHOWN_CONF_MAX, SHOWN_CONF_MIN, VIEW_DETECTIONS, VIEW_TRACKS
from . import tk_theme as T
from . import yoloe_views as YV
from .canvas_geometry import CanvasFit, canvas_to_source, source_box_to_canvas
from .display import prepare, to_ppm
from .display_schedule import DisplayScheduler, PrepRequest

logger = logging.getLogger(__name__)


class DisplayedRate:
    """Distinct painted packets per second (repaints never count)."""

    def __init__(self):
        self._t: list[float] = []
        self.total = 0

    def add(self, now: float) -> None:
        self._t.append(now)
        self.total += 1
        while self._t and now - self._t[0] > 2.0:
            self._t.pop(0)

    def fps(self, now: float) -> float:
        t = [x for x in self._t if now - x <= 2.0]
        return 0.0 if len(t) < 2 else (len(t) - 1) / max(1e-6, t[-1] - t[0])


class YoloePage:
    def __init__(self, win, parent, controller):
        from .tk_app import _photo, button, pill
        self._photo_fn, self._button, self._pill = _photo, button, pill
        self.win, self.root, self.c = win, win.root, controller
        self.scale = win.scale
        self.prep = LatestWorker("yoloe-display-prep", prepare)
        self.schedule = DisplayScheduler(self.prep)
        self.rate = DisplayedRate()
        self._photo = None
        self._img_item = None
        self._painted = None
        self._fit = None
        self._overlay_key = None
        self._rows_key = None
        self._chrome_t = 0.0
        self._placeholder = None
        self._review_photo = None
        self._review_item = None
        self._review_key = None
        self._lay = YV.layout(1440, 900, self.scale)
        self._side_view = "detections"
        self._details_shown = None
        self._rdetails_shown = None
        self._rview = "details"
        self._source_gen = controller.source_gen
        self._counts_blank = False
        self._build(parent)

    # =================================================================================================================
    # build
    # =================================================================================================================
    def _build(self, f) -> None:
        S = self.scale
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)
        top = tk.Frame(f, bg=T.BG, height=int(56 * S))
        top.grid(row=0, column=0, sticky="ew")
        top.pack_propagate(False)
        self._title_lbl = tk.Label(top, text="YOLOE", bg=T.BG, fg=T.TEXT, font=T.FONT_H1_COMPACT)
        self._title_lbl.pack(side="left")
        self.subtitle = tk.Label(top, text="Prompt-free detection and tracking test", bg=T.BG, fg=T.TEXT_MUTED,
                                 font=T.FONT_SMALL)
        self.subtitle.pack(side="left", padx=(T.SPACE_2, T.SPACE_3))
        self.file_lbl = tk.Label(top, text="No video chosen", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL,
                                 width=self._lay["file_chars"], anchor="w")
        self.file_lbl.pack(side="left")
        self.status = self._pill(top, "Idle")
        self.status.configure(width=12)
        self.status.pack(side="left", padx=T.SPACE_2)
        self.b_results = self._button(top, "View test results", self.c.show_results, "outline")
        self.b_stop = self._button(top, "Stop", self.c.stop, "danger")
        self.b_pause = self._button(top, "Pause", self._pause_resume)
        self.b_pause.configure(width=8)
        self.b_start = self._button(top, "Start", self.c.start, "primary")
        self.b_choose = self._button(top, "Choose video...", self.choose_video, "outline")
        for b in (self.b_results, self.b_stop, self.b_pause, self.b_start, self.b_choose):
            b.pack(side="right", padx=2)
        body = tk.Frame(f, bg=T.BG)
        body.grid(row=1, column=0, sticky="nsew", pady=(T.SPACE_2, 0))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.live = tk.Frame(body, bg=T.BG)
        self.live.grid(row=0, column=0, sticky="nsew")
        self.results = tk.Frame(body, bg=T.BG)
        self.results.grid(row=0, column=0, sticky="nsew")
        self._build_live(self.live)
        self._build_results(self.results)
        self.live.tkraise()
        self._panel = "live"

    def _build_live(self, f) -> None:
        S = self.scale
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(0, weight=1)
        sc = tk.Frame(f, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        sc.grid(row=0, column=0, sticky="nsew")
        sc.grid_columnconfigure(0, weight=1)
        sc.grid_rowconfigure(1, weight=1)
        hd = tk.Frame(sc, bg=T.SURFACE, height=int(40 * S), padx=T.SPACE_3)
        hd.grid(row=0, column=0, sticky="ew")
        hd.pack_propagate(False)
        self.v_tracks = self._button(hd, YV.VIEW_LABEL[VIEW_TRACKS], lambda: self._set_view(VIEW_TRACKS))
        self.v_dets = self._button(hd, YV.VIEW_LABEL[VIEW_DETECTIONS], lambda: self._set_view(VIEW_DETECTIONS))
        self.v_tracks.pack(side="left", pady=4)
        self.v_dets.pack(side="left", padx=(2, 0), pady=4)
        self.frame_lbl = tk.Label(hd, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, width=22, anchor="e")
        self.frame_lbl.pack(side="right")
        self.canvas = tk.Canvas(sc, bg=T.VIDEO_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=T.SPACE_2)
        self.canvas.bind("<ButtonPress-1>", self._canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._invalidate())
        foot = tk.Frame(sc, bg=T.SURFACE, height=int(30 * S), padx=T.SPACE_3)
        foot.grid(row=2, column=0, sticky="ew")
        foot.pack_propagate(False)
        self.footer = tk.Label(foot, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w")
        self.footer.pack(fill="both", expand=True)
        side = tk.Frame(f, bg=T.BG, width=self._lay["side_px"])
        side.grid(row=0, column=1, sticky="ns", padx=(T.SPACE_3, 0))
        side.grid_propagate(False)
        side.pack_propagate(False)
        self.side = side
        tiles = tk.Frame(side, bg=T.BG)
        tiles.pack(fill="x")
        self.tiles = {}
        for key, (n0, d0) in YV.count_texts(None, self.c.shown_conf).items():
            t = tk.Frame(tiles, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER, padx=8, pady=4)
            t.pack(side="left", padx=(0, 4))
            n = tk.Label(t, text=n0, bg=T.SURFACE, fg=T.ACCENT, font=T.FONT_COUNT, width=3, anchor="w")
            n.pack(anchor="w")
            d = tk.Label(t, text=d0, bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO, width=14, anchor="w")
            d.pack(anchor="w")
            self.tiles[key] = (n, d)
        # finish/status card: fixed height, message on top, actions in a 2x2 grid (nothing is cut off when narrow)
        fin = tk.Frame(side, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER,
                       height=self._lay["fin_px"], padx=T.SPACE_2, pady=T.SPACE_1)
        fin.pack(fill="x", pady=(T.SPACE_2, 0))
        fin.pack_propagate(False)
        fin.grid_propagate(False)
        fin.grid_columnconfigure(0, weight=1, uniform="fin")
        fin.grid_columnconfigure(1, weight=1, uniform="fin")
        self.fin = fin
        self.fin_title = tk.Label(fin, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL_BOLD, anchor="w")
        self.fin_title.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.fin_lbl = tk.Label(fin, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="nw",
                                justify="left", wraplength=self._lay["side_px"] - 30, height=3)
        self.fin_lbl.grid(row=1, column=0, columnspan=2, sticky="new")
        self.f_results = self._button(fin, "View test results", self.c.show_results, "outline")
        self.f_again = self._button(fin, "Run again", self.c.run_again)
        self.f_other = self._button(fin, "Choose another video", self.choose_video)
        self.f_details = self._button(fin, "Details", lambda: self._show_side("details"))
        for i, b in enumerate((self.f_results, self.f_again, self.f_other, self.f_details)):
            b.configure(font=T.FONT_SMALL, padx=6)
            b.grid(row=2 + i // 2, column=i % 2, sticky="ew", padx=(0, 2), pady=(2, 0))
        sw = tk.Frame(side, bg=T.BG)
        sw.pack(fill="x", pady=(T.SPACE_2, 0))
        self.s_dets = self._button(sw, "Detections", lambda: self._show_side("detections"))
        self.s_details = self._button(sw, "Details", lambda: self._show_side("details"))
        self.s_dets.pack(side="left")
        self.s_details.pack(side="left", padx=(2, 0))
        # the side body takes the REMAINING height; each view scrolls inside it (no view can push the others away)
        body = tk.Frame(side, bg=T.BG)
        body.pack(fill="both", expand=True, pady=(T.SPACE_1, 0))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.list_view = tk.Frame(body, bg=T.BG)
        self.list_view.grid(row=0, column=0, sticky="nsew")
        self.details_view = tk.Frame(body, bg=T.BG)
        self.details_view.grid(row=0, column=0, sticky="nsew")
        adv = tk.Frame(self.list_view, bg=T.BG)
        adv.pack(fill="x")
        tk.Label(adv, text="Shown confidence", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL).pack(side="left")
        self.conf_var = tk.StringVar(value=f"{self.c.shown_conf:.2f}")
        self.conf = ttk.Spinbox(adv, from_=SHOWN_CONF_MIN, to=SHOWN_CONF_MAX, increment=0.05, width=5,
                                textvariable=self.conf_var, command=self._conf_changed, format="%.2f")
        self.conf.pack(side="left", padx=T.SPACE_1)
        self.conf.bind("<Return>", lambda e: self._conf_changed())
        self.conf.bind("<FocusOut>", lambda e: self._conf_changed())
        self.detail = tk.Label(self.list_view, text="", bg=T.SURFACE_SUNKEN, fg=T.TEXT, font=T.FONT_SMALL, anchor="nw",
                               justify="left", height=4, wraplength=self._lay["side_px"] - 20, padx=6, pady=4)
        self.detail.pack(fill="x", pady=(T.SPACE_1, T.SPACE_1))
        lf = tk.Frame(self.list_view, bg=T.BG)
        lf.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(lf, columns=("name", "score", "track"), show="headings", style="EW.Treeview",
                                 height=self._lay["list_rows"], selectmode="browse")
        for col, text, w in (("name", "Name", 150), ("score", "Score", 50), ("track", "Track", 80)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=int(w * self.scale), stretch=(col == "name"))
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.tree.yview, style="EW.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._row_selected())
        self.details_txt = self._text_area(self.details_view)
        self.list_view.tkraise()

    def _text_area(self, parent):
        """A steady, read-only, selectable, scrollable text area (bounded content set by _set_text_area)."""
        txt = tk.Text(parent, bg=T.SURFACE_SUNKEN, fg=T.TEXT, font=T.FONT_MONO_SMALL, wrap="word", height=8,
                      relief="flat", borderwidth=0, padx=6, pady=4, state="disabled", cursor="xterm",
                      highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=txt.yview, style="EW.Vertical.TScrollbar")
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        return txt

    @staticmethod
    def _set_text_area(txt, text: str) -> None:
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.insert("1.0", text)
        txt.configure(state="disabled")

    def _build_results(self, f) -> None:
        S = self.scale
        f.grid_columnconfigure(1, weight=1)
        f.grid_rowconfigure(0, weight=1)
        left = tk.Frame(f, bg=T.BG, width=int(430 * S))
        left.grid(row=0, column=0, sticky="ns")
        left.grid_propagate(False)
        left.pack_propagate(False)
        bar = tk.Frame(left, bg=T.BG)
        bar.pack(fill="x")
        self._button(bar, "Back to live test", self.c.show_live, "outline").pack(side="left")
        self._button(bar, "Refresh", self.c.refresh_results).pack(side="left", padx=4)
        self._button(bar, "Open", self._open_selected_result, "primary").pack(side="right")
        tk.Label(left, text="Saved YOLOE tests", bg=T.BG, fg=T.TEXT, font=T.FONT_BODY_BOLD, anchor="w").pack(
            fill="x", pady=(T.SPACE_2, 0))
        self.runs_tree = ttk.Treeview(left, columns=("source", "when", "status", "frames"), show="headings",
                                      style="EW.Treeview", height=8, selectmode="browse")
        for col, text, w in (("source", "Video", 170), ("when", "Started", 110), ("status", "Status", 90),
                             ("frames", "Frames", 50)):
            self.runs_tree.heading(col, text=text)
            self.runs_tree.column(col, width=int(w * S), stretch=(col == "source"))
        self.runs_tree.pack(fill="x")
        self.runs_tree.bind("<Double-1>", lambda e: self._open_selected_result())
        tk.Label(left, text="Frame records", bg=T.BG, fg=T.TEXT, font=T.FONT_BODY_BOLD, anchor="w").pack(
            fill="x", pady=(T.SPACE_2, 0))
        self.rec_tree = ttk.Treeview(left, columns=("seq", "src", "dets", "note"), show="headings", style="EW.Treeview",
                                     height=8, selectmode="browse")
        for col, text, w in (("seq", "#", 50), ("src", "Source frame", 130), ("dets", "Detections", 130),
                             ("note", "", 90)):
            self.rec_tree.heading(col, text=text)
            self.rec_tree.column(col, width=int(w * S), stretch=(col == "dets"))
        self.rec_tree.pack(fill="x")
        self.rec_tree.bind("<<TreeviewSelect>>", lambda e: self._record_selected())
        pg = tk.Frame(left, bg=T.BG)
        pg.pack(fill="x", pady=(T.SPACE_1, 0))
        self._button(pg, "Previous page", lambda: self.c.page(-1)).pack(side="left")
        self._button(pg, "Next page", lambda: self.c.page(1)).pack(side="left", padx=4)
        self.page_lbl = tk.Label(pg, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL, width=16, anchor="e")
        self.page_lbl.pack(side="right")
        right = tk.Frame(f, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        right.grid(row=0, column=1, sticky="nsew", padx=(T.SPACE_3, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        rsw = tk.Frame(right, bg=T.SURFACE, padx=T.SPACE_2)
        rsw.grid(row=0, column=0, sticky="ew")
        self.r_frame_btn = self._button(rsw, "Frame", lambda: self._show_review("frame"))
        self.r_details_btn = self._button(rsw, "Details", lambda: self._show_review("details"))
        self.r_frame_btn.pack(side="left", pady=4)
        self.r_details_btn.pack(side="left", padx=(2, 0), pady=4)
        rbody = tk.Frame(right, bg=T.SURFACE)
        rbody.grid(row=1, column=0, sticky="nsew", padx=T.SPACE_2)
        rbody.grid_rowconfigure(0, weight=1)
        rbody.grid_columnconfigure(0, weight=1)
        self.rcanvas = tk.Canvas(rbody, bg=T.VIDEO_BG, highlightthickness=0)
        self.rcanvas.grid(row=0, column=0, sticky="nsew")
        self.rcanvas.bind("<Configure>", lambda e: setattr(self, "_review_key", None))
        self.rdetails_view = tk.Frame(rbody, bg=T.SURFACE)
        self.rdetails_view.grid(row=0, column=0, sticky="nsew")
        self.rdetails_txt = self._text_area(self.rdetails_view)
        info = tk.Frame(right, bg=T.SURFACE, height=int(96 * S), padx=T.SPACE_3)
        info.grid(row=2, column=0, sticky="ew")
        info.pack_propagate(False)
        self.rinfo = tk.Label(info, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="nw",
                              justify="left", wraplength=int(700 * S))
        self.rinfo.pack(fill="both", expand=True)

    # =================================================================================================================
    # commands
    # =================================================================================================================
    def choose_video(self) -> None:
        if self.c.busy():
            self.c.say("Stop the current test before choosing another video.", "warning")
            return
        p = filedialog.askopenfilename(parent=self.root, title="Choose a local video for a YOLOE test",
                                       filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts")])
        if p:
            self.c.choose_source(p)

    def _pause_resume(self) -> None:
        if self.c.phase == "paused":
            self.c.resume()
        else:
            self.c.pause()

    def _set_view(self, view: str) -> None:
        self.c.set_view(view)
        self._overlay_key = self._rows_key = None

    def _conf_changed(self) -> None:
        if not self.c.set_shown_conf(self.conf_var.get()):
            self.conf_var.set(f"{self.c.shown_conf:.2f}")
        self._overlay_key = self._rows_key = None

    def _show_side(self, view: str) -> None:
        """Detections list or the Details text in the SAME reserved side area (no popup, no size change)."""
        self._side_view = view
        (self.details_view if view == "details" else self.list_view).tkraise()
        for b, v in ((self.s_dets, "detections"), (self.s_details, "details")):
            b.configure(bg=T.ACCENT_SOFT if v == view else T.SURFACE_ALT, fg=T.ACCENT if v == view else T.TEXT)
        if view == "details":
            self._refresh_details(force=True)

    def _refresh_details(self, force: bool = False) -> None:
        text = self.c.details_text()
        if force or text != self._details_shown:
            self._details_shown = text
            self._set_text_area(self.details_txt, text)

    def _show_review(self, view: str) -> None:
        self._rview = view
        if view == "details":
            self.rdetails_view.tkraise()
        else:
            self.rcanvas.tkraise()
        for b, v in ((self.r_frame_btn, "frame"), (self.r_details_btn, "details")):
            b.configure(bg=T.ACCENT_SOFT if v == view else T.SURFACE_ALT, fg=T.ACCENT if v == view else T.TEXT)

    def _canvas_click(self, e) -> None:
        res = self._painted
        if res is None or self._fit is None:
            return
        pt = canvas_to_source(self._fit, e.x, e.y)
        if pt is None:
            return
        entries = YV.det_entries(res.packet, self.c.view, self.c.shown_conf)
        self.c.select(res.packet, YV.hit(entries, *pt))
        self._overlay_key = self._rows_key = None
        self._publish(res.packet)

    def _row_selected(self) -> None:
        if getattr(self, "_syncing", False) or self._painted is None:
            return
        sel = self.tree.selection()
        if not sel:
            return
        pk = self._painted.packet
        i = int(sel[0])
        if self.c.selection is None or self.c.selection.get("i") != i or self.c.selection.get("key") != pk.key:
            self.c.select(pk, i)
            self._overlay_key = None
            self._publish(pk)

    def _open_selected_result(self) -> None:
        sel = self.runs_tree.selection()
        if sel:
            self.c.open_result(sel[0])
            self._review_key = None

    def _record_selected(self) -> None:
        sel = self.rec_tree.selection()
        if sel:
            self.c.select_record(int(sel[0]), max(320, self.rcanvas.winfo_width()), max(180, self.rcanvas.winfo_height()))
            self._review_key = None
            self._show_review("frame")

    # =================================================================================================================
    # tick
    # =================================================================================================================
    def tick(self, now: float, visible: bool) -> None:
        self.c.pump()
        if self.c.panel != self._panel:
            self._panel = self.c.panel
            (self.results if self._panel == "results" else self.live).tkraise()
        if visible and self._panel == "live":
            self._monitor(now)
        if now - self._chrome_t >= 0.2:
            self._chrome_t = now
            self._chrome(now, visible)

    def _canvas_size(self) -> tuple[int, int]:
        return max(64, self.canvas.winfo_width() - 2), max(36, self.canvas.winfo_height() - 2)

    def _invalidate(self) -> None:
        self.schedule.submitted_key = None
        self._overlay_key = None
        self._placeholder = None

    def _monitor(self, now: float) -> None:
        st = self.c.session.state()
        if self._source_gen != self.c.source_gen:          # another video: attempt-specific presentation goes
            self._source_gen = self.c.source_gen
            self._clear()
        cw, ch = self._canvas_size()
        pk = self.c.session.latest_packet()
        gen = ("yoloe", pk.run_id, pk.epoch) if pk is not None else None
        want = PrepRequest((gen, pk.seq, cw, ch), gen, pk.seq, pk, pk.image, cw, ch) if pk is not None else None
        res = self.schedule.tick(want, gen, self.prep.drain())
        if res is not None:
            self._paint(res)
        if self._painted is not None and st["phase"] in ("preparing", "running") and \
                self._painted.packet.run_id != st["run_id"]:
            self._clear()
        if self._painted is None:
            ps = YV.page_state(self.c, st, False)
            if self._placeholder != (cw, ch, ps["title"], ps["text"]):
                self._placeholder = (cw, ch, ps["title"], ps["text"])
                self.canvas.delete("yov", "yph")
                col = {"danger": "#f87171", "warning": "#fbbf24"}.get(ps["tone"], T.TEXT_DIM)
                self.canvas.create_text(cw // 2, ch // 2 - 30, text=ps["title"], fill=col, font=T.FONT_H2,
                                        justify="center", tags=("yph",))
                self.canvas.create_text(cw // 2, ch // 2 + 4, text=ps["text"], fill=T.TEXT_MUTED, font=T.FONT_BODY,
                                        justify="center", anchor="n", width=max(200, cw - 80), tags=("yph",))
            self._blank_counts()
            return
        self._overlay(now, st)

    def _blank_counts(self) -> None:
        """No analysed image on the canvas: counters say so (never zeros that look like an empty result)."""
        if self._counts_blank:
            return
        self._counts_blank = True
        for k, (n, d) in YV.count_texts(None, self.c.shown_conf).items():
            n_lbl, d_lbl = self.tiles[k]
            n_lbl.configure(text=n)
            d_lbl.configure(text=d)

    def _clear(self) -> None:
        if self._img_item is not None:
            self.canvas.delete(self._img_item)
        self._img_item, self._photo, self._painted, self._fit = None, None, None, None
        self.canvas.delete("yov")
        self.schedule.forget_painted()
        self._placeholder = None
        self._rows_key = None
        self._counts_blank = False
        self._sync_tree([])
        if self.detail.cget("text"):
            self.detail.configure(text="")
        if self.frame_lbl.cget("text"):
            self.frame_lbl.configure(text="")

    def _paint(self, res) -> None:
        photo = self._photo_fn(self.root, res.ppm)
        cw, ch = self._canvas_size()
        pk = res.packet
        sw, sh = pk.size
        dw, dh = res.size
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        item = self.canvas.create_image(ox, oy, anchor="nw", image=photo, tags=("yimg",))
        self.canvas.tag_lower(item)
        if self._img_item is not None:
            self.canvas.delete(self._img_item)
        # the previous PhotoImage is released only now that no canvas item shows it
        self._img_item, self._photo = item, photo
        self.canvas.delete("yph")
        self._placeholder = None
        self._fit = CanvasFit(cw, ch, sw, sh, res.scale, float(ox), float(oy))
        self._painted = res
        self._overlay_key = None
        now = time.monotonic()
        if self.c.session.note_painted(pk, now):
            self.rate.add(now)
        self.c.follow(pk)
        self._publish(pk)

    def _publish(self, pk) -> None:
        """List rows, counts and the selected detail of the PAINTED packet (same callback as its image/boxes)."""
        key = (pk.key, self.c.view, self.c.shown_conf, None if self.c.selection is None else
               (self.c.selection["key"], self.c.selection["i"]), self.c.selection_note)
        if key == self._rows_key:
            return
        self._rows_key = key
        self._counts_blank = False
        rows = YV.list_rows(pk, self.c.view, self.c.shown_conf)
        self._sync_tree(rows)
        for k, (n, d) in YV.count_texts(pk, self.c.shown_conf).items():
            n_lbl, d_lbl = self.tiles[k]
            if n_lbl.cget("text") != n:
                n_lbl.configure(text=n)
            if d_lbl.cget("text") != d:
                d_lbl.configure(text=d)
        lines = YV.detail_lines(pk, self.c.selection, self.c.selection_note, self.c.view)
        txt = "\n".join(lines)
        if self.detail.cget("text") != txt:
            self.detail.configure(text=txt)
        pts = pk.media.get("media_pts", pk.media.get("pts"))
        ft = f"frame {pk.seq}" + (f" · {pts:.2f} s" if isinstance(pts, (int, float)) else "")
        if self.frame_lbl.cget("text") != ft:
            self.frame_lbl.configure(text=ft)

    def _sync_tree(self, rows) -> None:
        self._syncing = True
        try:
            want = [iid for iid, _v in rows]
            have = list(self.tree.get_children())
            gone = [x for x in have if x not in set(want)]
            if gone:
                self.tree.delete(*gone)
            have_set = set(have) - set(gone)
            for pos, (iid, values) in enumerate(rows):
                if iid in have_set:
                    if tuple(self.tree.item(iid, "values")) != tuple(values):
                        self.tree.item(iid, values=values)
                    if self.tree.index(iid) != pos:
                        self.tree.move(iid, "", pos)
                else:
                    self.tree.insert("", pos, iid=iid, values=values)
            sel = self.c.selection
            want_sel = (str(sel["i"]),) if sel is not None and self._painted is not None and \
                sel["key"] == self._painted.packet.key and str(sel["i"]) in set(want) else ()
            if tuple(self.tree.selection()) != want_sel:
                self.tree.selection_set(want_sel) if want_sel else self.tree.selection_remove(*self.tree.selection())
        finally:
            self._syncing = False

    def _overlay(self, now: float, st: dict) -> None:
        res = self._painted
        pk = res.packet
        live, banner = YV.authority(pk, st, now)
        sel = self.c.selection
        key = (id(res), live, banner if not live else None, self.c.view, self.c.shown_conf,
               None if sel is None else (sel["key"], sel["i"]))
        if key == self._overlay_key:
            return
        self._overlay_key = key
        cv, f = self.canvas, self._fit
        cv.delete("yov")
        for e in YV.det_entries(pk, self.c.view, self.c.shown_conf):
            if not e["shown"]:
                continue
            x1, y1, x2, y2 = source_box_to_canvas(f, e["xyxy"])
            picked = sel is not None and sel["key"] == pk.key and sel["i"] == e["i"]
            cv.create_rectangle(x1, y1, x2, y2, outline=e["color"], width=3 if picked else 2, dash=e["dash"],
                                tags=("yov",))
            ty = y1 - 9 if y1 - f.offset_y > 18 else y1 + 9
            t = cv.create_text(x1 + 4, ty, text=e["label"], anchor="w", fill="#04131d", font=T.FONT_SMALL_BOLD,
                               tags=("yov",))
            bx = cv.bbox(t)
            if bx:
                r = cv.create_rectangle(bx[0] - 3, bx[1] - 1, bx[2] + 3, bx[3] + 1, fill=e["color"], outline="",
                                        tags=("yov",))
                cv.tag_raise(t, r)
        note = YV.empty_note(pk, self.c.shown_conf)
        if note:
            cv.create_text(f.offset_x + 10, f.offset_y + (40 if not live and banner else 14), anchor="w",
                           fill=T.TEXT, font=T.FONT_SMALL_BOLD, text=note, tags=("yov",))
        if not live and banner:
            cv.create_rectangle(f.offset_x, f.offset_y, f.offset_x + f.display_w, f.offset_y + 26, fill="#3b2a08",
                                outline="", tags=("yov",))
            cv.create_text(f.offset_x + 10, f.offset_y + 13, anchor="w", fill="#fbbf24", font=T.FONT_SMALL_BOLD,
                           text=banner, tags=("yov",))

    # =================================================================================================================
    # chrome (5 Hz)
    # =================================================================================================================
    def _chrome(self, now: float, visible: bool) -> None:
        from .tk_app import set_enabled, set_pill, set_text
        c = self.c
        snap = c.session.snapshot()
        can = c.can()
        ps = YV.page_state(c, snap, self._painted is not None)
        set_pill(self.status, ps["label"], ps["tone"])
        set_text(self.file_lbl, YV.ellipsize(c.source_name or "No video chosen", self._lay["file_chars"] + 4) +
                 (" (checking...)" if c.source_status == "hashing" else ""))
        set_enabled(self.b_choose, can["choose"])
        set_enabled(self.b_start, can["start"])
        set_text(self.b_start, "Run again" if c.finished is not None else "Start")
        set_enabled(self.b_pause, can["pause"] or can["resume"])
        set_text(self.b_pause, "Resume" if c.phase == "paused" else "Pause")
        set_enabled(self.b_stop, can["stop"])
        for b, v in ((self.v_tracks, VIEW_TRACKS), (self.v_dets, VIEW_DETECTIONS)):
            on = c.view == v
            want_bg = T.ACCENT_SOFT if on else T.SURFACE_ALT
            if b.cget("bg") != want_bg:
                b.configure(bg=want_bg, fg=T.ACCENT if on else T.TEXT)
        if not visible:
            return
        set_text(self.footer, YV.footer(snap, self.rate.fps(now)))
        if self._side_view == "details":
            self._refresh_details()
        set_text(self.fin_title, ps["title"])
        set_text(self.fin_lbl, ps["text"])
        set_enabled(self.f_again, can["run_again"])
        set_enabled(self.f_other, can["choose_another"])
        if self._panel == "results":
            self._render_results()

    # =================================================================================================================
    # results
    # =================================================================================================================
    def _render_results(self) -> None:
        from .tk_app import set_text
        c = self.c
        rows = YV.results_rows(c.results)
        key = tuple(r[0] for r in rows), c.results_status
        if key != getattr(self, "_runs_key", None):
            self._runs_key = key
            self.runs_tree.delete(*self.runs_tree.get_children())
            for iid, values in rows:
                self.runs_tree.insert("", "end", iid=iid, values=values)
        rv = c.review
        if rv is None:
            set_text(self.rinfo, "Open a saved test to page through its frame records. Reviewing never runs the model."
                     if c.results_status != "loading" else "Loading saved tests...")
            return
        if rv.get("details") != self._rdetails_shown:
            self._rdetails_shown = rv.get("details")
            self._set_text_area(self.rdetails_txt, rv.get("details") or "")
            if rv.get("record") is None:
                self._show_review("details")
        rkey = (rv["run_id"], rv.get("page_start"), id(rv.get("rows")))
        if rkey != getattr(self, "_rec_key", None):
            self._rec_key = rkey
            self.rec_tree.delete(*self.rec_tree.get_children())
            for iid, values in YV.record_rows(rv.get("rows") or []):
                self.rec_tree.insert("", "end", iid=iid, values=values)
        n = rv.get("count")
        set_text(self.page_lbl, f"{rv['page_start'] + 1}-{min(n, rv['page_start'] + 50)} of {n}" if n else "")
        m = rv["manifest"]
        oc = m.get("outcome") or {}
        lines = [f"{(m.get('source') or {}).get('name')} · {oc.get('status') or m.get('status')} · "
                 f"{'evidence complete' if oc.get('complete') else 'evidence INCOMPLETE / partial'}",
                 YV.ellipsize(f"Reason: {oc.get('reason') or '-'}", 220),
                 f"Original video: {rv['source'].get('why') or 'verified (same content)'}"]
        rec, fr = rv.get("record"), rv.get("frame")
        if rec is not None:
            md = rec.get("media") or {}
            c_ = rec.get("counts") or {}
            lines.append(f"Frame {rec.get('seq')} · source frame {md.get('src_n', 'unverified')} · "
                         f"{len(rec.get('detections') or [])} detections · {c_.get('associated', 0)} with track ID · "
                         f"{rec.get('disposition')}")
            if fr is not None:
                lines.append("Showing the ORIGINAL frame with its saved boxes" + (
                    " (signature matches)" if fr.matches else f" - WARNING: {fr.error or 'signature differs'}")
                             if fr.image is not None else f"Frame unavailable: {fr.error}")
            elif rv.get("frame_gap"):
                lines.append(f"Frame unavailable: {rv['frame_gap']}")
        set_text(self.rinfo, "\n".join(lines))
        self._paint_review(rv)

    def _paint_review(self, rv) -> None:
        fr, rec = rv.get("frame"), rv.get("record")
        key = (id(fr), id(rec), self.c.view, self.c.shown_conf)
        if key == self._review_key:
            return
        self._review_key = key
        cv = self.rcanvas
        cv.delete("rov")
        if fr is None or fr.image is None or rec is None:
            if self._review_item is not None:
                cv.delete(self._review_item)
                self._review_item, self._review_photo = None, None
            return
        cw, ch = max(64, cv.winfo_width() - 2), max(36, cv.winfo_height() - 2)
        ppm, (dw, dh), sc = to_ppm(fr.image, cw, ch)
        photo = self._photo_fn(self.root, ppm)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        item = cv.create_image(ox, oy, anchor="nw", image=photo)
        cv.tag_lower(item)
        if self._review_item is not None:
            cv.delete(self._review_item)
        self._review_item, self._review_photo = item, photo
        h, w = fr.image.shape[:2]
        fit = CanvasFit(cw, ch, w, h, sc, float(ox), float(oy))
        pk = SimpleNamespace(run_id=rv["run_id"], epoch=rec.get("epoch") or "", detections=rec.get("detections") or [],
                             assoc={int(k): v for k, v in (rec.get("assoc") or {}).items()})
        for e in YV.det_entries(pk, self.c.view, self.c.shown_conf):
            if not e["shown"]:
                continue
            x1, y1, x2, y2 = source_box_to_canvas(fit, e["xyxy"])
            cv.create_rectangle(x1, y1, x2, y2, outline=e["color"], width=2, dash=e["dash"], tags=("rov",))
            cv.create_text(x1 + 4, max(y1 - 9, oy + 8), text=e["label"], anchor="w", fill=e["color"],
                           font=T.FONT_SMALL_BOLD, tags=("rov",))

    def apply_layout(self, w: int, h: int) -> None:
        """Window resize (debounced by the window): side width and list rows follow the layout tier; Details collapses
        automatically only when the tier changes, so the video and the core controls stay usable at 1280x720."""
        lay = YV.layout(w, h, self.scale)
        old, self._lay = self._lay, lay
        if lay == old:
            return
        self.side.configure(width=lay["side_px"])
        self.tree.configure(height=lay["list_rows"])
        self.fin.configure(height=lay["fin_px"])
        for lbl in (self.fin_lbl, self.detail):
            lbl.configure(wraplength=lay["side_px"] - 30)
        self.file_lbl.configure(width=lay["file_chars"])
        if lay["compact"] != old["compact"]:               # deliberate breakpoint: the optional subtitle goes first
            if lay["compact"]:
                self.subtitle.pack_forget()
            else:
                self.subtitle.pack(side="left", padx=(T.SPACE_2, T.SPACE_3), after=self._title_lbl)
        self._invalidate()

    def shutdown(self) -> None:
        self.prep.shutdown(1.0)
