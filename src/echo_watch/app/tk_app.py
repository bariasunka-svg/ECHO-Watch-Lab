# SPDX-License-Identifier: AGPL-3.0-only
"""The ECHO Watch native workbench window (Tk presentation over app.controller.WorkbenchController).

EW-LAB-05: one resizable navy/cyan window with a persistent rail (Display Assets, Library, Runs & Review, Activity,
Settings) and the guided Display Assets workspace: Set up display monitoring in four numbered steps -
Discover (retained setup frame, real Scan, all candidate outlines, thumbnail grid, honest progress/empty/failure/retry),
Select (single-click outline/checkbox selection, separate detail focus, overlap disambiguation, friendly names and
categories, Use saved items, inline new/reuse/replace decisions, Remember & configure N items),
Configure (saved reference-card registry, display name, review allowance, outputs, factual readiness, Save
configuration / Start monitoring) and Monitor (dominant paired scene, summary counts, horizontal saved-reference tray,
selected-item evidence and timeline, Pause/Resume/Stop, finish panel with Review saved run / Reuse this setup / New
setup). A sticky bottom bar holds one primary action and Back. Runs & Review opens saved Review in the same window
(decision list, item timeline, the decision's own clean frame, playback, export); Same item / Undo stays contextual
(from a profile, a saved decision or the correction history).

Layout and interaction patterns adapted from the frozen PatronSight reference (app/live_view.py paint/selection/session
poll patterns, app/visitor_profile.py profile formatting, app/visitor_history_page.py activity paging and corrections,
app/session_review.py saved-review layout, app/settings_workspace.py Pending/Applied/Running) and composed after the
owner's four Display Assets references - see docs/UX_PARITY_MATRIX.md. No donor module is imported; no person/service/
speech/multicamera code is used. Screen content comes from the Tk-free projections in app.views.

Rules kept by this view: one Tk thread; no inference, decoding, hashing, SQLite, export or joins here (they run in
controller tasks, the review LatestWorker or the display-prep LatestWorker); paints coalesce to the newest committed
packet and draw ONLY that packet's own boxes on its own image; tray rows, counts and the inspector during monitoring
are read from the PAINTED packet so a pending repaint never presents newer state against an older image; displayed FPS
counts distinct painted packets only. Constructing this class creates real windows: it is called ONLY from the owner's
manual launch.
"""
from __future__ import annotations

import logging
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

from ..engine.contracts import BOXED_STATES, LABEL, PHASE_ACTIVE, PHASE_IDLE, PHASE_PAUSED
from ..review.frames import DecodeRequest, FrameDecoder, FrameMaps, LatestWorker, PlaybackStream
from . import tk_theme as T
from . import views as V
from .canvas_geometry import CanvasFit, canvas_to_source, fit, source_box_to_canvas
from .controller import CATEGORIES
from .correction_flow import CorrectionFlow
from .display import UiMetrics, prepare, to_ppm
from .presentation import packet_identity, project
from .display_schedule import DisplayScheduler, live_authority, packet_gen, packet_request, setup_gen, setup_request

logger = logging.getLogger(__name__)
TICK_MS = 15
STALE_S = 1.0
NAV = (("yoloe", "YOLOE", "YOLOE"), ("assets", "Display Assets", "Assets"), ("library", "Library", "Library"),
       ("runs", "Runs & Review", "Runs"), ("activity", "Activity", "Activity"), ("settings", "Settings", "Settings"))
PAGE_TITLE = {"yoloe": "YOLOE", "assets": "Display Assets", "library": "Library", "runs": "Runs & Review",
              "activity": "Activity", "corrections": "Same item corrections", "settings": "Settings"}
CHECK, UNCHECK = "☑", "☐"


def _wall(t: float | None) -> str:
    return "-" if not t else datetime.fromtimestamp(t).strftime("%H:%M:%S")


def _photo(root, ppm: bytes):
    try:
        return tk.PhotoImage(master=root, data=ppm, format="PPM")
    except tk.TclError:
        from PIL import Image, ImageTk
        head, rest = ppm.split(b"\n", 1)
        _p, w, h, _m = head.split()
        return ImageTk.PhotoImage(Image.frombytes("RGB", (int(w), int(h)), rest), master=root)


def button(parent, text, command, kind="normal", width=None):
    bg, fg, abg = {"primary": (T.ACCENT, T.ACCENT_TEXT, T.ACCENT_PRESSED), "danger": ("#4a1414", "#fecaca", "#6b1c1c"),
                   "outline": (T.SURFACE, T.ACCENT, T.ACCENT_SOFT)}.get(kind, (T.SURFACE_ALT, T.TEXT, T.BORDER_STRONG))
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
                  disabledforeground=T.TEXT_DIM, relief="flat", bd=0, padx=14, pady=T.CONTROL_PADY // 2 + 2,
                  highlightthickness=1 if kind == "outline" else 0, highlightbackground=T.ACCENT, highlightcolor=T.ACCENT,
                  font=T.FONT_BODY_BOLD if kind == "primary" else T.FONT_BODY, cursor="hand2")
    if width:
        b.configure(width=width)
    return b


def set_enabled(widget, on: bool) -> None:
    want = "normal" if on else "disabled"
    if str(widget.cget("state")) != want:
        widget.configure(state=want)


class ScrollFrame(tk.Frame):
    """A vertically scrollable frame (minimum widths + scrolling instead of clipped controls)."""

    def __init__(self, parent, bg=T.SURFACE, width=None):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0, width=width or 10)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview, style="EW.Vertical.TScrollbar")
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.inner.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.inner.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120), "units")


def card(parent, title: str | None = None, bg=T.SURFACE):
    f = tk.Frame(parent, bg=bg, highlightthickness=1, highlightbackground=T.BORDER, padx=T.SPACE_4, pady=T.SPACE_3)
    if title:
        tk.Label(f, text=title, bg=bg, fg=T.TEXT, font=T.FONT_H2, anchor="w").pack(fill="x", pady=(0, T.SPACE_2))
    return f


def pill(parent, text="", tone="neutral"):
    bg, fg = T.TONES[tone]
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=T.FONT_SMALL_BOLD, padx=10, pady=3)


def set_pill(lbl, text, tone):
    bg, fg = T.TONES.get(tone, T.TONES["neutral"])
    if lbl.cget("text") != text or lbl.cget("bg") != bg:
        lbl.configure(text=text, bg=bg, fg=fg)


def show_image(lbl, img, chars=(12, 5)) -> None:
    """An image label created with a size in TEXT units would clip an image to that many PIXELS (a tiny sliver):
    with an image the label takes the image's own size; without one it keeps its placeholder size in text units."""
    if img:
        lbl.configure(image=img, width=0, height=0)
    else:
        lbl.configure(image="", width=chars[0], height=chars[1])


def set_text(lbl, text, **kw) -> None:
    if lbl.cget("text") != text:
        lbl.configure(text=text, **kw)
    elif kw:
        lbl.configure(**kw)


class CardStrip(tk.Frame):
    """The horizontal saved-reference tray (thumbnail, name, state, last-supported time). It offers the small
    Treeview-like API (insert / item / exists / delete / index / move / get_children / selection*) that the shared
    ``_render_tray`` uses, so the monitoring tray is driven by exactly the same paired rows as before (and as the
    GUI-free sink in the tests)."""

    def __init__(self, parent, on_select, card_px: int = 180, card_h: int = 64):
        super().__init__(parent, bg=T.BG)
        # FIXED sizes (EW-LAB-05 R3): changing card text/images must never change the strip's height - a height change
        # resized the scene canvas about once a second (clock-driven text) and flashed a rescaled image
        self.card_h = card_h
        self.canvas = tk.Canvas(self, bg=T.BG, highlightthickness=0, height=card_h + 4)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.inner = tk.Frame(self.canvas, bg=T.BG)
        self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.configure(xscrollcommand=self.hsb.set)
        self.canvas.pack(fill="x")
        self.hsb.pack(fill="x")
        self.inner.bind("<Configure>", self._fit)
        self.on_select, self.card_px = on_select, card_px
        self._cards: dict[str, dict] = {}
        self._order: list[str] = []
        self._sel: tuple = ()

    def _fit(self, _e=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))      # scroll extent only; the height is fixed

    def _build(self, iid: str) -> dict:
        f = tk.Frame(self.inner, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER, padx=6, pady=4,
                     width=self.card_px, height=self.card_h, cursor="hand2")
        f.grid_propagate(False)                       # the card keeps its size whatever its text or image
        img = tk.Label(f, bg=T.SURFACE_SUNKEN, width=7, height=3)
        img.grid(row=0, column=0, rowspan=3, padx=(0, 6), sticky="nw")
        name = tk.Label(f, bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL_BOLD, anchor="w", justify="left")
        name.grid(row=0, column=1, sticky="w")
        state = tk.Label(f, bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w", justify="left")
        state.grid(row=1, column=1, sticky="w")
        seen = tk.Label(f, bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO, anchor="w", justify="left")
        seen.grid(row=2, column=1, sticky="w")
        for w in (f, img, name, state, seen):
            w.bind("<Button-1>", lambda e, i=iid: self._click(i))
        return {"frame": f, "img": img, "name": name, "state": state, "seen": seen, "text": "", "values": ("", ""),
                "image": "", "tone": "neutral"}

    def _click(self, iid: str) -> None:
        self.selection_set(iid)
        self.on_select(iid)

    def _paint(self, iid: str) -> None:
        c = self._cards[iid]
        c["name"].configure(text=V.ellipsize(c["text"], 20))
        bg, fg = T.TONES.get(c["tone"], T.TONES["neutral"])
        c["state"].configure(text=V.ellipsize(c["values"][0], 24), fg=fg)
        c["seen"].configure(text=f"Last supported {c['values'][1]}")
        show_image(c["img"], c["image"], (7, 3))
        c["frame"].configure(highlightbackground=T.ACCENT if (iid,) == self._sel else T.BORDER,
                             highlightthickness=2 if (iid,) == self._sel else 1)

    def _relayout(self) -> None:
        for i, iid in enumerate(self._order):
            self._cards[iid]["frame"].grid(row=0, column=i, padx=(0, 8), sticky="nsw")

    def get_children(self):
        return tuple(self._order)

    def exists(self, iid) -> bool:
        return iid in self._cards

    def insert(self, parent, index, iid, text, values, image="", tone="neutral"):
        c = self._build(iid)
        c.update(text=text, values=tuple(values), image=image, tone=tone)
        self._cards[iid] = c
        self._order.insert(index, iid)
        self._paint(iid)
        self._relayout()

    def item(self, iid, option=None, **kw):
        c = self._cards[iid]
        if kw:
            for k, v in kw.items():
                c[k] = tuple(v) if k == "values" else v
            self._paint(iid)
            return None
        return c[option]

    def delete(self, *iids):
        for iid in iids:
            c = self._cards.pop(iid, None)
            if c is not None:
                c["frame"].destroy()
                self._order.remove(iid)
        self._relayout()

    def index(self, iid) -> int:
        return self._order.index(iid)

    def move(self, iid, parent, index):
        self._order.remove(iid)
        self._order.insert(index, iid)
        self._relayout()

    def selection(self):
        return self._sel

    def selection_set(self, iid):
        old, self._sel = self._sel, (iid,)
        for x in set(old) | {iid}:
            if x in self._cards:
                self._paint(x)

    def selection_remove(self, *iids):
        old, self._sel = self._sel, ()
        for x in old:
            if x in self._cards:
                self._paint(x)


class Stepper(tk.Canvas):
    """The four numbered steps with complete / current / upcoming states (clickable when allowed)."""

    def __init__(self, parent, on_click):
        super().__init__(parent, bg=T.BG, highlightthickness=0, height=44)
        self.on_click, self._sig, self._hits = on_click, None, []
        self.bind("<Configure>", lambda e: self.render(self._steps if hasattr(self, "_steps") else [], force=True))
        self.bind("<Button-1>", self._click)

    def render(self, steps: list[dict], force: bool = False) -> None:
        self._steps = steps
        w = max(200, self.winfo_width())
        sig = (tuple((s["key"], s["state"], s["enabled"]) for s in steps), w)
        if sig == self._sig and not force:
            return
        self._sig = sig
        self.delete("all")
        self._hits = []
        n = len(steps) or 1
        seg = w / n
        for i, s in enumerate(steps):
            x = 18 + i * seg
            st = s["state"]
            col = T.ACCENT if st in ("complete", "current") else T.TEXT_DIM
            fill = T.ACCENT if st == "current" else T.BG
            self.create_oval(x - 14, 8, x + 14, 36, outline=col, width=2, fill=fill)
            glyph = "✓" if st == "complete" else str(s["n"])
            self.create_text(x, 22, text=glyph, fill=T.ACCENT_TEXT if st == "current" else col, font=T.FONT_BODY_BOLD)
            self.create_text(x + 22, 22, text=s["label"], anchor="w", fill=T.ACCENT if st == "current" else
                             (T.TEXT if st == "complete" else T.TEXT_MUTED),
                             font=T.FONT_BODY_BOLD if st == "current" else T.FONT_BODY)
            if i < n - 1:
                self.create_line(x + 110, 22, x + seg - 24, 22, fill=T.ACCENT if st == "complete" else T.BORDER_STRONG,
                                 width=2)
            self._hits.append((x - 16, x + 100, s))

    def _click(self, e) -> None:
        for x0, x1, s in self._hits:
            if x0 <= e.x <= x1 and s["enabled"]:
                self.on_click(s["key"])
                return


class WorkbenchWindow:
    def __init__(self, root: tk.Tk, controller, metrics_dir: Path, title: str = "ECHO Watch", yoloe=None):
        self.root, self.c = root, controller
        self.yc = yoloe                   # EW-LAB-06 YOLOE workspace controller (None: page shows it is unavailable)
        self.yoloe_page = None
        self.metrics = UiMetrics(metrics_dir)
        self.root.title(title)
        self.root.minsize(960, 560)
        T.install(root)
        try:
            self.scale = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        except tk.TclError:
            self.scale = 1.0
        self.page = "assets"
        self._photo = None
        self._painted = None              # the Prepared currently on the canvas (monitor or setup)
        self._fit = None
        self.prep = LatestWorker("display-prep", prepare)
        self.schedule = DisplayScheduler(self.prep)
        self._thumbs: dict = {}
        self._thumb_wanted: set = set()
        self._thumbs_changed = False
        self._cand_photos: dict = {}      # (scan_id, candidate_id, size) -> PhotoImage
        self._cand_thumb_scan = None
        self._drag = None
        self._draw_mode = None            # None | "manual" | "correct"
        self._hover = None
        self._last_tick = time.monotonic()
        self._last_tray = 0.0
        self._snapshot = None
        self._closing_shown = False
        self._force_armed = False
        self._layout = V.layout(1440, 900, self.scale)
        self._details_open = True
        self._wraps: list = []
        self._view_step = None
        # review state
        self.review = None
        self.review_gen = 0
        self.review_decoder = FrameDecoder()
        self.review_worker = LatestWorker("review-frame", self.review_decoder.decode)
        self.review_photo = None
        self.playback: PlaybackStream | None = None
        self.play_gen = 0
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.request_exit)
        self.root.bind("<Configure>", self._on_root_configure)
        self.root.after(TICK_MS, self._tick)
        self.c.refresh_library()
        self.c.refresh_runs()
        self.c.refresh_setups()
        self.metrics.event("window_built")

    # =================================================================================================================
    # shell
    # =================================================================================================================
    def _wrap(self, lbl):
        self._wraps.append(lbl)
        return lbl

    def _build(self) -> None:
        r = self.root
        r.grid_columnconfigure(1, weight=1)
        r.grid_rowconfigure(0, weight=1)
        nav = tk.Frame(r, bg=T.RAIL, width=self._layout["rail_px"])
        nav.grid(row=0, column=0, sticky="ns")
        nav.grid_propagate(False)
        nav.pack_propagate(False)
        self.nav = nav
        tk.Label(nav, text="ECHO Watch", bg=T.RAIL, fg=T.ACCENT, font=T.FONT_BRAND, anchor="w").pack(fill="x", padx=16,
                                                                                                    pady=(18, 0))
        tk.Label(nav, text="Display assets", bg=T.RAIL, fg=T.TEXT_DIM, font=T.FONT_MICRO, anchor="w").pack(fill="x",
                                                                                                          padx=16,
                                                                                                          pady=(0, 14))
        self.nav_buttons = {}
        for key, text, _short in NAV:
            b = tk.Button(nav, text=text, anchor="w", relief="flat", bd=0, padx=16, pady=10, bg=T.RAIL, fg=T.TEXT_MUTED,
                          activebackground=T.SURFACE_ALT, activeforeground=T.TEXT, font=T.FONT_NAV, cursor="hand2",
                          command=lambda k=key: self.show(k))
            b.pack(fill="x", pady=1, padx=6)
            self.nav_buttons[key] = b
        self.nav_status = tk.Label(nav, text="", bg=T.RAIL, fg=T.TEXT_DIM, font=T.FONT_MICRO, justify="left", anchor="w",
                                   wraplength=self._layout["rail_px"] - 24)
        self.nav_status.pack(side="bottom", fill="x", padx=14, pady=14)
        main = tk.Frame(r, bg=T.BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)
        # compact top context: display name, file, honest replay status, ONE session state; Pause/Stop on every page
        head = tk.Frame(main, bg=T.BG)
        head.grid(row=0, column=0, sticky="ew", padx=T.SPACE_4, pady=(T.SPACE_3, 0))
        self.ctx_name = tk.Label(head, text="", bg=T.BG, fg=T.TEXT, font=T.FONT_BODY_BOLD)
        self.ctx_name.pack(side="left")
        self.ctx_file = tk.Label(head, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL)
        self.ctx_file.pack(side="left", padx=(T.SPACE_2, T.SPACE_3))
        self.src_pill = pill(head, "")
        self.src_pill.pack(side="left")
        self.status_pill = pill(head, "Monitoring not started")
        self.status_pill.pack(side="left", padx=T.SPACE_2)
        self.h_stop = button(head, "Stop", self._header_stop, "danger")
        self.h_pause = button(head, "Pause", self._pause_resume)
        self.h_stop.pack(side="right", padx=2)
        self.h_pause.pack(side="right", padx=2)
        set_enabled(self.h_stop, False)
        set_enabled(self.h_pause, False)
        self.rate_lbl = tk.Label(head, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL, width=34, anchor="e")
        self.rate_lbl.pack(side="right", padx=T.SPACE_2)
        self.pages = tk.Frame(main, bg=T.BG)
        self.pages.grid(row=1, column=0, sticky="nsew", padx=T.SPACE_4, pady=T.SPACE_2)
        self.pages.grid_columnconfigure(0, weight=1)
        self.pages.grid_rowconfigure(0, weight=1)
        # fixed-height message bar: a longer message never changes the height of the pages (and so the scene)
        msgf = tk.Frame(main, bg=T.SURFACE_SUNKEN, height=int(40 * self.scale))
        msgf.grid(row=2, column=0, sticky="ew")
        msgf.pack_propagate(False)
        self.msg = tk.Label(msgf, text="Monitoring has not started. Choose a video to begin.", bg=T.SURFACE_SUNKEN,
                            fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w", justify="left", padx=T.SPACE_4)
        self.msg.pack(fill="both", expand=True)
        self.page_frames = {}
        for key, builder in (("yoloe", self._build_yoloe), ("assets", self._build_assets), ("library", self._build_library),
                             ("runs", self._build_runs),
                             ("activity", self._build_activity), ("corrections", self._build_corrections),
                             ("settings", self._build_settings)):
            f = tk.Frame(self.pages, bg=T.BG)
            f.grid(row=0, column=0, sticky="nsew")
            builder(f)
            self.page_frames[key] = f
        r.grid_propagate(False)                       # EW-LAB-05 R3: content text can never resize the window/scene
        self.show("assets")

    def show(self, key: str) -> None:
        """Navigation never starts, pauses or stops monitoring; it only raises another page."""
        self.page = key
        self.page_frames[key].tkraise()
        for k, b in self.nav_buttons.items():
            on = k == key or (key == "corrections" and k == "library")
            b.configure(bg=T.SURFACE_ALT if on else T.RAIL, fg=T.TEXT if on else T.TEXT_MUTED,
                        font=T.FONT_NAV_ACTIVE if on else T.FONT_NAV)
        if key == "library":
            self.c.refresh_library()
            self.c.refresh_setups()
        elif key == "runs":
            self.c.refresh_runs()
        elif key == "activity":
            self._activity_refresh_filters()
            self.activity_load(0)
        elif key == "corrections":
            self._corr_refresh_profiles()
        elif key == "settings":
            self._settings_render()
        elif key == "yoloe" and self.yc is not None and self.yc.panel == "results":
            self.yc.refresh_results()

    def _build_yoloe(self, f) -> None:
        """EW-LAB-06: the YOLOE workspace (app.yoloe_page). Opening it loads no model and starts nothing."""
        if self.yc is None:
            tk.Label(f, text="The YOLOE workspace is not available in this launch.", bg=T.BG, fg=T.TEXT_MUTED,
                     font=T.FONT_H2).pack(pady=40)
            return
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(0, weight=1)
        inner = tk.Frame(f, bg=T.BG)
        inner.grid(row=0, column=0, sticky="nsew")
        try:
            from .yoloe_page import YoloePage
            self.yoloe_page = YoloePage(self, inner, self.yc)
        except Exception as e:  # noqa: BLE001 - a YOLOE page fault must never take Display Assets down with it
            logger.exception("the YOLOE page could not be built")
            self.yoloe_page = None
            inner.destroy()
            tk.Label(f, text=f"The YOLOE page could not be built ({type(e).__name__}); see the log. Display Assets is "
                             "unaffected.", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_H2, wraplength=600).grid(row=0, column=0)

    def _yoloe_busy(self) -> bool:
        return self.yc is not None and self.yc.busy()

    def _header_stop(self) -> None:
        """Header Stop acts on the mode that holds analysis (reachable from every page)."""
        if self._yoloe_busy():
            self.yc.stop()
        else:
            self.c.stop()

    def _on_root_configure(self, e) -> None:
        if e.widget is not self.root:
            return
        if getattr(self, "_layout_after", None):
            self.root.after_cancel(self._layout_after)
        self._layout_after = self.root.after(120, self._apply_layout)

    def _apply_layout(self) -> None:
        """Reflow by available logical space (app.views.layout): rail, side panel, compact header, collapsible details,
        label wrapping. Text is never shrunk and the action bar never scrolls."""
        self._layout_after = None
        lay = V.layout(self.root.winfo_width(), self.root.winfo_height(), self.scale)
        old, self._layout = self._layout, lay
        self.nav.configure(width=lay["rail_px"])
        self.nav_status.configure(wraplength=max(60, lay["rail_px"] - 24))
        for key, text, short in NAV:
            self.nav_buttons[key].configure(text=short if lay["nav_short"] else text)
        self.side.configure(width=lay["side_px"])
        for lbl in self._wraps:
            try:
                lbl.configure(wraplength=max(120, lay["wrap_px"]))
            except tk.TclError:
                pass
        self.a_title.configure(font=T.FONT_H1_COMPACT if lay["compact_header"] else T.FONT_H1)
        if lay["compact_header"]:
            self.a_sub.grid_remove()
        else:
            self.a_sub.grid()
        if not lay["side_collapsible"]:
            self._details_open = True
        self._place_side()
        if old["grid_cols"] != lay["grid_cols"]:
            self._dg_sig = None
        if old["card_cols"] != lay["card_cols"]:
            self._cfg_sig = None
        self.tray.card_px = lay["tray_card_px"]
        self._under_height()
        if self.yoloe_page is not None:
            self.yoloe_page.apply_layout(self.root.winfo_width(), self.root.winfo_height())

    def _under_height(self) -> None:
        """The area under the scene has a FIXED height that depends only on the step and the layout tier, so changing
        text (clocks, states, messages) can never resize the scene canvas."""
        compact = self._layout["compact_header"]
        if self._view_step == "monitor":
            h = 128 if compact else 204
            if compact:
                self.tl_title.pack_forget()
                self.timeline.pack_forget()
            elif not self.timeline.winfo_ismapped():
                self.tl_title.pack(fill="x", pady=(T.SPACE_2, 0))
                self.timeline.pack(fill="x")
        else:
            h = 44
        self.under.configure(height=int(h * self.scale))

    def _place_side(self) -> None:
        if self._view_step == "configure":
            return
        if self._layout["side_collapsible"] and not self._details_open:
            self.side.grid_remove()
        else:
            self.side.grid()

    def _toggle_details(self) -> None:
        self._details_open = not self._details_open
        self._place_side()

    # =================================================================================================================
    # Display Assets (Discover / Select / Configure / Monitor)
    # =================================================================================================================
    def _build_assets(self, f) -> None:
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(2, weight=1)
        top = tk.Frame(f, bg=T.BG)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        self.a_title = tk.Label(top, text="Set up display monitoring", bg=T.BG, fg=T.TEXT, font=T.FONT_H1, anchor="w")
        self.a_title.grid(row=0, column=0, sticky="w")
        self.a_sub = tk.Label(top, text="Discover items", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SUB, anchor="w")
        self.a_sub.grid(row=1, column=0, sticky="w")
        # monitoring summary tiles (derived from the same painted rows as the tray/inspector/boxes)
        self.tiles = tk.Frame(top, bg=T.BG)
        self.tiles.grid(row=0, column=1, rowspan=2, sticky="e")
        self.tile_lbls = {}
        for key, text, tone in (("monitored", "monitored", "accent"), ("visible", "Visible", "success"),
                                ("not_visible", "Not visible", "danger"), ("other", "other states", "neutral")):
            t = tk.Frame(self.tiles, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER, padx=12, pady=6)
            t.pack(side="left", padx=(T.SPACE_2, 0))
            n = tk.Label(t, text="0", bg=T.SURFACE, fg=T.TONES[tone][1], font=T.FONT_COUNT)
            n.pack(side="left", padx=(0, 6))
            d = tk.Label(t, text=text, bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, justify="left", anchor="w",
                         width=22 if key == "other" else len(text) + 1)
            d.pack(side="left")
            self.tile_lbls[key] = (n, d)
        self.stepper = Stepper(f, self._step_click)
        self.stepper.grid(row=1, column=0, sticky="ew", pady=(T.SPACE_2, T.SPACE_2))
        content = tk.Frame(f, bg=T.BG)
        content.grid(row=2, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)
        self.content = content
        self._build_scene(content)
        self.side = tk.Frame(content, bg=T.BG, width=self._layout["side_px"])
        self.side.grid(row=0, column=1, sticky="nsew", padx=(T.SPACE_4, 0))
        self.side.grid_propagate(False)
        self.side.grid_columnconfigure(0, weight=1)
        self.side.grid_rowconfigure(0, weight=1)
        self.side_panels = {}
        for key, builder in (("discover", self._build_discover_panel), ("select", self._build_select_panel),
                             ("monitor", self._build_monitor_panel), ("finish", self._build_finish_panel)):
            p = tk.Frame(self.side, bg=T.BG)
            p.grid(row=0, column=0, sticky="nsew")
            builder(p)
            self.side_panels[key] = p
        self.cfg = tk.Frame(content, bg=T.BG)
        self._build_configure(self.cfg)
        # sticky bottom action bar: Back + hint + secondary + ONE primary (outside every scroll area)
        bar = tk.Frame(f, bg=T.BG, highlightthickness=1, highlightbackground=T.BORDER, pady=T.SPACE_2)
        bar.grid(row=3, column=0, sticky="ew", pady=(T.SPACE_2, 0))
        bar.grid_columnconfigure(1, weight=1)
        self.ab_back = button(bar, "Back", lambda: self._action("back"))
        self.ab_back.grid(row=0, column=0, padx=(T.SPACE_3, T.SPACE_2))
        self.ab_hint = tk.Label(bar, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="e", justify="right")
        self.ab_hint.grid(row=0, column=1, sticky="ew", padx=T.SPACE_2)
        self.ab_details = button(bar, "Details", self._toggle_details)
        self.ab_sec = [button(bar, "", lambda i=i: self._action_sec(i), "outline") for i in range(2)]
        for i, b in enumerate(self.ab_sec):
            b.grid(row=0, column=3 + i, padx=(0, T.SPACE_2))
        self.ab_primary = button(bar, "Continue", lambda: self._action_primary(), "primary")
        self.ab_primary.grid(row=0, column=5, padx=(0, T.SPACE_3))
        self._ab_keys = {"primary": None, "sec": [None, None]}

    def _build_scene(self, parent) -> None:
        sc = tk.Frame(parent, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        sc.grid(row=0, column=0, sticky="nsew")
        sc.grid_columnconfigure(0, weight=1)
        sc.grid_rowconfigure(1, weight=1)
        self.scene = sc
        hd = tk.Frame(sc, bg=T.SURFACE, padx=T.SPACE_3, pady=T.SPACE_2)
        hd.grid(row=0, column=0, sticky="ew")
        self.scene_name = tk.Label(hd, text="Display 01", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_H2)
        self.scene_name.pack(side="left")
        self.scene_badge = pill(hd, "Setup frame", "accent")
        self.scene_badge.pack(side="left", padx=T.SPACE_2)
        self.scene_tools = tk.Frame(hd, bg=T.SURFACE)
        self.scene_tools.pack(side="right")
        self.b_choose = button(self.scene_tools, "Choose video...", self.choose_video, "outline")
        self.b_adjust = button(self.scene_tools, "Adjust outline", lambda: self._set_draw("correct"))
        self.b_manual = button(self.scene_tools, "Add missed item", lambda: self._set_draw("manual"))
        self.b_health = button(self.scene_tools, "Health details", self._health_details)
        self.scene_time = tk.Label(hd, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, width=9, anchor="e")
        self.scene_time.pack(side="right", padx=T.SPACE_2)
        self.canvas = tk.Canvas(sc, bg=T.VIDEO_BG, highlightthickness=0, cursor="crosshair", takefocus=1)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=T.SPACE_2)
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Motion>", self._canvas_motion)
        self.canvas.bind("<space>", lambda e: self._toggle_focused())
        self.canvas.bind("<Configure>", lambda e: self._invalidate())
        # under the scene: Discover progress | Monitor tray + timeline
        self.under = tk.Frame(sc, bg=T.SURFACE, padx=T.SPACE_3, pady=T.SPACE_2, height=int(44 * self.scale))
        self.under.grid(row=2, column=0, sticky="ew")
        self.under.pack_propagate(False)              # its height is set by step/layout only (see _under_height)
        self.prog_row = tk.Frame(self.under, bg=T.SURFACE)
        self.prog = ttk.Progressbar(self.prog_row, mode="indeterminate", length=220)
        self.prog_lbl = self._wrap(tk.Label(self.prog_row, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL,
                                            anchor="w", justify="left"))
        self.prog_lbl.pack(side="left", fill="x", expand=True)
        self.mon_row = tk.Frame(self.under, bg=T.SURFACE)
        self.tray_title = tk.Label(self.mon_row, text="Saved references", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_BODY_BOLD,
                                   anchor="w")
        self.tray_title.pack(fill="x")
        self.tray = CardStrip(self.mon_row, self._tray_card_click, self._layout["tray_card_px"], int(64 * self.scale))
        self.tray.pack(fill="x", pady=(T.SPACE_1, 0))
        self.tray_empty = tk.Label(self.mon_row, text="No items in this run.", bg=T.SURFACE, fg=T.TEXT_DIM,
                                   font=T.FONT_SMALL, anchor="w")
        self.tray_empty.pack(fill="x")
        self.tl_title = tk.Label(self.mon_row, text="Event timeline", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_BODY_BOLD,
                                 anchor="w")
        self.tl_title.pack(fill="x", pady=(T.SPACE_2, 0))
        self.timeline = tk.Canvas(self.mon_row, bg=T.SURFACE, highlightthickness=0, height=46)
        self.timeline.pack(fill="x")

    # ---- side panels ---------------------------------------------------------------------------------------------------
    def _build_discover_panel(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(2, weight=1)
        c = card(p, "Discovery")
        c.grid(row=0, column=0, sticky="ew")
        self.d_file = tk.Label(c, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_BODY, anchor="w")
        self.d_file.pack(fill="x")
        self.d_status = self._wrap(tk.Label(c, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w",
                                            justify="left"))
        self.d_status.pack(fill="x", pady=(T.SPACE_1, T.SPACE_2))
        row = tk.Frame(c, bg=T.SURFACE)
        row.pack(fill="x")
        self.d_scan = button(row, "Discover items", self.c.scan, "primary")
        self.d_scan.pack(side="left")
        self.d_choose = button(row, "Choose video...", self.choose_video)
        self.d_choose.pack(side="left", padx=T.SPACE_2)
        self.d_setups = tk.Frame(c, bg=T.SURFACE)
        self.d_setups.pack(fill="x", pady=(T.SPACE_2, 0))
        self._d_setups_sig = None
        hdr = tk.Frame(p, bg=T.BG)
        hdr.grid(row=1, column=0, sticky="ew", pady=(T.SPACE_3, T.SPACE_1))
        self.d_count_title = tk.Label(hdr, text="Scan candidates", bg=T.BG, fg=T.TEXT, font=T.FONT_H2, anchor="w")
        self.d_count_title.pack(fill="x")
        self.d_count = tk.Label(hdr, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w")
        self.d_count.pack(fill="x")
        self.d_grid = ScrollFrame(p, bg=T.BG)
        self.d_grid.grid(row=2, column=0, sticky="nsew")
        self.d_empty = self._wrap(tk.Label(self.d_grid.inner, text="", bg=T.BG, fg=T.TEXT_DIM, font=T.FONT_SMALL,
                                           justify="left", anchor="w"))
        self._dg_sig = None
        self._dg_cards: dict = {}
        self.d_add = button(p, "+  Add item manually", lambda: self._set_draw("manual"), "outline")
        self.d_add.grid(row=3, column=0, sticky="ew", pady=(T.SPACE_2, 0))

    def _build_select_panel(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        sp = ScrollFrame(p, bg=T.BG)
        sp.grid(row=0, column=0, sticky="nsew")
        p.grid_rowconfigure(0, weight=1)
        inner = sp.inner
        # overlap decisions: at the TOP of the panel, always visible when a selected suggestion needs one
        ch = tk.Frame(inner, bg=T.SURFACE, highlightthickness=2, highlightbackground="#fbbf24", padx=T.SPACE_4,
                      pady=T.SPACE_3)
        self.s_choice = ch
        self.s_ch_title = tk.Label(ch, text="", bg=T.SURFACE, fg="#fbbf24", font=T.FONT_H2, anchor="w")
        self.s_ch_title.pack(fill="x")
        self.s_ch_text = self._wrap(tk.Label(ch, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL, justify="left",
                                             anchor="w"))
        self.s_ch_text.pack(fill="x", pady=(T.SPACE_1, T.SPACE_2))
        self.s_ch_all_use = button(ch, "Use saved items for all", lambda: self._choices_all("reuse"), "primary")
        self.s_ch_all_use.pack(fill="x")
        self.s_ch_all_new = button(ch, "Create separate items for all", lambda: self._choices_all("new"))
        self.s_ch_all_new.pack(fill="x", pady=(T.SPACE_1, 0))
        tk.Label(ch, text="Or decide one at a time:", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO,
                 anchor="w").pack(fill="x", pady=(T.SPACE_2, 0))
        one = tk.Frame(ch, bg=T.SURFACE)
        one.pack(fill="x")
        self.s_ch_use = button(one, "", lambda: self._choice_next("reuse"), "outline")
        self.s_ch_use.pack(side="left", fill="x", expand=True)
        self.s_ch_new = button(one, "", lambda: self._choice_next("new"))
        self.s_ch_new.pack(side="left", fill="x", expand=True, padx=(T.SPACE_1, 0))
        self._choice_next_key = None
        lst = card(inner)
        lst.pack(fill="x")
        self.s_lst_card = lst
        hd = tk.Frame(lst, bg=T.SURFACE)
        hd.pack(fill="x")
        tk.Label(hd, text="Selected items", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_H2).pack(side="left")
        self.s_count = tk.Label(hd, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL)
        self.s_count.pack(side="right")
        self.s_list = ttk.Treeview(lst, columns=("use", "status"), show="tree", height=7, style="EW.Treeview",
                                   selectmode="browse")
        self.s_list.column("#0", width=150, minwidth=110)
        self.s_list.column("use", width=34, minwidth=30, anchor="center")
        self.s_list.column("status", width=110, minwidth=80)
        self.s_list.pack(fill="x", pady=(T.SPACE_1, 0))
        self.s_list.bind("<Button-1>", lambda e: self._list_click(self.s_list, e))
        self.s_list.bind("<<TreeviewSelect>>", lambda e: self._list_focus(self.s_list))
        self.s_list.bind("<space>", lambda e: (self._toggle_focused(), "break")[1])
        self.s_hint = self._wrap(tk.Label(lst, text="Click an outline or its checkbox to select it; click a name to see "
                                          "its details. Space toggles the highlighted suggestion.", bg=T.SURFACE,
                                          fg=T.TEXT_DIM, font=T.FONT_MICRO, justify="left", anchor="w"))
        self.s_hint.pack(fill="x", pady=(T.SPACE_1, 0))
        sv = card(inner)
        sv.pack(fill="x", pady=(T.SPACE_2, 0))
        self.s_saved_title = tk.Label(sv, text="Use saved items", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_H2, anchor="w")
        self.s_saved_title.pack(fill="x")
        self.s_saved = ttk.Treeview(sv, columns=("use", "status"), show="tree", height=3, style="EW.Treeview",
                                    selectmode="browse")
        self.s_saved.column("#0", width=150, minwidth=110)
        self.s_saved.column("use", width=34, minwidth=30, anchor="center")
        self.s_saved.column("status", width=110, minwidth=80)
        self.s_saved.pack(fill="x", pady=(T.SPACE_1, 0))
        self.s_saved.bind("<Button-1>", lambda e: self._list_click(self.s_saved, e))
        self.s_saved.bind("<<TreeviewSelect>>", lambda e: self._list_focus(self.s_saved))
        self.s_saved.bind("<space>", lambda e: (self._toggle_focused(), "break")[1])
        self.s_blocked = self._wrap(tk.Label(sv, text="", bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO, justify="left",
                                             anchor="w"))
        self.s_blocked.pack(fill="x")
        dc = card(inner, "Item details")
        dc.pack(fill="x", pady=(T.SPACE_2, 0))
        self.sd = dc
        top = tk.Frame(dc, bg=T.SURFACE)
        top.pack(fill="x")
        self.sd_thumb = tk.Label(top, bg=T.SURFACE_SUNKEN, width=12, height=5)
        self.sd_thumb.pack(side="left", padx=(0, T.SPACE_2))
        col = tk.Frame(top, bg=T.SURFACE)
        col.pack(side="left", fill="x", expand=True)
        tk.Label(col, text="Name", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO, anchor="w").pack(fill="x")
        self.sd_name = tk.StringVar()
        self.sd_entry = ttk.Entry(col, textvariable=self.sd_name, style="EW.TEntry", width=20)
        self.sd_entry.pack(fill="x")
        self.sd_entry.bind("<Return>", lambda e: self._sd_save_name())
        self.sd_entry.bind("<FocusOut>", lambda e: self._sd_save_name())
        tk.Label(col, text="Suggested category", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO,
                 anchor="w").pack(fill="x", pady=(T.SPACE_1, 0))
        self.sd_cat = ttk.Combobox(col, values=list(CATEGORIES), style="EW.TCombobox", width=18)
        self.sd_cat.pack(fill="x")
        self.sd_cat.bind("<<ComboboxSelected>>", lambda e: self._sd_set_category())
        self.sd_cat.bind("<FocusOut>", lambda e: self._sd_set_category())
        self.sd_detected = self._wrap(tk.Label(dc, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL,
                                               justify="left", anchor="w"))
        self.sd_detected.pack(fill="x", pady=(T.SPACE_2, 0))
        self.sd_status = pill(dc, "")
        self.sd_status.pack(anchor="w", pady=(T.SPACE_1, 0))
        self.sd_cmp = tk.Frame(dc, bg=T.SURFACE_SUNKEN, padx=T.SPACE_2, pady=T.SPACE_2)
        self.sd_cmp_lbl = self._wrap(tk.Label(self.sd_cmp, text="", bg=T.SURFACE_SUNKEN, fg=T.TEXT, font=T.FONT_SMALL,
                                              justify="left", anchor="w"))
        self.sd_cmp_lbl.pack(fill="x")
        self.sd_match = ttk.Combobox(self.sd_cmp, state="readonly", style="EW.TCombobox", width=28)
        crow = tk.Frame(self.sd_cmp, bg=T.SURFACE_SUNKEN)
        crow.pack(fill="x", pady=(T.SPACE_1, 0))
        self.sd_use = button(crow, "Use saved item", lambda: self._sd_disposition("reuse"), "outline")
        self.sd_use.pack(side="left")
        self.sd_new = button(crow, "Create separate item", lambda: self._sd_disposition("new"))
        self.sd_new.pack(side="left", padx=T.SPACE_1)
        self.sd_replace = button(self.sd_cmp, "Replace reference...", self._sd_replace)
        self.sd_replace.pack(anchor="w", pady=(T.SPACE_1, 0))
        self.sd_explain = self._wrap(tk.Label(dc, text="", bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO,
                                              justify="left", anchor="w"))
        self.sd_explain.pack(fill="x", pady=(T.SPACE_2, 0))
        brow = tk.Frame(dc, bg=T.SURFACE)
        brow.pack(fill="x", pady=(T.SPACE_2, 0))
        self.sd_toggle = button(brow, "Remove selection", self._toggle_focused)
        self.sd_toggle.pack(side="left")
        self.sd_adjust = button(brow, "Adjust outline", lambda: self._set_draw("correct"))
        self.sd_adjust.pack(side="left", padx=T.SPACE_1)
        self.s_banner = self._wrap(tk.Label(inner, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL, justify="left",
                                            anchor="w"))
        self.s_banner.pack(fill="x", pady=(T.SPACE_2, 0))
        self._sd_for = None

    def _build_monitor_panel(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        sp = ScrollFrame(p, bg=T.BG)
        sp.grid(row=0, column=0, sticky="nsew")
        p.grid_rowconfigure(0, weight=1)
        self.insp = card(sp.inner)
        self.insp.pack(fill="x")
        self.name_var = tk.StringVar()
        tk.Label(self.insp, textvariable=self.name_var, bg=T.SURFACE, fg=T.TEXT, font=T.FONT_TITLE, anchor="w",
                 justify="left").pack(fill="x")
        self.insp_state = pill(self.insp, "")
        self.insp_state.pack(anchor="w", pady=(T.SPACE_1, T.SPACE_2))
        imgs = tk.Frame(self.insp, bg=T.SURFACE)
        imgs.pack(fill="x")
        imgs.grid_columnconfigure(0, weight=1)
        imgs.grid_columnconfigure(1, weight=1)
        self.insp_thumb = tk.Label(imgs, bg=T.SURFACE_SUNKEN, width=14, height=6)
        self.insp_thumb.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.insp_current = tk.Label(imgs, bg=T.SURFACE_SUNKEN, width=14, height=6, fg=T.TEXT_DIM, font=T.FONT_MICRO,
                                     wraplength=130, justify="center")
        self.insp_current.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.insp_cap_saved = tk.Label(imgs, text="Saved reference", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO,
                                       wraplength=140, justify="left", anchor="w")
        self.insp_cap_saved.grid(row=1, column=0, sticky="w")
        self.insp_cap_cur = tk.Label(imgs, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO, wraplength=140,
                                     justify="left", anchor="w")
        self.insp_cap_cur.grid(row=1, column=1, sticky="w")
        self.insp_text = self._wrap(tk.Label(self.insp, text="Select an item in the scene or the tray.", bg=T.SURFACE,
                                             fg=T.TEXT_MUTED, font=T.FONT_SMALL, justify="left", anchor="w"))
        self.insp_text.pack(fill="x", pady=(T.SPACE_2, 0))
        self.insp_activity = self._wrap(tk.Label(self.insp, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL,
                                                 justify="left", anchor="w"))
        self.insp_activity.pack(fill="x", pady=(T.SPACE_1, 0))
        self.b_review_decision = button(self.insp, "Review decision", self._review_decision, "outline")
        self.b_review_decision.pack(fill="x", pady=(T.SPACE_2, 0))
        row = tk.Frame(self.insp, bg=T.SURFACE)
        row.pack(fill="x", pady=(T.SPACE_1, 0))
        self.b_all_activity = button(row, "View all activity", self._view_all_activity)
        self.b_all_activity.pack(side="left", fill="x", expand=True)
        self.b_same = button(row, "Same item...", self._insp_same)
        self.b_same.pack(side="left", padx=(T.SPACE_1, 0))
        self.insp_review_note = self._wrap(tk.Label(self.insp, text="", bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO,
                                                    justify="left", anchor="w"))
        self.insp_review_note.pack(fill="x", pady=(T.SPACE_1, 0))
        rules = card(sp.inner, "Rules for this run")
        rules.pack(fill="x", pady=(T.SPACE_2, 0))
        self.m_rules = self._wrap(tk.Label(rules, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL,
                                           justify="left", anchor="w"))
        self.m_rules.pack(fill="x")
        self._insp_cur_key = None

    def _build_finish_panel(self, p) -> None:
        p.grid_columnconfigure(0, weight=1)
        c = card(p)
        c.grid(row=0, column=0, sticky="new")
        self.f_title = tk.Label(c, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_TITLE, anchor="w")
        self.f_title.pack(fill="x")
        self.f_outputs = self._wrap(tk.Label(c, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL, justify="left",
                                             anchor="w"))
        self.f_outputs.pack(fill="x", pady=(T.SPACE_2, 0))
        self.f_notes = self._wrap(tk.Label(c, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, justify="left",
                                           anchor="w"))
        self.f_notes.pack(fill="x", pady=(T.SPACE_2, 0))
        tk.Label(c, text="Next: Review saved run, Reuse this setup (same items and reference versions, starts nothing), "
                         "or New setup - in the bar below.", bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO,
                 justify="left", anchor="w", wraplength=280).pack(fill="x", pady=(T.SPACE_2, 0))

    # ---- configure ---------------------------------------------------------------------------------------------------
    def _build_configure(self, f) -> None:
        f.grid_columnconfigure(0, weight=3)
        f.grid_columnconfigure(1, weight=2)
        f.grid_rowconfigure(0, weight=1)
        left = tk.Frame(f, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        hd = tk.Frame(left, bg=T.SURFACE, padx=T.SPACE_4, pady=T.SPACE_3)
        hd.grid(row=0, column=0, sticky="ew")
        tk.Label(hd, text="Saved reference registry", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_TITLE).pack(side="left")
        self.cf_count = tk.Label(hd, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL)
        self.cf_count.pack(side="left", padx=T.SPACE_2)
        self.cf_context = tk.Label(hd, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL)
        self.cf_context.pack(side="right")
        self.cf_grid = ScrollFrame(left, bg=T.SURFACE)
        self.cf_grid.grid(row=1, column=0, sticky="nsew", padx=T.SPACE_3)
        self.cf_line = tk.Label(left, text="", bg=T.SURFACE_SUNKEN, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w",
                                padx=T.SPACE_3, pady=T.SPACE_2)
        self.cf_line.grid(row=2, column=0, sticky="ew", padx=T.SPACE_3, pady=T.SPACE_2)
        self._cfg_sig = None
        self._cfg_cards: dict = {}
        right = ScrollFrame(f, bg=T.BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(T.SPACE_4, 0))
        r = right.inner
        dc = card(r)
        dc.pack(fill="x")
        self.cd_title = tk.Label(dc, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_TITLE, anchor="w")
        self.cd_title.pack(fill="x")
        body = tk.Frame(dc, bg=T.SURFACE)
        body.pack(fill="x", pady=(T.SPACE_2, 0))
        self.cd_thumb = tk.Label(body, bg=T.SURFACE_SUNKEN, width=14, height=6)
        self.cd_thumb.pack(side="left", padx=(0, T.SPACE_3))
        self.cd_rows = tk.Label(body, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL, justify="left",
                                anchor="nw", wraplength=200)
        self.cd_rows.pack(side="left", fill="x", expand=True)
        nrow = tk.Frame(dc, bg=T.SURFACE)
        nrow.pack(fill="x", pady=(T.SPACE_2, 0))
        self.cd_name = tk.StringVar()
        ttk.Entry(nrow, textvariable=self.cd_name, style="EW.TEntry", width=22).pack(side="left", fill="x", expand=True)
        self.cd_rename = button(nrow, "Save name", self._cd_save_name)
        self.cd_rename.pack(side="left", padx=(T.SPACE_1, 0))
        self.cd_remove = button(dc, "Remove from this setup", self._cd_remove)
        self.cd_remove.pack(anchor="w", pady=(T.SPACE_1, 0))
        rc = card(r, "Monitoring rules")
        rc.pack(fill="x", pady=(T.SPACE_2, 0))
        tk.Label(rc, text="Display / setup name", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO, anchor="w").pack(fill="x")
        self.cr_name = tk.StringVar(value=self.c.setup_name)
        e = ttk.Entry(rc, textvariable=self.cr_name, style="EW.TEntry")
        e.pack(fill="x")
        e.bind("<FocusOut>", lambda ev: self.c.set_setup_name(self.cr_name.get()))
        e.bind("<Return>", lambda ev: self.c.set_setup_name(self.cr_name.get()))
        arow = tk.Frame(rc, bg=T.SURFACE)
        arow.pack(fill="x", pady=(T.SPACE_2, 0))
        tk.Label(arow, text="Request review after", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_BODY).pack(side="left")
        self.cr_allow = tk.StringVar(value=str(self.c.pending_allowance))
        sb = tk.Spinbox(arow, from_=1, to=3600, increment=1, textvariable=self.cr_allow, width=6, bg=T.SURFACE_SUNKEN,
                        fg=T.TEXT, insertbackground=T.TEXT, buttonbackground=T.SURFACE_ALT, relief="flat",
                        command=self._cr_allowance)
        sb.pack(side="left", padx=T.SPACE_1)
        sb.bind("<FocusOut>", lambda ev: self._cr_allowance())
        sb.bind("<Return>", lambda ev: self._cr_allowance())
        tk.Label(arow, text="seconds of supported Not visible", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_BODY).pack(side="left")
        self.cr_note = self._wrap(tk.Label(rc, text="", bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO, justify="left",
                                           anchor="w"))
        self.cr_note.pack(fill="x", pady=(T.SPACE_1, 0))
        oc = card(r, "Recording and evidence")
        oc.pack(fill="x", pady=(T.SPACE_2, 0))
        self.co_vars = {k: tk.BooleanVar(value=v) for k, v in self.c.pending.to_dict().items()}
        for k, text in (("clean_source", "Clean recording (unannotated source frames)"),
                        ("record_annotated", "Annotated recording (committed results with their own boxes)"),
                        ("decision_trace", "Decision trace (every decision, candidate and timing)")):
            ttk.Checkbutton(oc, text=text, variable=self.co_vars[k], style="EW.TCheckbutton",
                            command=self._co_changed).pack(anchor="w")
        tk.Label(oc, text="Advanced", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_MICRO, anchor="w").pack(fill="x",
                                                                                                     pady=(T.SPACE_1, 0))
        ttk.Checkbutton(oc, text="Repeat the video when it ends (each loop is a new monitoring epoch)",
                        variable=self.co_vars["repeat"], style="EW.TCheckbutton", command=self._co_changed).pack(anchor="w")
        self.co_running = self._wrap(tk.Label(oc, text="", bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO,
                                              justify="left", anchor="w"))
        self.co_running.pack(fill="x")
        kc = card(r, "Readiness")
        kc.pack(fill="x", pady=(T.SPACE_2, 0))
        self.ck_rows = []
        for _ in range(5):
            lbl = self._wrap(tk.Label(kc, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL, justify="left", anchor="w"))
            lbl.pack(fill="x", pady=1)
            self.ck_rows.append(lbl)
        button(kc, "Health details", self._health_details).pack(anchor="w", pady=(T.SPACE_2, 0))
        self._cfg_sync = None

    # ---- step/page rendering -------------------------------------------------------------------------------------------
    def _render_assets(self) -> None:
        c = self.c
        vs = c.view_step
        if vs != self._view_step:
            self._view_step = vs
            if vs != "select":                              # Add missed item from Discover continues into Select
                self._draw_mode = None
            if vs == "configure":
                self.scene.grid_remove()
                self.side.grid_remove()
                self.cfg.grid(row=0, column=0, columnspan=2, sticky="nsew")
                self._cfg_sync = None
            else:
                self.cfg.grid_remove()
                self.scene.grid()
                self._place_side()
            for w in self.scene_tools.winfo_children():
                w.pack_forget()
            if vs in ("discover",):
                self.b_choose.pack(side="left", padx=2)
            if vs == "select":
                self.b_adjust.pack(side="left", padx=2)
                self.b_manual.pack(side="left", padx=2)
            self.b_health.pack(side="left", padx=2)
            self.prog_row.pack_forget()
            self.mon_row.pack_forget()
            if vs in ("discover", "select"):
                self.prog_row.pack(fill="x")
            elif vs == "monitor":
                self.mon_row.pack(fill="x")
            self._under_height()
            self._overlay_key = None
        t = V.title(c)
        set_text(self.a_title, t["title"])
        set_text(self.a_sub, t["subtitle"])
        self.stepper.render(V.stepper(c))
        if vs == "monitor" and c.finished is None:
            self.tiles.grid()
        else:
            self.tiles.grid_remove()
        set_text(self.scene_name, c.setup_name)
        if vs == "monitor" and c.finished is None:
            set_pill(self.scene_badge, "Monitoring - video replay", "success" if c.phase == PHASE_ACTIVE else "warning")
        else:
            set_pill(self.scene_badge, "Setup frame - not monitoring" if c.setup_frame is not None else "No video",
                     "accent" if c.setup_frame is not None else "neutral")
        panel = {"discover": "discover", "select": "select", "monitor": "finish" if c.finished else "monitor"}.get(vs)
        if panel and getattr(self, "_side_shown", None) != panel:
            self._side_shown = panel
            self.side_panels[panel].tkraise()
        if vs == "discover":
            self._render_discover()
        elif vs == "select":
            self._render_select()
        elif vs == "configure":
            self._render_configure()
        elif vs == "monitor":
            if c.finished is not None:
                self._render_finish()
            else:
                self._render_monitor_side()
        self._render_action_bar()
        if self._layout["side_collapsible"] and vs != "configure":
            self.ab_details.grid(row=0, column=2, padx=(0, T.SPACE_2))
            self.ab_details.configure(text="Hide details" if self._details_open else "Details")
        else:
            self.ab_details.grid_remove()

    def _render_action_bar(self) -> None:
        a = V.action_bar(self.c)
        back = a.get("back")
        if back is None:
            self.ab_back.grid_remove()
        else:
            self.ab_back.grid()
            set_enabled(self.ab_back, back[2])
        set_text(self.ab_hint, V.ellipsize(a.get("hint") or "", 140))
        sec = a.get("secondary") or []
        for i, b in enumerate(self.ab_sec):
            if i < len(sec):
                k, text, en = sec[i]
                self._ab_keys["sec"][i] = k
                b.grid()
                if b.cget("text") != text:
                    b.configure(text=text)
                set_enabled(b, en)
            else:
                self._ab_keys["sec"][i] = None
                b.grid_remove()
        k, text, en = a["primary"]
        self._ab_keys["primary"] = k
        if self.ab_primary.cget("text") != text:
            self.ab_primary.configure(text=text)
        danger = a.get("danger_primary")
        want = ("#4a1414", "#fecaca") if danger else (T.ACCENT, T.ACCENT_TEXT)
        if self.ab_primary.cget("bg") != want[0]:
            self.ab_primary.configure(bg=want[0], fg=want[1], activebackground=want[0])
        set_enabled(self.ab_primary, en)

    def _action_primary(self) -> None:
        self._action(self._ab_keys["primary"])

    def _action_sec(self, i: int) -> None:
        self._action(self._ab_keys["sec"][i])

    def _action(self, key) -> None:
        c = self.c
        fn = {"back": c.back, "continue": lambda: c.go("select"), "remember": self._remember,
              "reconcile": c.reconcile_enrollment, "save": self._save_config, "start": self._start,
              "use_all": lambda: c.resolve_choices("reuse"),
              "pause": c.pause, "resume": c.resume, "stop": c.stop, "review": self._finish_review,
              "reuse": c.reuse_setup, "new": c.new_setup}.get(key)
        if fn is not None:
            fn()
            self._overlay_key = None

    def _step_click(self, key: str) -> None:
        self.c.go(key)

    def _remember(self) -> None:
        self._sd_save_name()
        self.c.remember()

    def _save_config(self) -> None:
        self._cfg_pull()
        self.c.save_configuration()

    def _start(self) -> None:
        self._cfg_pull()
        self.c.start()

    # ---- discover -----------------------------------------------------------------------------------------------------
    def _render_discover(self) -> None:
        d = V.discover(self.c)
        set_text(self.d_file, d["filename"])
        fg = {"danger": "#f87171", "success": "#4ade80", "warning": "#fbbf24", "accent": T.ACCENT}.get(d["tone"], T.TEXT_MUTED)
        set_text(self.d_status, d["status"], fg=fg)
        set_text(self.d_scan, d["primary"][0])
        set_enabled(self.d_scan, d["primary"][1])
        set_enabled(self.d_choose, self.c.stopped)
        set_enabled(self.d_add, self.c.can()["manual"])
        set_text(self.d_count_title, d["count_title"])
        set_text(self.d_count, d["count"])
        self._progress(d["progress"], d["status"] if d["progress"] else (d["frame_caption"] or ""))
        sig = tuple((s["setup_id"], s["text"]) for s in d["saved_setups"])
        if sig != self._d_setups_sig:
            self._d_setups_sig = sig
            for w in self.d_setups.winfo_children():
                w.destroy()
            if d["saved_setups"]:
                tk.Label(self.d_setups, text="Saved setups for this recording", bg=T.SURFACE, fg=T.TEXT_MUTED,
                         font=T.FONT_MICRO, anchor="w").pack(fill="x")
                for s in d["saved_setups"][:4]:
                    row = tk.Frame(self.d_setups, bg=T.SURFACE)
                    row.pack(fill="x", pady=1)
                    tk.Label(row, text=V.ellipsize(s["text"], 34), bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL,
                             anchor="w").pack(side="left", fill="x", expand=True)
                    button(row, "Use setup", lambda i=s["setup_id"]: self.c.load_setup(i)).pack(side="right")
        self._render_discover_grid(d)

    def _progress(self, mode, text) -> None:
        if mode == "indeterminate":
            if not self.prog.winfo_ismapped():
                self.prog.pack(side="left", padx=(0, T.SPACE_2), before=self.prog_lbl)
                self.prog.start(12)
        elif self.prog.winfo_ismapped():
            self.prog.stop()
            self.prog.pack_forget()
        set_text(self.prog_lbl, text)

    def _render_discover_grid(self, d) -> None:
        cols = self._layout["grid_cols"]
        sig = (self.c.scan_id, tuple(x["key"] for x in d["cards"]), cols)
        if sig != self._dg_sig:
            self._dg_sig = sig
            for w in self.d_grid.inner.winfo_children():
                if w is not self.d_empty:
                    w.destroy()
            self._dg_cards = {}
            for i, cd in enumerate(d["cards"]):
                fr = tk.Frame(self.d_grid.inner, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER,
                              cursor="hand2")
                fr.grid(row=i // cols, column=i % cols, padx=3, pady=3, sticky="nsew")
                img = tk.Label(fr, bg=T.SURFACE_SUNKEN, width=16, height=5)
                img.pack(fill="x")
                lbl = tk.Label(fr, text=V.ellipsize(cd["name"], 18), bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL, anchor="w")
                lbl.pack(fill="x", padx=4)
                sub = tk.Label(fr, text=V.ellipsize(cd["label"], 20), bg=T.SURFACE, fg=T.TEXT_DIM, font=T.FONT_MICRO,
                               anchor="w")
                sub.pack(fill="x", padx=4, pady=(0, 3))
                for w in (fr, img, lbl, sub):
                    w.bind("<Button-1>", lambda e, k=cd["key"]: self._focus(k))
                self._dg_cards[cd["key"]] = (fr, img, lbl)
            for k in range(cols):
                self.d_grid.inner.grid_columnconfigure(k, weight=1)
            if d["empty"] or not d["cards"]:
                self.d_empty.grid(row=0, column=0, columnspan=cols, sticky="ew")
            else:
                self.d_empty.grid_forget()
            self._request_cand_thumbs()
        set_text(self.d_empty, "No candidates were found on this frame. Discover again, or use Add item manually to "
                               "outline a missed item." if d["empty"] else
                 ("Candidates appear here when discovery finishes." if not d["cards"] else ""))
        sel = self.c.selected
        for key, (fr, img, lbl) in self._dg_cards.items():
            on = key == sel or key == self._hover
            want = T.ACCENT if on else T.BORDER
            if fr.cget("highlightbackground") != want:
                fr.configure(highlightbackground=want)
            ph = self._cand_photos.get((self.c.scan_id, key[5:], "grid"))
            if ph is not None and not img.cget("image"):
                img.configure(image=ph, width=0, height=0)

    def _request_cand_thumbs(self) -> None:
        c = self.c
        f, scan_id = c.setup_frame, c.scan_id
        if f is None or scan_id is None:
            return
        boxes = [(x.candidate_id, tuple(x.box)) for x in c.candidates]
        key = (scan_id, tuple(boxes))
        if key == self._cand_thumb_scan:
            return
        self._cand_thumb_scan = key
        image = f.image

        def work():
            from ..geometry import crop
            out = {}
            for cid, box in boxes:
                try:
                    cr = crop(image, box)
                    out[(scan_id, cid, "grid")] = to_ppm(cr, 120, 72)[0]
                    out[(scan_id, cid, "list")] = to_ppm(cr, 40, 28)[0]
                    out[(scan_id, cid, "detail")] = to_ppm(cr, 110, 80)[0]
                except Exception:  # noqa: BLE001 - a missing thumbnail is cosmetic; the outline still works
                    pass
            return out
        c.tasks.submit("cand_thumbs", ("cand_thumbs", scan_id), work)

    # ---- select -------------------------------------------------------------------------------------------------------
    def _render_select(self) -> None:
        c = self.c
        s = V.select(c)
        set_text(self.s_count, s["count_text"])
        self._progress(None, "Setup frame - monitoring has not started. " + s["summary"])
        self._sync_list(self.s_list, s["rows"], "list")
        set_text(self.s_saved_title, s["saved_title"])
        self._sync_list(self.s_saved, s["saved"], None)
        self.s_saved.configure(height=max(1, min(4, len(s["saved"]))))
        set_text(self.s_blocked, s["blocked"] or ("" if s["saved"] else "No saved items for this recording yet."))
        if s["banner"]:
            fg = {"danger": "#f87171", "warning": "#fbbf24", "accent": T.ACCENT}.get(s["banner"][1], T.TEXT_MUTED)
            set_text(self.s_banner, s["banner"][0], fg=fg)
        else:
            set_text(self.s_banner, "")
        self._render_select_detail(s["detail"])
        self._render_choices(s["choices"])

    def _render_choices(self, ch) -> None:
        if ch is None:
            self._choice_next_key = None
            if self.s_choice.winfo_ismapped():
                self.s_choice.pack_forget()
            return
        if not self.s_choice.winfo_ismapped():
            self.s_choice.pack(fill="x", pady=(0, T.SPACE_2), before=self.s_lst_card)
        self._choice_next_key = (ch["next_key"], ch["next_saved"])
        set_text(self.s_ch_title, ch["title"])
        set_text(self.s_ch_text, ch["text"])
        set_text(self.s_ch_use, ch["use_text"])
        set_text(self.s_ch_new, ch["new_text"])
        for b in (self.s_ch_all_use, self.s_ch_all_new, self.s_ch_new):
            set_enabled(b, self.c.stopped)
        set_enabled(self.s_ch_use, self.c.stopped and ch["next_saved"] is not None)

    def _choices_all(self, disposition: str) -> None:
        self.c.resolve_choices(disposition)
        self._overlay_key = None

    def _choice_next(self, disposition: str) -> None:
        nk = self._choice_next_key
        if nk is None:
            return
        key, saved = nk
        self._focus(key)
        self.c.set_disposition(key[5:], disposition, saved if disposition == "reuse" else None)
        self._overlay_key = None

    def _sync_list(self, tv, rows, thumb) -> None:
        want = [r["key"] for r in rows]
        for iid in tv.get_children():
            if iid not in want:
                tv.delete(iid)
        for i, r in enumerate(rows):
            vals = (CHECK if r["checked"] else UNCHECK, V.ellipsize(r["status"], 40))
            text = V.ellipsize(r["name"], 26)
            kw = {}
            if thumb:
                ph = self._cand_photos.get((self.c.scan_id, r["key"][5:], thumb))
                if ph is not None:
                    kw["image"] = ph
            if not tv.exists(r["key"]):
                tv.insert("", i, iid=r["key"], text=text, values=vals, tags=("warn",) if r.get("warn") else (), **kw)
            else:
                if tuple(tv.item(r["key"], "values")) != vals or tv.item(r["key"], "text") != text:
                    tv.item(r["key"], text=text, values=vals, tags=("warn",) if r.get("warn") else ())
                if kw and not tv.item(r["key"], "image"):
                    tv.item(r["key"], image=kw["image"])
                if tv.index(r["key"]) != i:
                    tv.move(r["key"], "", i)
        tv.tag_configure("warn", foreground="#fbbf24")
        sel = self.c.selected or ""
        if sel in want:
            if tv.selection() != (sel,):
                tv.selection_set(sel)
                tv.see(sel)
        elif tv.selection():
            tv.selection_remove(*tv.selection())

    def _list_click(self, tv, e):
        row = tv.identify_row(e.y)
        if not row:
            return None
        if tv.identify_column(e.x) == "#1":            # the checkbox column toggles; the name only focuses
            self._focus(row)
            self._toggle_focused()
            return "break"
        return None

    def _list_focus(self, tv) -> None:
        s = tv.selection()
        if s and self.c.selected != s[0]:
            self._focus(s[0])

    def _focus(self, key: str) -> None:
        self._sd_save_name()
        self.c.select(key)
        self._overlay_key = None
        self._inspector_for = None

    def _toggle_focused(self) -> None:
        key = self.c.selected or ""
        if key.startswith("cand:"):
            self.c.toggle_candidate(key[5:])
        elif key.startswith("saved:"):
            self.c.toggle_saved(key[6:])
        self._overlay_key = None

    def _render_select_detail(self, d) -> None:
        key = None if d is None else d["key"]
        if key != self._sd_for:
            self._sd_for = key
            self.sd_name.set("" if d is None else d["name"])
            if d is not None and d["kind"] == "candidate":
                self.sd_cat.set(d["category"])
            img = None
            if d is not None and d["kind"] == "candidate":
                img = self._cand_photos.get((self.c.scan_id, key[5:], "detail"))
            elif d is not None:
                img = self._thumb(d["item_id"], d.get("crop_path"), (110, 80))
            show_image(self.sd_thumb, img, (12, 5))
        if d is None:
            set_text(self.sd_detected, "Select a suggestion (outline or list) or a saved item to see its details.")
            set_pill(self.sd_status, "", "neutral")
            self.sd_cmp.pack_forget()
            set_text(self.sd_explain, "")
            for b in (self.sd_toggle, self.sd_adjust):
                set_enabled(b, False)
            self.sd_entry.state(["disabled"])
            return
        if not self.sd_thumb.cget("image"):
            img = (self._cand_photos.get((self.c.scan_id, key[5:], "detail")) if d["kind"] == "candidate" else
                   self._thumb(d["item_id"], d.get("crop_path"), (110, 80)))
            if img is not None:
                show_image(self.sd_thumb, img, (12, 5))
        if d["kind"] == "candidate":
            self.sd_entry.state(["!disabled"])
            self.sd_cat.state(["!disabled"])
            set_text(self.sd_detected, d["detected"])
            tone = "warning" if d["disposition"] == "choose" else ("accent" if d["checked"] else "neutral")
            set_pill(self.sd_status, d["status"], tone)
            if d["matches"]:
                if not self.sd_cmp.winfo_ismapped():
                    self.sd_cmp.pack(fill="x", pady=(T.SPACE_2, 0), before=self.sd_explain)
                txt = "Overlaps saved " + ("item " + d["matches"][0]["text"] if len(d["matches"]) == 1 else
                                           f"items: {', '.join(m['text'] for m in d['matches'][:3])}")
                set_text(self.sd_cmp_lbl, txt)
                vals = [m["text"] for m in d["matches"]]
                if len(vals) > 1:
                    if tuple(self.sd_match.cget("values") or ()) != tuple(vals):
                        self.sd_match.configure(values=vals)
                        self.sd_match.current(0)
                    if not self.sd_match.winfo_ismapped():
                        self.sd_match.pack(fill="x", pady=(T.SPACE_1, 0), before=self.sd_use.master)
                else:
                    self.sd_match.pack_forget()
                set_enabled(self.sd_use, self.c.stopped)
                set_enabled(self.sd_new, self.c.stopped)
                set_enabled(self.sd_replace, self.c.stopped and not self.c.replace_pending)
            else:
                self.sd_cmp.pack_forget()
            set_text(self.sd_explain, d["explain"])
            set_text(self.sd_toggle, "Remove selection" if d["checked"] else "Select this item")
            set_enabled(self.sd_toggle, self.c.stopped)
            set_enabled(self.sd_adjust, self.c.stopped)
        else:
            self.sd_entry.state(["disabled"])
            self.sd_cat.state(["disabled"])
            set_text(self.sd_detected, d["status"])
            set_pill(self.sd_status, "Used in this selection" if d["checked"] else "Not used", "accent" if d["checked"]
                     else "neutral")
            self.sd_cmp.pack_forget()
            set_text(self.sd_explain, d["explain"])
            set_text(self.sd_toggle, "Stop using" if d["checked"] else "Use this saved item")
            set_enabled(self.sd_toggle, self.c.stopped)
            set_enabled(self.sd_adjust, False)

    def _sd_match_item(self) -> str | None:
        key = self.c.selected or ""
        cnd = self.c.candidate(key[5:]) if key.startswith("cand:") else None
        if cnd is None or not cnd.matches:
            return None
        i = self.sd_match.current() if self.sd_match.winfo_ismapped() else 0
        return cnd.matches[max(0, i)]["item_id"]

    def _sd_disposition(self, disp: str) -> None:
        key = self.c.selected or ""
        if key.startswith("cand:"):
            self.c.set_disposition(key[5:], disp, self._sd_match_item() if disp == "reuse" else None)
            self._overlay_key = None

    def _sd_replace(self) -> None:
        key = self.c.selected or ""
        item = self._sd_match_item()
        if key.startswith("cand:") and item:
            self.c.replace_reference(key[5:], item)

    def _sd_save_name(self) -> None:
        key = self._sd_for or ""
        if key.startswith("cand:"):
            cnd = self.c.candidate(key[5:])
            name = self.sd_name.get().strip()
            if cnd is not None and name and name != cnd.name:
                self.c.rename_candidate(key[5:], name)

    def _sd_set_category(self) -> None:
        key = self._sd_for or ""
        if key.startswith("cand:"):
            self.c.set_category(key[5:], self.sd_cat.get())

    # ---- configure rendering -----------------------------------------------------------------------------------------
    def _render_configure(self) -> None:
        c = self.c
        v = V.configure(c)
        set_text(self.cf_count, v["count_text"])
        set_text(self.cf_context, f"Recording: {V.ellipsize(v['context'], 40)}")
        tone_fg = {"success": "#4ade80", "warning": "#fbbf24", "danger": "#f87171", "accent": T.ACCENT}
        set_text(self.cf_line, v["setup_line"][0], fg=tone_fg.get(v["setup_line"][1], T.TEXT_MUTED))
        cols = self._layout["card_cols"]
        sig = (tuple((x["item_id"], x["name"], x["version"]) for x in v["cards"]), cols)
        if sig != self._cfg_sig:
            self._cfg_sig = sig
            for w in self.cf_grid.inner.winfo_children():
                w.destroy()
            self._cfg_cards = {}
            for i, cd in enumerate(v["cards"]):
                fr = tk.Frame(self.cf_grid.inner, bg=T.SURFACE_ALT, highlightthickness=1, highlightbackground=T.BORDER,
                              padx=8, pady=8, cursor="hand2")
                fr.grid(row=i // cols, column=i % cols, padx=4, pady=4, sticky="nsew")
                img = tk.Label(fr, bg=T.SURFACE_SUNKEN, width=20, height=6)
                img.pack(fill="x")
                tk.Label(fr, text=cd["badge"], bg=T.SURFACE_SUNKEN, fg=T.TEXT, font=T.FONT_MICRO).place(x=6, y=6)
                tk.Label(fr, text=V.ellipsize(cd["name"], 24), bg=T.SURFACE_ALT, fg=T.TEXT, font=T.FONT_BODY_BOLD,
                         anchor="w").pack(fill="x", pady=(T.SPACE_1, 0))
                tk.Label(fr, text=cd["sub"], bg=T.SURFACE_ALT, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w").pack(fill="x")
                br = tk.Frame(fr, bg=T.SURFACE_ALT)
                br.pack(fill="x", pady=(T.SPACE_1, 0))
                pill(br, cd["status"], "accent").pack(side="left")
                tk.Label(br, text=cd["disposition"], bg=T.SURFACE_ALT, fg=T.TEXT_DIM, font=T.FONT_MICRO).pack(side="right")
                for w in [fr, img] + list(fr.winfo_children()):
                    w.bind("<Button-1>", lambda e, k=cd["key"]: self._focus(k))
                self._cfg_cards[cd["key"]] = (fr, img, cd)
            for k in range(cols):
                self.cf_grid.inner.grid_columnconfigure(k, weight=1)
        sel = c.selected
        for key, (fr, img, cd) in self._cfg_cards.items():
            want = T.ACCENT if key == sel else T.BORDER
            if fr.cget("highlightbackground") != want:
                fr.configure(highlightbackground=want, highlightthickness=2 if key == sel else 1)
            if not img.cget("image"):
                ph = self._thumb(cd["item_id"], cd["crop_path"], (180, 108))
                if ph is not None:
                    img.configure(image=ph, width=0, height=0)
        d = v["detail"]
        if d is None and v["cards"]:
            c.select(v["cards"][0]["key"])
        if d is not None:
            if getattr(self, "_cd_for", None) != d["item_id"]:
                self._cd_for = d["item_id"]
                self.cd_name.set(d["name"])
                show_image(self.cd_thumb, None, (14, 6))
            set_text(self.cd_title, d["name"])
            set_text(self.cd_rows, "\n".join(f"{k}: {val}" for k, val in d["rows"]))
            if not self.cd_thumb.cget("image"):
                ph = self._thumb(d["item_id"], d["crop_path"], (120, 90))
                if ph is not None:
                    self.cd_thumb.configure(image=ph, width=0, height=0)
        set_enabled(self.cd_rename, c.stopped and d is not None)
        set_enabled(self.cd_remove, c.stopped and d is not None)
        sync = (id(c.setup), c.setup_status == "saved", c.step, len(c.tray))
        if sync != self._cfg_sync:
            self._cfg_sync = sync
            self.cr_name.set(c.setup_name)
            self.cr_allow.set(str(c.pending_allowance))
            for k, var in self.co_vars.items():
                var.set(getattr(c.pending, k))
        set_text(self.cr_note, v["allowance_note"])
        set_text(self.co_running, v["running_note"] or "Pending edits apply to the next Start; they never change a "
                                                       "running configuration.")
        for lbl, r in zip(self.ck_rows, v["readiness"]):
            glyph, fg = ("✓", "#4ade80") if r["ok"] is True else (("!", "#f87171") if r["ok"] is False else
                                                                        ("•", T.TEXT_MUTED))
            set_text(lbl, f"{glyph}  {r['label']} - {r['detail']}", fg=fg)

    def _cfg_pull(self) -> None:
        if self.c.view_step != "configure":
            return
        self.c.set_setup_name(self.cr_name.get())
        self._cr_allowance()
        self._co_changed()

    def _cr_allowance(self) -> None:
        if not self.c.set_allowance(self.cr_allow.get()):
            self.cr_allow.set(str(self.c.pending_allowance))

    def _co_changed(self) -> None:
        self.c.set_pending(**{k: v.get() for k, v in self.co_vars.items()})

    def _cd_save_name(self) -> None:
        item = getattr(self, "_cd_for", None)
        if item and self.c.stopped and self.cd_name.get().strip():
            self.c.rename_item(item, self.cd_name.get())
            self.c.say("Name saved to the profile (history keeps the earlier name). Save configuration to keep it in "
                       "this setup.", "neutral")

    def _cd_remove(self) -> None:
        item = getattr(self, "_cd_for", None)
        if item and self.c.remove_from_run(item):
            self.c.say("Removed from this setup only; the saved profile and its history are kept.", "neutral")
            self._cd_for = None

    # ---- monitor side / finish ---------------------------------------------------------------------------------------------
    def _render_monitor_side(self) -> None:
        c = self.c
        rs = c.running_setup
        o = c.running_settings or {}
        if rs:
            outs = ", ".join(t for k, t in (("clean_source", "clean"), ("record_annotated", "annotated"),
                                            ("decision_trace", "trace")) if o.get(k)) or "none"
            txt = (f"{rs['name']} v{rs['version']} · {V.plural(len(rs['items']), 'item')}\n"
                   f"Review after {o.get('review_allowance_s')} s of supported Not visible (unknown/paused time excluded)\n"
                   f"Recording: {outs}; repeat {'on' if o.get('repeat') else 'off'}\n"
                   "Boxes follow each item where it is supported; there is no fixed position rule.")
        else:
            txt = "Preparing: references are being acknowledged by the GPU worker." if c.start_pending or \
                c.phase != PHASE_IDLE else ""
        set_text(self.m_rules, txt)
        set_enabled(self.b_review_decision, False)
        set_text(self.insp_review_note, "Decisions of this run are saved when it finalizes; then Review saved run "
                                        "opens them (exact frames, Same item and Undo).")

    def _render_finish(self) -> None:
        f = V.finish(self.c)
        if f is None:
            return
        set_text(self.f_title, f["title"])
        set_text(self.f_outputs, "\n".join(f["outputs"]) + (f"\n{f['setup']}" if f["setup"] else ""))
        set_text(self.f_notes, "\n".join(f["notes"]))

    def _finish_review(self) -> None:
        run_id = self.c.review_saved_run()
        if run_id:
            self.show("runs")

    def _health_details(self) -> None:
        self.show("settings")
        try:
            self.set_cats.selection_clear(0, "end")
            self.set_cats.selection_set(2)
        except tk.TclError:
            pass
        self._settings_render()

    def choose_video(self) -> None:
        if not self.c.stopped:
            self.c.say("Stop monitoring before choosing another video.", "warning")
            return
        p = filedialog.askopenfilename(parent=self.root, title="Choose a local video",
                                       filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts")])
        if p:
            self._painted = None
            self.schedule.forget_painted()
            self.c.choose_source(p)

    def _pause_resume(self) -> None:
        if self._yoloe_busy():
            if self.yc.phase == "paused":
                self.yc.resume()
            else:
                self.yc.pause()
            return
        if self.c.phase == PHASE_PAUSED:
            self.c.resume()
        else:
            self.c.pause()

    def _set_draw(self, mode: str) -> None:
        if not self.c.stopped or self.c.view_step not in ("discover", "select"):
            return
        if self.c.scan_status != "done":
            self.c.say("Discover items first: Add missed item only recovers an item discovery missed.", "warning")
            return
        if mode == "correct" and not (self.c.selected or "").startswith("cand:"):
            self.c.say("Select a suggestion first, then drag its adjusted outline on the frame.", "warning")
            return
        if self.c.view_step == "discover":
            self.c.go("select")
        self._draw_mode = None if self._draw_mode == mode else mode
        if self._draw_mode:
            self.c.say("Drag on the frame to outline the missed item (a reference image, not a monitoring zone)."
                       if mode == "manual" else "Drag on the frame to adjust this item's reference outline.", "neutral")

    # ---- canvas interaction ------------------------------------------------------------------------------------------
    def _to_src(self, x, y):
        return None if self._fit is None else canvas_to_source(self._fit, x, y)

    def _canvas_press(self, e) -> None:
        self.canvas.focus_set()
        vs = self.c.view_step
        if self._draw_mode and self.c.stopped and self.c.setup_frame is not None:
            self._drag = (e.x, e.y)
            self.canvas.delete("drag")
            self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline=T.ACCENT, dash=(4, 3), width=2, tags=("drag",))
            return
        p = self._to_src(e.x, e.y)
        if p is None:
            return
        keys = V.pick_at(list(self._hit_boxes()), p)
        if not keys:
            return
        if vs == "select" and keys[0].startswith("cand:"):
            if e.state & 0x0001 and len(keys) > 1:          # Shift+click cycles focus through overlapping outlines
                cur = self.c.selected
                k = keys[(keys.index(cur) + 1) % len(keys)] if cur in keys else keys[0]
                self._focus(k)
                self.c.say(f"Focused {self._name_of(k)}; Space or its checkbox selects it.", "neutral")
            else:
                self._focus(keys[0])
                self.c.toggle_candidate(keys[0][5:])
                if len(keys) > 1:
                    others = ", ".join(self._name_of(k) for k in keys[1:4])
                    self.c.say(f"{len(keys)} outlines overlap here; the smallest ({self._name_of(keys[0])}) was changed. "
                               f"Others: {others} - Shift+click to cycle, or use the list.", "neutral")
        else:
            self._focus(keys[0])
            self._sync_tree_selection()
        self._invalidate()

    def _name_of(self, key: str) -> str:
        if key.startswith("cand:"):
            c = self.c.candidate(key[5:])
            return c.name if c else key
        return key

    def _canvas_motion(self, e) -> None:
        if self.c.view_step != "discover":
            return
        p = self._to_src(e.x, e.y)
        keys = V.pick_at(list(self._hit_boxes()), p) if p is not None else []
        h = keys[0] if keys else None
        if h != self._hover:
            self._hover = h
            self._overlay_key = None

    def _canvas_drag(self, e) -> None:
        if self._drag is not None:
            x0, y0 = self._drag
            self.canvas.coords("drag", x0, y0, e.x, e.y)

    def _canvas_release(self, e) -> None:
        if self._drag is None:
            return
        x0, y0 = self._drag
        self._drag = None
        self.canvas.delete("drag")
        a, b = self._to_src(min(x0, e.x), min(y0, e.y)), self._to_src(max(x0, e.x), max(y0, e.y))
        mode, self._draw_mode = self._draw_mode, None
        if a is None or b is None:
            self.c.say("Draw inside the video frame.", "warning")
            return
        box = (a[0], a[1], b[0] - a[0], b[1] - a[1])
        if mode == "manual":
            self.c.add_manual_region(box)
        elif mode == "correct" and (self.c.selected or "").startswith("cand:"):
            if self.c.correct_region(self.c.selected[5:], box):
                self.c.say("Outline adjusted for this item (used for its saved reference).", "neutral")
        self._invalidate()

    def _hit_boxes(self):
        pk = self._painted.packet if self._painted is not None else None
        if pk is not None and hasattr(pk, "items"):
            for r in self._tray_rows():             # only boxes that are drawn (live, current, paired) can be clicked
                if r.get("box") is not None and r.get("live") and r.get("state") in BOXED_STATES:
                    yield f"item:{r['item_id']}", r["box"]
        elif self.c.stopped and self.c.view_step in ("discover", "select"):
            for cnd in self.c.candidates:
                yield f"cand:{cnd.candidate_id}", cnd.box

    # ---- tick: the only periodic UI work ------------------------------------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()
        delay = (now - self._last_tick) * 1000 - TICK_MS
        self._last_tick = now
        self.metrics.loop_delay(max(0.0, delay))
        try:
            self.c.pump()
            self._snapshot = self.c.engine.session.snapshot()
            self._monitor_frame()
            if now - self._last_tray >= 0.2:
                self._last_tray = now
                self._render_chrome()
            self._review_tick()
            if self.yoloe_page is not None:
                try:
                    self.yoloe_page.tick(now, self.page == "yoloe")
                except Exception:  # noqa: BLE001 - contained: the Display Assets part of this tick still runs
                    self._yoloe_tick_errors = getattr(self, "_yoloe_tick_errors", 0) + 1
                    if self._yoloe_tick_errors <= 5 or self._yoloe_tick_errors % 500 == 0:
                        logger.exception("YOLOE page tick failed (%d so far)", self._yoloe_tick_errors)
            ana = (self._snapshot.rates or {}).get("analyzed_fps") if self._snapshot else None
            self.metrics.tick({"phase": self.c.phase, "analyzed_fps": ana, "run_id": self._snapshot.run_id if self._snapshot else None,
                               "preparations_submitted": self.schedule.submitted,
                               "preparations_dropped": dict(self.schedule.dropped)})
        except Exception:  # noqa: BLE001 - a UI tick failure is logged and the loop continues
            logger.exception("workbench tick failed")
        if self.c.closed:
            return
        if self.c.closing and not self.c.engine.session.busy and not self._yoloe_session_busy() and \
                not getattr(self, "_shutdown_started", False):
            self._finish_exit()
            return
        self.root.after(TICK_MS, self._tick)

    def _canvas_size(self) -> tuple[int, int]:
        return max(64, self.canvas.winfo_width() - 2), max(36, self.canvas.winfo_height() - 2)

    def _invalidate(self) -> None:
        """Canvas resized: the next tick asks for the current image at the new size (the old image stays usable)."""
        self.schedule.submitted_key = None
        self._overlay_key = None

    def _current_display(self):
        """(current validity generation, wanted preparation) from the session and controller state of THIS tick."""
        cw, ch = self._canvas_size()
        snap = self._snapshot
        if self.c.phase != PHASE_IDLE and snap is not None and snap.run_id:
            gen = packet_gen(snap.run_id, snap.basis_id)
            lp = self.c.engine.latest_result()
            want = packet_request(lp, cw, ch) if lp is not None and lp.run_id == snap.run_id and \
                lp.basis.basis_id == snap.basis_id else None
            return gen, want
        f = self.c.setup_frame
        if f is not None and self.c.phase == PHASE_IDLE:
            return setup_gen(self.c.gen["source"], id(f)), setup_request(self.c.gen["source"], f, cw, ch)
        return None, None

    def _monitor_frame(self) -> None:
        """Latest-completed paired display (app.display_schedule): paint the newest completed, still-current
        preparation (its own pixels with its own packet's boxes), then coalesce the newest wanted preparation."""
        cw, ch = self._canvas_size()
        gen, want = self._current_display()
        res = self.schedule.tick(want, gen, self.prep.drain())
        if res is not None:
            self._paint(res)
        # a setup image of another source (or none at all) must not remain on screen; a monitoring image stays,
        # marked stale/paused by the overlay, until a current one is painted
        if self._painted is not None and (gen is None or (self._painted.gen and self._painted.gen[0] == "setup"
                                                          and self._painted.gen != gen)):
            self._painted = None
            self.schedule.forget_painted()
        if self._painted is None:
            if getattr(self, "_placeholder", None) != (cw, ch):
                self._placeholder = (cw, ch)
                self.canvas.delete("all")
                self.canvas.create_text(cw // 2, ch // 2, text="Choose a video to begin.\nMonitoring is stopped.",
                                        fill=T.TEXT_DIM, font=T.FONT_H2, justify="center", tags=("ph",))
        else:
            self._placeholder = None
            self._overlay()
        self._publish_presentation()

    def _paint(self, res) -> None:
        t0 = time.perf_counter()
        self._photo = _photo(self.root, res.ppm)
        cw, ch = self._canvas_size()
        sw, sh = self._src_size(res)
        dw, dh = res.size
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        # the fit is the PREPARED image's own scale/offset, so boxes land exactly on the pixels that were drawn
        self._fit = CanvasFit(cw, ch, sw, sh, res.scale, float(ox), float(oy))
        self.canvas.delete("img")
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo, tags=("img",))
        self.canvas.tag_lower("img")
        self.canvas.delete("ph")
        self._painted = res
        self._overlay_key = None
        render_ms = (time.perf_counter() - t0) * 1000
        pk = res.packet
        if pk is not None:
            self.metrics.painted_packet(packet_identity(pk), pk.frame_t_mono, render_ms, res.prep_ms)
            set_text(self.scene_time, _wall(pk.frame_t_wall))

    @staticmethod
    def _src_size(res):
        pk = res.packet
        if pk is not None:
            return pk.size
        return res.src_size or res.size

    def _chip(self, x1, y1, text, fg, bg=T.SURFACE) -> None:
        w = 10 + 7 * len(text)
        self.canvas.create_rectangle(x1, y1 - 20, x1 + w, y1, fill=bg, outline=fg, tags=("ov",))
        self.canvas.create_text(x1 + 5, y1 - 10, text=text, fill=fg, anchor="w", font=T.FONT_SMALL_BOLD, tags=("ov",))

    def _overlay(self) -> None:
        res = self._painted
        pk = res.packet
        phase = self.c.phase
        age = None if pk is None else time.monotonic() - pk.frame_t_mono
        snap = self._snapshot
        live = snap is not None and live_authority(pk, snap.run_id, snap.basis_id, phase, time.monotonic(), STALE_S)
        sel = self.c.selected
        vs = self.c.view_step
        key = (id(res), live, sel, vs, self._hover, self.c.scan_status,
               tuple((c.candidate_id, c.checked, c.box, c.name, c.disposition) for c in self.c.candidates)
               if pk is None else None, phase)
        if key == getattr(self, "_overlay_key", None):
            return
        self._overlay_key = key
        self.canvas.delete("ov")
        f = self._fit
        if f is None:
            return
        if pk is not None:
            if live:
                for o in pk.items:
                    if o.state not in BOXED_STATES or o.box is None:
                        continue
                    x, y, w, h = o.box
                    x1, y1, x2, y2 = source_box_to_canvas(f, (x, y, x + w, y + h))
                    picked = sel == f"item:{o.item_id}"
                    col = T.STATE_BOX[o.state]
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=col, width=4 if picked else 2,
                                                 dash=() if o.state == "visible" else (6, 4), tags=("ov",))
                    txt = f"{o.name} · {LABEL[o.state]} ✓" if o.state == "visible" else \
                        f"{o.name} · Checking (tentative)"
                    self._chip(x1, y1, txt, col)
            else:
                why = {PHASE_PAUSED: "PAUSED", "stopping": "FINALIZING", "starting": "PREPARING"}.get(phase, "STALE")
                self.canvas.create_rectangle(f.offset_x, f.offset_y, f.offset_x + f.display_w, f.offset_y + 26,
                                             fill="#3b2a08", outline="", tags=("ov",))
                self.canvas.create_text(f.offset_x + 10, f.offset_y + 13, anchor="w", fill="#fbbf24", font=T.FONT_SMALL_BOLD,
                                        text=f"{why}: last analysed image ({0 if age is None else age:.1f} s old) - no live "
                                             "boxes; this image has no current authority", tags=("ov",))
            return
        if vs in ("discover", "select"):
            for cnd in self.c.candidates:
                x, y, w, h = cnd.box
                x1, y1, x2, y2 = source_box_to_canvas(f, (x, y, x + w, y + h))
                focused = sel == f"cand:{cnd.candidate_id}"
                hover = self._hover == f"cand:{cnd.candidate_id}"
                if vs == "discover":
                    col = T.ACCENT if (focused or hover) else T.OUTLINE_MUTED
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=col, width=2 if (focused or hover) else 1,
                                                 tags=("ov",))
                    if focused or hover:
                        self._chip(x1, y1, V.ellipsize(cnd.name, 24), T.ACCENT)
                    continue
                if cnd.checked:
                    col = "#fbbf24" if cnd.disposition == "choose" else T.ACCENT
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=col, width=3 if focused else 2, tags=("ov",))
                    self.canvas.create_oval(x2 - 9, y1 - 9, x2 + 9, y1 + 9, fill=col, outline="", tags=("ov",))
                    self.canvas.create_text(x2, y1, text="✓" if cnd.disposition != "choose" else "?",
                                            fill=T.ACCENT_TEXT, font=T.FONT_SMALL_BOLD, tags=("ov",))
                    self._chip(x1, y1, V.ellipsize(cnd.name, 24), col)
                else:
                    col = T.TEXT if focused else T.OUTLINE_MUTED
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=col, width=2 if focused else 1, dash=(4, 3),
                                                 tags=("ov",))
                    self.canvas.create_rectangle(x2 - 8, y1 - 8, x2 + 8, y1 + 8, outline=col, fill=T.SURFACE, tags=("ov",))
                    if focused:
                        self._chip(x1, y1, V.ellipsize(cnd.name, 24), col)
        if self.c.scan_status in ("running", "waiting_worker"):
            self.canvas.create_text(f.offset_x + 12, f.offset_y + 14, anchor="w", fill=T.ACCENT, font=T.FONT_H2,
                                    text="Scanning this frame..." if self.c.scan_status == "running" else
                                    "Starting the GPU worker for discovery...", tags=("ov",))
        else:
            self.canvas.create_rectangle(f.offset_x + 6, f.offset_y + 6, f.offset_x + 330, f.offset_y + 28, fill=T.BG,
                                         outline="", tags=("ov",))
            self.canvas.create_text(f.offset_x + 12, f.offset_y + 17, anchor="w", fill=T.TEXT_MUTED, font=T.FONT_SMALL,
                                    text="Setup frame - monitoring has not started" if self.c.finished is None else
                                    "Setup frame - the run has finished; not monitoring", tags=("ov",))

    # ---- chrome (5 Hz) ----------------------------------------------------------------------------------------------------
    def _render_chrome(self) -> None:
        c, snap = self.c, self._snapshot
        h = V.header(c)
        if self.page == "yoloe" and self.yc is not None:
            ys = self.yc.source_status
            set_text(self.ctx_name, "YOLOE test")
            set_text(self.ctx_file, V.ellipsize(self.yc.source_name or "No video chosen", 48))
            set_pill(self.src_pill, *{"ready": ("Local video replay", "accent"), "hashing": ("Checking video", "neutral"),
                                      "failed": ("Video unusable", "danger")}.get(ys, ("No video", "neutral")))
        else:
            set_text(self.ctx_name, h["setup_name"])
            set_text(self.ctx_file, V.ellipsize(h["source_name"], 48))
            set_pill(self.src_pill, *h["source"])
        yoloe_page_owns_header = self.page == "yoloe" and self.yc is not None and not self.c.analysis_busy()
        if self._yoloe_busy() or yoloe_page_owns_header:
            # EW-LAB-06: while a YOLOE test holds analysis the header's ONE session state and Pause/Stop are its own.
            # R1: on the YOLOE page (Display Assets idle) the header describes the YOLOE attempt, e.g. a failed start,
            # instead of Display Assets' unrelated "Monitoring not started"; control routing is unchanged.
            from . import yoloe_views as YV
            yst = YV.status(self.yc, self.yc.session.state())
            ycan = self.yc.can()
            set_pill(self.status_pill, f"YOLOE test · {yst[0]}", yst[1])
            set_enabled(self.h_pause, ycan["pause"] or ycan["resume"])
            set_enabled(self.h_stop, ycan["stop"])
            set_text(self.h_pause, "Resume" if self.yc.phase == "paused" else "Pause")
            set_text(self.h_stop, "Finalizing..." if self.yc.phase == "finalizing" else "Stop")
        else:
            set_pill(self.status_pill, *h["session"])
            set_enabled(self.h_pause, h["show_controls"] and (c.can()["pause"] or c.can()["resume"]))
            set_enabled(self.h_stop, h["show_controls"] and c.can()["stop"])
            set_text(self.h_pause, "Resume" if c.phase == PHASE_PAUSED else "Pause")
            set_text(self.h_stop, "Finalizing..." if c.stop_pending else "Stop")
        ana = (snap.rates or {}).get("analyzed_fps") if snap else None
        rate = f"Analysis {ana:.1f}/s · Display {self.metrics.displayed_fps:.1f}/s" if (ana is not None and c.phase != PHASE_IDLE) else ""
        set_text(self.rate_lbl, rate)
        msgs = self.yc.messages if (self.page == "yoloe" and self.yc is not None and self.yc.messages) else None
        msg = (msgs[-1][0] if msgs else c.messages[-1].text) if (msgs or c.messages) else ""
        tone_fg = {"success": "#4ade80", "warning": "#fbbf24", "danger": "#f87171"}.get(
            (msgs[-1][1] if msgs else c.messages[-1].tone) if (msgs or c.messages) else "", T.TEXT_MUTED)
        msg = V.ellipsize(msg, 260)
        if self.msg.cget("text") != msg:
            self.msg.configure(text=msg, fg=tone_fg, wraplength=max(400, self.root.winfo_width() - 260))
        ns = f"GPU worker: {c.worker_status if c.worker_status != 'ready' else 'ready (' + str((c.worker_receipt or {}).get('device')) + ')'}"
        if self.page == "yoloe" and self.yc is not None:
            ns = self.yc.worker_text()
        set_text(self.nav_status, ns)
        self._render_assets()
        if c.view_step == "monitor" and c.finished is None:
            self._render_tray()
            self._render_inspector()
        if self.page == "library":
            self._render_library()
        elif self.page == "runs":
            self._render_runs()
            if c.review_request:
                self._open_requested_review()
        elif self.page == "settings":
            self._settings_render()
        self._thumbs_changed = False

    def _presentation(self) -> tuple[list[dict], str, tuple]:
        """The paired presentation of THIS callback (app.presentation): observation rows only from the eligible
        painted packet; otherwise explicitly awaiting / lifecycle with no live authority."""
        pk = self._painted.packet if self._painted is not None else None
        return project(self.c.tray_rows(None), pk, self._snapshot, self.c.phase, time.monotonic(), LABEL, STALE_S)

    def _tray_rows(self) -> list[dict]:
        return self._presentation()[0]

    def _publish_presentation(self) -> None:
        """Tray rows and the selected item's observation fields follow the painted image in the SAME callback whenever
        the presented packet, its authority (expiry, cutover) or a row state changes. Unrelated chrome stays at 5 Hz."""
        rows, _auth, key = self._presentation()
        if key != getattr(self, "_published_key", None):
            self._published_key = key
            self._render_tray(rows)
            self._render_inspector(rows)

    def _render_tray(self, rows: list[dict] | None = None) -> None:
        rows = self._tray_rows() if rows is None else rows
        want = [r["item_id"] for r in rows]
        have = list(self.tray.get_children())
        for iid in have:
            if iid not in want:
                self.tray.delete(iid)
        for i, r in enumerate(rows):
            label = r["display_label"]                   # shared presentation wording (app.presentation.display)
            vals = (label, _wall(r["last_supported_wall"]))
            img = self._thumb(r["item_id"], r.get("crop_path"))
            if not self.tray.exists(r["item_id"]):
                kw = {"image": img} if img else {}
                self.tray.insert("", i, iid=r["item_id"], text=r["name"], values=vals, tone=r["display_tone"], **kw)
            else:
                if tuple(self.tray.item(r["item_id"], "values")) != vals or self.tray.item(r["item_id"], "text") != r["name"]:
                    self.tray.item(r["item_id"], text=r["name"], values=vals, tone=r["display_tone"])
                if img and not self.tray.item(r["item_id"], "image"):
                    self.tray.item(r["item_id"], image=img)
                if self.tray.index(r["item_id"]) != i:
                    self.tray.move(r["item_id"], "", i)
        if rows:
            self.tray_empty.pack_forget()
        else:
            self.tray_empty.pack(fill="x")
        self._sync_tree_selection()
        counts = getattr(self, "_render_counts", None)         # summary tiles: the SAME rows, same callback
        if counts is not None:
            counts(rows)

    def _render_counts(self, rows: list[dict]) -> None:
        s = V.monitor_summary(rows)
        for key in ("monitored", "visible", "not_visible", "other"):
            n, d = self.tile_lbls[key]
            set_text(n, str(s[key]))
        set_text(self.tile_lbls["other"][1], "other states" if s["other"] == 0 else
                 V.ellipsize("other: " + s["other_text"], 34))
        set_text(self.tray_title, f"Saved references ({len(rows)})")

    def _sync_tree_selection(self) -> None:
        sel = self.c.selected or ""
        if sel.startswith("item:") and self.tray.exists(sel[5:]):
            if self.tray.selection() != (sel[5:],):
                self.tray.selection_set(sel[5:])
        elif self.tray.selection():
            self.tray.selection_remove(*self.tray.selection())

    def _tray_card_click(self, item_id: str) -> None:
        if self.c.selected != f"item:{item_id}":
            self.c.select(f"item:{item_id}")
            self._overlay_key = None
            self._inspector_for = None
            self._published_key = None

    def _thumb(self, item_id: str, crop_path: str | None, size=(56, 40)):
        key = (crop_path, tuple(size))
        if key in self._thumbs:
            return self._thumbs[key]
        if not crop_path or key in self._thumb_wanted:
            return None
        self._thumb_wanted.add(key)

        def load():
            import cv2
            img = cv2.imread(crop_path, cv2.IMREAD_COLOR)
            if img is None:
                raise OSError("reference image missing")
            return key, to_ppm(img, size[0], size[1])[0]
        self.c.tasks.submit("thumb", ("thumb", key), load)
        return None

    def _render_inspector(self, rows: list[dict] | None = None) -> None:
        c = self.c
        sel = c.selected or ""
        if sel.startswith("item:"):
            item_id = sel[5:]
            row = next((r for r in (self._tray_rows() if rows is None else rows) if r["item_id"] == item_id), None)
            prof = c.tray_profiles.get(item_id) or next((p for p in c.library if p["item_id"] == item_id), {})
            if getattr(self, "_inspector_for", None) != sel:
                self._inspector_for = sel
                self.name_var.set((row or prof).get("name", ""))
                img = self._thumb(item_id, prof.get("crop_path"))
                show_image(self.insp_thumb, img, (14, 6))
            if row is None:
                return
            # badge text/tone and time qualification come from the shared presentation authority, never from the
            # historical observation's state alone (a stale Visible is not shown as a live success)
            set_pill(self.insp_state, row["display_label"], row["display_tone"])
            lines = [f"Item {item_id} · reference v{prof.get('ref_version') or prof.get('current_ref_version', 1)}",
                     f"Reason: {row['reason']}" if row["reason"] else "",
                     f"Last supported: {_wall(row['last_supported_wall'])}{row['last_supported_note']}"]
            self._set_lbl(self.insp_text, "\n".join(x for x in lines if x))
            # recent events up to the PRESENTED packet only (never activity newer than the image on screen)
            shown = row.get("packet")
            live = [e for e in c.engine.session.live_events(item_id, 12)
                    if shown is not None and (e.get("packet_id") or "") <= shown[1]][:5] if c.phase != PHASE_IDLE else []
            act = "\n".join(f"{_wall(e['wall'])}  {e['label']}" for e in live) or "No activity in this run yet."
            self._set_lbl(self.insp_activity, "Recent (live, saved at finalization):\n" + act if live else act)
            set_enabled(self.b_all_activity, True)
            extra = getattr(self, "_render_evidence", None)    # images, captions and timeline of the SAME row
            if extra is not None:
                extra(row, prof, live)
            return
        if getattr(self, "_inspector_for", None) is not None:
            self._inspector_for = None
            self.name_var.set("")
            show_image(self.insp_thumb, None, (14, 6))
            set_pill(self.insp_state, "", "neutral")
            self._set_lbl(self.insp_text, "Select an item in the scene or the saved-reference tray.")
            self._set_lbl(self.insp_activity, "")

    def _render_evidence(self, row: dict, prof: dict, events: list[dict]) -> None:
        """Saved reference vs the current supported view - cropped from the PAINTED packet's own image only when this
        row is live and boxed; otherwise an honest caption (historical last-supported time, never a ghost)."""
        pk = self._painted.packet if self._painted is not None else None
        ev = V.evidence(row, prof, pk.frame_t_wall if pk is not None else None)
        set_text(self.insp_cap_saved, ev["saved"])
        img = self._thumb(row["item_id"], prof.get("crop_path"))
        if img is not None and not self.insp_thumb.cget("image"):
            show_image(self.insp_thumb, img, (14, 6))
        key = (packet_identity(pk), row["item_id"], tuple(row["box"]) if ev["current_live"] else None)
        now = time.monotonic()
        # at most once per second (and at once when the selected item changes): a small panel that switched image /
        # no-image on every packet would flicker; its caption always carries the shown image's own time
        due = row["item_id"] != getattr(self, "_insp_cur_item", None) or             now - getattr(self, "_insp_cur_t", -1e9) >= 1.0
        if key != self._insp_cur_key and due:
            self._insp_cur_key, self._insp_cur_item, self._insp_cur_t = key, row["item_id"], now
            set_text(self.insp_cap_cur, ev["current"])
            old = getattr(self, "_insp_cur_photo", None)
            try:
                if ev["current_live"] and pk is not None:
                    from ..geometry import crop
                    new = _photo(self.root, to_ppm(crop(pk.image, row["box"]), 120, 90)[0])
                    show_image(self.insp_current, new, (14, 6))     # the label holds the NEW image first...
                    self.insp_current.configure(text="")
                    self._insp_cur_photo = new                     # ...then the old one may be released
                else:
                    show_image(self.insp_current, None, (14, 6))
                    self.insp_current.configure(text="No current supported view")
                    self._insp_cur_photo = None
            except tk.TclError:
                logger.exception("current-view image could not be shown")
                self._insp_cur_photo = old
                try:
                    show_image(self.insp_current, None, (14, 6))
                    self.insp_current.configure(text="Current view unavailable")
                except tk.TclError:
                    pass
        self._render_timeline(V.timeline(events))

    def _render_timeline(self, items: list[dict]) -> None:
        sig = tuple((x["time"], x["label"]) for x in items)
        if sig == getattr(self, "_tl_sig", None):
            return
        self._tl_sig = sig
        cv = self.timeline
        cv.delete("all")
        w = max(200, cv.winfo_width())
        if not items:
            cv.create_text(8, 22, anchor="w", text="No events for the selected item in this run yet.", fill=T.TEXT_DIM,
                           font=T.FONT_SMALL)
            return
        cv.create_line(20, 14, w - 20, 14, fill=T.BORDER_STRONG, width=2)
        step = (w - 40) / max(1, len(items) - 1) if len(items) > 1 else 0
        col = {"gap_started": T.DANGER, "review_requested": T.DANGER, "recovered": T.SUCCESS,
               "first_confirmed": T.SUCCESS}
        for i, x in enumerate(items):
            cx = 20 + i * step if len(items) > 1 else w / 2
            cv.create_oval(cx - 6, 8, cx + 6, 20, fill=col.get(x["kind"], T.ACCENT), outline="")
            cv.create_text(cx, 30, text=x["time"], fill=T.TEXT, font=T.FONT_MICRO)
            cv.create_text(cx, 42, text=V.ellipsize(x["label"], 28), fill=T.TEXT_MUTED, font=T.FONT_MICRO)

    @staticmethod
    def _set_lbl(lbl, text) -> None:
        if lbl.cget("text") != text:
            lbl.configure(text=text)

    def _view_all_activity(self) -> None:
        sel = self.c.selected or ""
        if sel.startswith("item:"):
            self._activity_item_preset = sel[5:]
            self.show("activity")

    def _insp_same(self) -> None:
        sel = self.c.selected or ""
        if sel.startswith("item:"):
            self._corr_preset = sel[5:]
            self.show("corrections")

    def _review_decision(self) -> None:
        self.c.say("Decisions are saved when the run finalizes; Review saved run then opens this item's decisions.",
                   "neutral")

    # =================================================================================================================
    # Library (saved profiles and saved setups - never the current membership)
    # =================================================================================================================
    def _build_library(self, f) -> None:
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=3)
        f.grid_rowconfigure(3, weight=1)
        tb = tk.Frame(f, bg=T.BG)
        tb.grid(row=0, column=0, sticky="ew", pady=(0, T.SPACE_2))
        tk.Label(tb, text="Saved profiles. Opening the Library never adds anything to the current setup.", bg=T.BG,
                 fg=T.TEXT_MUTED, font=T.FONT_SMALL).pack(side="left")
        self.b_lib_add = button(tb, "Use in current setup", self._lib_add, "primary")
        self.b_lib_add.pack(side="right", padx=2)
        self.b_lib_act = button(tb, "View activity", self._lib_activity)
        self.b_lib_act.pack(side="right", padx=2)
        self.b_lib_same = button(tb, "Same item...", self._lib_same)
        self.b_lib_same.pack(side="right", padx=2)
        self.lib = ttk.Treeview(f, columns=("ver", "scope", "runs", "size", "created"), show="tree headings",
                                style="EW.Treeview", selectmode="browse")
        for col, text, w in (("#0", "Profile", 220), ("ver", "Reference", 110), ("scope", "Recording", 170),
                             ("runs", "Runs", 60), ("size", "Frame", 100), ("created", "Created", 150)):
            self.lib.heading(col, text=text)
            self.lib.column(col, width=w, minwidth=60)
        self.lib.grid(row=1, column=0, sticky="nsew")
        tk.Label(f, text="Saved setups (each version keeps the exact items and reference versions it saved)", bg=T.BG,
                 fg=T.TEXT, font=T.FONT_H2, anchor="w").grid(row=2, column=0, sticky="ew", pady=(T.SPACE_3, T.SPACE_1))
        self.lib_setups = ttk.Treeview(f, columns=("ver", "rec", "items", "saved", "state"), show="headings",
                                       style="EW.Treeview", selectmode="browse", height=5)
        for col, text, w in (("ver", "Name / version", 200), ("rec", "Recording", 200), ("items", "Items", 60),
                             ("saved", "Saved", 140), ("state", "Status", 260)):
            self.lib_setups.heading(col, text=text)
            self.lib_setups.column(col, width=w, minwidth=60)
        self.lib_setups.grid(row=3, column=0, sticky="nsew")
        bb = tk.Frame(f, bg=T.BG)
        bb.grid(row=4, column=0, sticky="ew", pady=(T.SPACE_1, 0))
        button(bb, "Use this setup (same recording)", self._lib_use_setup).pack(side="right")
        self._lib_sig = None
        self._setups_sig = None

    def _render_library(self) -> None:
        sig = tuple((p["item_id"], p["name"], p["current_ref_version"], p["runs"]) for p in self.c.library)
        if sig != self._lib_sig or self._thumbs_changed:
            self._lib_sig = sig
            sel = self.lib.selection()
            self.lib.delete(*self.lib.get_children())
            for p in self.c.library:
                img = self._thumb(p["item_id"], p.get("crop_path"), (40, 28))
                kw = {"image": img} if img else {}
                scope = p.get("source_id") or ""
                rec = ("this recording" if scope and scope == (self.c.source_info or {}).get("scope") else
                       (scope[12:24] + "..." if scope.startswith("file-sha256:") else "unknown (legacy)"))
                self.lib.insert("", "end", iid=p["item_id"], text=p["name"],
                                values=(f"v{p['current_ref_version']} ({p['versions']} total)", rec, p["runs"],
                                        f"{p['frame_width']}x{p['frame_height']}",
                                        datetime.fromtimestamp(p["created_wall"]).strftime("%Y-%m-%d %H:%M")), **kw)
            if sel and self.lib.exists(sel[0]):
                self.lib.selection_set(sel[0])
        ssig = tuple((s["setup_id"], s["version"], s["error"]) for s in self.c.all_setups)
        if ssig != self._setups_sig:
            self._setups_sig = ssig
            self.lib_setups.delete(*self.lib_setups.get_children())
            for s in self.c.all_setups:
                self.lib_setups.insert("", "end", iid=s["setup_id"], values=(
                    f"{s['name']} v{s['version']}" if s["version"] else s["name"], s["source_name"] or "-", s["items"],
                    datetime.fromtimestamp(s["created_wall"]).strftime("%Y-%m-%d %H:%M") if s["created_wall"] else "-",
                    s["error"] or f"{s['versions']} version(s) kept"))

    def _lib_selected(self) -> str | None:
        s = self.lib.selection()
        return s[0] if s else None

    def _lib_add(self) -> None:
        i = self._lib_selected()
        if i and self.c.add_to_run(i):
            self.c.go("configure")
            self.show("assets")

    def _lib_use_setup(self) -> None:
        s = self.lib_setups.selection()
        if s and self.c.load_setup(s[0]):
            self.show("assets")

    def _lib_activity(self) -> None:
        i = self._lib_selected()
        if i:
            self._activity_item_preset = i
            self.show("activity")

    def _lib_same(self) -> None:
        i = self._lib_selected()
        if i:
            self._corr_preset = i
            self.show("corrections")

    # ---- window-owned completions ------------------------------------------------------------------------------------------
    def _on_window_completion(self, c) -> None:
        k = c.kind
        if k == "thumb":
            if c.ok:
                key, ppm = c.value
                self._thumbs[key] = _photo(self.root, ppm)
                self._thumbs_changed = True
                self._lib_sig = None
        elif k == "cand_thumbs":
            if c.ok:
                for key, ppm in c.value.items():
                    self._cand_photos[key] = _photo(self.root, ppm)
                self._sd_for = None
        elif k == "export":
            self.c.say(f"Export written: {c.value['dir']}" if c.ok else f"Export failed: {c.error}",
                       "success" if c.ok else "danger")
        elif k == "review_open":
            if self.review is None or c.token != ("review", self.review["gen"]):
                return
            if not c.ok:
                self._set_lbl(self.rv_info, f"The run could not be opened: {c.error}")
                return
            v = c.value
            self.review.update(loading=False, **v, page=0, rows=[], frame_tok=0, maps=FrameMaps(Path(v["run"]["dir"])))
            names = [f"{i['name'] or i['item_id']}  ({i['item_id']})" for i in v["run"]["items"]]
            self.rv_item.configure(values=names)
            if names:
                self.rv_item.set(names[0])
            self._review_info()
            self._review_fill_decisions()
        elif k == "corr_preview":
            self._corr_on_preview(c)
        elif k == "corr_apply":
            self._corr_on_apply(c)
        elif k == "corr_history":
            self._corr_on_history(c)
        elif k == "activity":
            self._activity_on_page(c)
        elif k == "decision":
            self._review_on_decision(c)

    def _open_requested_review(self) -> None:
        """Review saved run (finish panel): select that run in Runs & Review and open it (no inference starts)."""
        run_id = self.c.review_request
        if not run_id or not self.runs_tv.exists(run_id):
            return
        self.c.review_request = None
        self.runs_tv.selection_set(run_id)
        self.runs_tv.see(run_id)
        self.review_open()

    # =================================================================================================================
    # Runs & Review
    # =================================================================================================================
    def _build_runs(self, f) -> None:
        f.grid_columnconfigure(1, weight=1)
        f.grid_rowconfigure(1, weight=1)
        top = tk.Frame(f, bg=T.BG)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, T.SPACE_2))
        self.runs_tv = ttk.Treeview(top, columns=("when", "items", "outcome", "outputs"), show="headings", height=5,
                                    style="EW.Treeview", selectmode="browse")
        for col, text, w in (("when", "Started", 140), ("items", "Items", 220), ("outcome", "Outcome", 260),
                             ("outputs", "Clean / Annotated / Trace", 260)):
            self.runs_tv.heading(col, text=text)
            self.runs_tv.column(col, width=w, minwidth=80)
        self.runs_tv.pack(side="left", fill="x", expand=True)
        rb = tk.Frame(top, bg=T.BG)
        rb.pack(side="left", padx=T.SPACE_2)
        button(rb, "Open review", self.review_open, "primary").pack(fill="x", pady=1)
        button(rb, "Export CSV", self.review_export).pack(fill="x", pady=1)
        button(rb, "Refresh", self.c.refresh_runs).pack(fill="x", pady=1)
        self._runs_sig = None
        # review: left decisions, centre frame, right inspector
        left = tk.Frame(f, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        left.grid(row=1, column=0, sticky="nsw")
        self.rv_item = ttk.Combobox(left, state="readonly", width=28, style="EW.TCombobox")
        self.rv_item.pack(fill="x", padx=6, pady=6)
        self.rv_item.bind("<<ComboboxSelected>>", lambda e: self._review_fill_decisions())
        self.rv_state = ttk.Combobox(left, state="readonly", width=28, style="EW.TCombobox",
                                     values=["All states"] + [LABEL[s] for s in LABEL])
        self.rv_state.set("All states")
        self.rv_state.pack(fill="x", padx=6)
        self.rv_state.bind("<<ComboboxSelected>>", lambda e: self._review_fill_decisions())
        self.rv_timeline = tk.Canvas(left, height=26, bg=T.SURFACE_SUNKEN, highlightthickness=0)
        self.rv_timeline.pack(fill="x", padx=6, pady=6)
        self.rv_timeline.bind("<Button-1>", self._timeline_click)
        self.rv_list = ttk.Treeview(left, columns=("t", "state"), show="headings", style="EW.Treeview", selectmode="browse",
                                    height=18)
        self.rv_list.heading("t", text="Media s")
        self.rv_list.heading("state", text="State")
        self.rv_list.column("t", width=70, minwidth=60, anchor="e")
        self.rv_list.column("state", width=170, minwidth=120)
        self.rv_list.pack(fill="both", expand=True, padx=6)
        self.rv_list.bind("<<TreeviewSelect>>", lambda e: self._review_pick())
        pg = tk.Frame(left, bg=T.SURFACE)
        pg.pack(fill="x", padx=6, pady=6)
        button(pg, "< Prev", lambda: self._review_page(-1)).pack(side="left")
        self.rv_page_lbl = tk.Label(pg, text="", bg=T.SURFACE, fg=T.TEXT_MUTED, font=T.FONT_SMALL)
        self.rv_page_lbl.pack(side="left", padx=6)
        button(pg, "Next >", lambda: self._review_page(1)).pack(side="left")
        mid = tk.Frame(f, bg=T.BG)
        mid.grid(row=1, column=1, sticky="nsew", padx=T.SPACE_2)
        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=1)
        self.rv_canvas = tk.Canvas(mid, bg=T.VIDEO_BG, highlightthickness=1, highlightbackground=T.BORDER)
        self.rv_canvas.grid(row=0, column=0, sticky="nsew")
        ctl = tk.Frame(mid, bg=T.BG)
        ctl.grid(row=1, column=0, sticky="ew", pady=(T.SPACE_1, 0))
        button(ctl, "< Decision", lambda: self._review_step(-1)).pack(side="left")
        button(ctl, "Decision >", lambda: self._review_step(1)).pack(side="left", padx=2)
        self.b_play = button(ctl, "Play clean video", self._review_play)
        self.b_play.pack(side="left", padx=8)
        self.rv_pos = tk.Label(ctl, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL)
        self.rv_pos.pack(side="left", padx=6)
        right = ScrollFrame(f, bg=T.BG, width=T.RIGHT_RAIL_WIDTH)
        right.grid(row=1, column=2, sticky="ns")
        self.rv_info = tk.Label(right.inner, text="Open a run to review it. Opening a run starts no inference.", bg=T.SURFACE,
                                fg=T.TEXT, font=T.FONT_SMALL, justify="left", anchor="nw", wraplength=330, padx=8, pady=8)
        self.rv_info.pack(fill="x")
        self.rv_detail = tk.Label(right.inner, text="", bg=T.SURFACE_SUNKEN, fg=T.TEXT_MUTED, font=T.FONT_MONO_SMALL,
                                  justify="left", anchor="nw", wraplength=330, padx=8, pady=8)
        self.rv_detail.pack(fill="x", pady=(T.SPACE_2, 0))
        cb = tk.Frame(right.inner, bg=T.BG)
        cb.pack(fill="x", pady=T.SPACE_2)
        button(cb, "Same item...", self._review_same).pack(side="left")

    def _render_runs(self) -> None:
        sig = tuple((r["run_id"], r["phase"]) for r in self.c.runs)
        if sig == self._runs_sig:
            return
        self._runs_sig = sig
        sel = self.runs_tv.selection()
        self.runs_tv.delete(*self.runs_tv.get_children())
        for r in self.c.runs:
            o = r.get("outcome") or {}
            outcome = o.get("kind", r["phase"]) + (f" ({o.get('ended_by')})" if o.get("ended_by") else "")
            if o.get("error"):
                outcome += f": {o['error']}"
            outs = " / ".join(str(o.get(k, "-")) for k in ("clean", "annotated", "trace"))
            self.runs_tv.insert("", "end", iid=r["run_id"], values=(_wall(r["started_wall"]) + " " +
                                                                    datetime.fromtimestamp(r["started_wall"]).strftime("%m-%d"),
                                                                    ", ".join(i["name"] or i["item_id"] for i in r["items"]),
                                                                    outcome, outs))
        if sel and self.runs_tv.exists(sel[0]):
            self.runs_tv.selection_set(sel[0])

    def review_open(self) -> None:
        s = self.runs_tv.selection()
        if not s:
            self.c.say("Select a run first.", "warning")
            return
        run = next((r for r in self.c.runs if r["run_id"] == s[0]), None)
        if run is None or run["phase"] not in ("finalized", "stopped_after_failure", "rolled_back"):
            self.c.say("That run has not finished; review opens finished, partial or failed runs.", "warning")
            return
        self._review_stop_play()
        self.review_gen += 1
        gen = self.review_gen
        from ..review.runs import build_index, item_intervals, run_manifest
        store = self.c.engine.store

        def load():
            idx = build_index(run["run_id"], Path(run["dir"]))
            ivs = {i["item_id"]: item_intervals(store, run["run_id"], i["item_id"]) for i in run["items"]}
            return {"run": run, "index": idx, "manifest": run_manifest(run), "intervals": ivs}
        self.review = {"gen": gen, "loading": True, "run": run}
        self._set_lbl(self.rv_info, f"Loading run {run['run_id']}...")
        self.c.tasks.submit("review_open", ("review", gen), load)

    def review_export(self) -> None:
        s = self.runs_tv.selection()
        run = next((r for r in self.c.runs if s and r["run_id"] == s[0]), None)
        if run is None:
            self.c.say("Select a run to export.", "warning")
            return
        from ..review.export import export_run
        self.c.tasks.submit("export", ("export", run["run_id"]), lambda: export_run(Path(run["dir"])))
        self.c.say("Exporting CSV projections of the decision trace in the background...", "neutral")

    def _review_tick(self) -> None:
        # completions owned by the window (review / thumbs / export / replace ref) are drained from the controller's
        # outbox by the controller; window-specific kinds are re-dispatched here
        for c in getattr(self.c, "_window_completions", []):
            self._on_window_completion(c)
        self.c._window_completions = []
        for tok, res in self.review_worker.drain():
            if self.review is None or tok != self.review.get("frame_tok"):
                continue
            self._review_show_frame(res)
        if self.playback is not None:
            self._review_play_tick()

    def _review_info(self) -> None:
        rv = self.review
        man, idx, run = rv.get("manifest") or {}, rv["index"], rv["run"]
        o = (man.get("outcome") or run.get("outcome") or {})
        outs = man.get("outputs") or {}
        lines = [f"Run {run['run_id']}", f"Source: {Path((run.get('source') or {}).get('path') or '?').name}",
                 f"Outcome: {o.get('kind')} ({o.get('ended_by')})" + (f" - {o.get('error')}" if o.get("error") else ""),
                 f"Committed decisions: {len(idx.rows)}   trace: {idx.status}"]
        for k in ("clean", "annotated", "trace"):
            d = outs.get(k) or {}
            lines.append(f"{k}: {d.get('state', 'not requested')}" + (f", dropped {d.get('frames_dropped') or d.get('dropped')}"
                                                                     if (d.get('frames_dropped') or d.get('dropped')) else ""))
        if idx.problems:
            lines.append("Evidence gaps: " + "; ".join(idx.problems[:3]))
        gate = man.get("admission_gate") or {}
        if gate:
            lines.append(f"Admission gate: {gate.get('descriptor')} M={gate.get('M')} qualified={gate.get('qualified')}")
        self._set_lbl(self.rv_info, "\n".join(lines))
        self._review_timeline()

    def _review_item(self) -> str | None:
        v = self.rv_item.get()
        return v[v.rfind("(") + 1:-1] if "(" in v else None

    def _review_fill_decisions(self) -> None:
        rv = self.review
        if not rv or rv.get("loading"):
            return
        item = self._review_item()
        want = self.rv_state.get()
        inv = {v: k for k, v in LABEL.items()}
        st = inv.get(want)
        rows = [r for r in rv["index"].rows if item in r["items"] and (st is None or r["items"][item]["state"] == st)]
        rv["rows"], rv["page"] = rows, 0
        self._review_render_page()
        self._review_timeline()

    PAGE = 200

    def _review_render_page(self) -> None:
        rv = self.review
        rows, p, item = rv["rows"], rv["page"], self._review_item()
        self.rv_list.delete(*self.rv_list.get_children())
        for r in rows[p * self.PAGE:(p + 1) * self.PAGE]:
            o = r["items"][item]
            t = r.get("media_pts")
            self.rv_list.insert("", "end", iid=str(r["seq"]), values=("-" if t is None else f"{t:.2f}", LABEL.get(o["state"], o["state"])))
        n = max(1, (len(rows) + self.PAGE - 1) // self.PAGE)
        self.rv_page_lbl.configure(text=f"page {p + 1}/{n} ({len(rows)} decisions)")

    def _review_page(self, d: int) -> None:
        rv = self.review
        if not rv or rv.get("loading"):
            return
        n = max(1, (len(rv["rows"]) + self.PAGE - 1) // self.PAGE)
        rv["page"] = max(0, min(n - 1, rv["page"] + d))
        self._review_render_page()

    def _review_timeline(self) -> None:
        rv = self.review
        cv = self.rv_timeline
        cv.delete("all")
        if not rv or rv.get("loading"):
            return
        item = self._review_item()
        ivs = rv["intervals"].get(item) or []
        rows = rv["index"].rows
        if not rows:
            return
        t0, t1 = rows[0]["t_mono"], rows[-1]["t_mono"]
        w = max(10, cv.winfo_width())
        span = max(1e-6, t1 - t0)
        col = {"visible": T.SUCCESS, "checking": T.WARNING, "not_seen": T.DANGER, "uncertain": T.PENDING}
        for iv in ivs:
            x0 = (iv["start_mono"] - t0) / span * w
            x1 = max(x0 + 1, (iv["end_mono"] - t0) / span * w)
            cv.create_rectangle(x0, 2, x1, 24, fill=col.get(iv["state"], T.TEXT_DIM), outline="")
        rv["tl"] = (t0, span, w)

    def _timeline_click(self, e) -> None:
        rv = self.review
        if not rv or "tl" not in rv:
            return
        t0, span, w = rv["tl"]
        t = t0 + e.x / w * span
        rows = rv["rows"] or rv["index"].rows
        if rows:
            best = min(rows, key=lambda r: abs(r["t_mono"] - t))
            self._review_select_seq(best["seq"])

    def _review_select_seq(self, seq: int) -> None:
        rv = self.review
        k = next((i for i, r in enumerate(rv["rows"]) if r["seq"] == seq), None)
        if k is None:
            return
        rv["page"] = k // self.PAGE
        self._review_render_page()
        self.rv_list.selection_set(str(seq))
        self.rv_list.see(str(seq))

    def _review_step(self, d: int) -> None:
        rv = self.review
        if not rv or rv.get("loading") or not rv["rows"]:
            return
        s = self.rv_list.selection()
        cur = int(s[0]) if s else rv["rows"][0]["seq"]
        k = next((i for i, r in enumerate(rv["rows"]) if r["seq"] == cur), 0)
        k = max(0, min(len(rv["rows"]) - 1, k + d))
        self._review_select_seq(rv["rows"][k]["seq"])

    def _review_pick(self) -> None:
        rv = self.review
        s = self.rv_list.selection()
        if not rv or not s:
            return
        self._review_stop_play()
        row = rv["index"].by_seq(int(s[0]))
        if row is None:
            return
        rv["current"] = row
        from ..review.runs import read_decision
        idx, maps = rv["index"], rv["maps"]
        tok = ("decision", rv["gen"], row["seq"])
        rv["decision_tok"] = tok
        self.c.tasks.submit("decision", tok, lambda: (read_decision(idx, row), maps.locate_clean_frame(row["frame"])))

    def _review_on_decision(self, c) -> None:
        rv = self.review
        if not rv or c.token != rv.get("decision_tok"):
            return
        if not c.ok:
            self._set_lbl(self.rv_detail, f"The decision could not be read: {c.error}")
            return
        rec, loc = c.value
        rv["record"], rv["lookup"] = rec, loc
        item = self._review_item()
        o = next((x for x in rec.get("items", []) if x["item_id"] == item), {})
        t = rec.get("timing") or {}
        w = rec.get("worker") or {}
        lines = [f"Decision {rec['seq']} ({rec['packet_id']})  frame {rec['frame'][1]}  media {((rec.get('media') or {}).get('media_pts'))} s",
                 f"State: {o.get('label')}  (prior: {(rec.get('prior_states') or {}).get(item)})",
                 f"Reason: {o.get('reason') or '-'}",
                 f"Appearance {o.get('app')}  admission {o.get('admission')}  detector conf {o.get('conf')}  track {o.get('track_id')}",
                 f"Box: {o.get('box')}", f"Capture->commit {t.get('capture_to_commit_ms')} ms; stages {t.get('stage_ms')}",
                 f"Worker: {w.get('device')} infer {w.get('infer_ms')} ms; model {str(w.get('vp_model_sha256'))[:12]}",
                 f"Basis {rec.get('basis_id')} gen {rec.get('control_generation')}", "Candidates evaluated:"]
        for cand in rec.get("candidates", [])[:12]:
            lines.append(f"  #{cand.get('idx', '-')} {'low-conf ' if cand.get('low_confidence') else ''}conf {cand.get('conf')} "
                         f"app {cand.get('app', {}).get(item)} -> {cand.get('admission', {}).get(item)}"
                         f"{' withheld' if cand.get('withheld') else ''} track {cand.get('track')}")
        for a in (o.get("alternatives") or [])[:6]:
            lines.append(f"  alt: {a.get('why')} (app {a.get('app')}, {a.get('admission')})")
        self._set_lbl(self.rv_detail, "\n".join(lines))
        cw, ch = max(64, self.rv_canvas.winfo_width()), max(36, self.rv_canvas.winfo_height())
        if not loc.ok:
            self.rv_canvas.delete("all")
            self.rv_canvas.create_text(cw // 2, ch // 2, text="EVIDENCE GAP\n" + loc.gap, fill="#fbbf24", font=T.FONT_H2,
                                       justify="center", width=cw - 40)
            self.rv_pos.configure(text="no clean frame for this decision")
            return
        rv["frame_tok"] = rv.get("frame_tok", 0) + 1
        self.review_worker.submit(rv["frame_tok"], DecodeRequest(loc.video, loc.index, loc.fps, cw, ch, loc.entry.get("sig")))
        self.rv_pos.configure(text=f"clean frame {loc.index} (decoding...)")

    def _review_show_frame(self, res) -> None:
        rv = self.review
        cw, ch = max(64, self.rv_canvas.winfo_width()), max(36, self.rv_canvas.winfo_height())
        self.rv_canvas.delete("all")
        if isinstance(res, Exception) or res.image is None:
            self.rv_canvas.create_text(cw // 2, ch // 2, text=f"Frame unavailable: {getattr(res, 'error', res)}",
                                       fill="#f87171", font=T.FONT_H2, width=cw - 40)
            return
        ppm, (dw, dh), sc = to_ppm(res.image, cw, ch)
        self.review_photo = _photo(self.root, ppm)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.rv_canvas.create_image(ox, oy, anchor="nw", image=self.review_photo)
        rec, item = rv.get("record") or {}, self._review_item()
        sw, sh = res.source_size
        f = fit(cw, ch, sw, sh)
        for o in rec.get("items", []):
            if o.get("box") and o["state"] in BOXED_STATES:
                x, y, w, h = o["box"]
                x1, y1, x2, y2 = source_box_to_canvas(f, (x, y, x + w, y + h))
                col = T.STATE_BOX[o["state"]]
                self.rv_canvas.create_rectangle(x1, y1, x2, y2, outline=col, width=3 if o["item_id"] == item else 1,
                                                dash=() if o["state"] == "visible" else (6, 4))
                self.rv_canvas.create_text(x1 + 3, y1 - 8, anchor="w", fill=col, font=T.FONT_SMALL_BOLD,
                                           text=f"{o['item_id'][:12]} {o.get('label')}")
        if item and not any(o.get("box") for o in rec.get("items", []) if o["item_id"] == item):
            self.rv_canvas.create_text(ox + 10, oy + 14, anchor="w", fill="#fbbf24", font=T.FONT_SMALL_BOLD,
                                       text="No box for the selected item in this decision (state shown at right).")
        diff = res.signature_diff
        ver = "frame verified (16x9 signature)" if diff is not None and diff <= 12 else (
            "signature MISMATCH - inspect with care" if diff is not None else "signature not checked")
        self.rv_pos.configure(text=f"clean frame {res.request.index}: the decision's own analysed frame; {ver}")

    def _review_play(self) -> None:
        rv = self.review
        if self.playback is not None:
            self._review_stop_play()
            return
        if not rv or not rv.get("lookup") or not rv["lookup"].ok:
            self.c.say("Select a decision with a clean frame to play from.", "warning")
            return
        loc = rv["lookup"]
        self.play_gen += 1
        cw, ch = max(64, self.rv_canvas.winfo_width()), max(36, self.rv_canvas.winfo_height())
        self.playback = PlaybackStream(loc.video, loc.index, loc.fps, cw, ch, self.play_gen)
        self.playback.start()
        self._play_t0 = time.monotonic()
        self.b_play.configure(text="Pause playback")

    def _review_stop_play(self) -> None:
        if self.playback is not None:
            pb, self.playback = self.playback, None
            import threading
            threading.Thread(target=pb.stop, daemon=True).start()     # never join on the Tk thread
        if hasattr(self, "b_play"):
            self.b_play.configure(text="Play clean video")

    def _review_play_tick(self) -> None:
        pb = self.playback
        due = pb.start_index + int((time.monotonic() - self._play_t0) * pb.fps)
        item = None
        while True:
            try:
                nxt = pb.q.get_nowait()
            except Exception:  # noqa: BLE001 - queue.Empty
                break
            if nxt is None:
                self._review_stop_play()
                self.c.say("Playback reached the end of the clean recording.", "neutral")
                return
            item = nxt
            if nxt[0] >= due:
                break
        if item is None:
            return
        idx, img = item
        cw, ch = max(64, self.rv_canvas.winfo_width()), max(36, self.rv_canvas.winfo_height())
        ppm, (dw, dh), _ = to_ppm(img, cw, ch)
        self.review_photo = _photo(self.root, ppm)
        self.rv_canvas.delete("all")
        self.rv_canvas.create_image((cw - dw) // 2, (ch - dh) // 2, anchor="nw", image=self.review_photo)
        maps = self.review["maps"].load("clean") or {}
        gaps = [g for g in maps.get("gaps", []) if g.get("after_i") == idx - 1]
        note = f"playing clean frame {idx} (recorded pixels; boxes only on exact decision frames)"
        if gaps:
            note += f" - capture GAP of {gaps[0]['seconds']} s just before this frame (not compressed silently)"
        self.rv_pos.configure(text=note)

    def _review_same(self) -> None:
        item = self._review_item()
        if item:
            self._corr_preset = item
            self.show("corrections")

    # =================================================================================================================
    # Activity (paged, filtered, saved facts)
    # =================================================================================================================
    def _build_activity(self, f) -> None:
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)
        fl = tk.Frame(f, bg=T.BG)
        fl.grid(row=0, column=0, sticky="ew", pady=(0, T.SPACE_2))
        self.act_item = ttk.Combobox(fl, state="readonly", width=32, style="EW.TCombobox")
        self.act_item.pack(side="left")
        self.act_run = ttk.Combobox(fl, state="readonly", width=26, style="EW.TCombobox")
        self.act_run.pack(side="left", padx=4)
        self.act_kind = ttk.Combobox(fl, state="readonly", width=26, style="EW.TCombobox",
                                     values=["All events", "monitoring_started", "first_confirmed", "gap_started", "recovered",
                                             "review_requested", "monitoring_ended"])
        self.act_kind.set("All events")
        self.act_kind.pack(side="left", padx=4)
        button(fl, "Apply filters", lambda: self.activity_load(0)).pack(side="left", padx=4)
        button(fl, "< Newer", lambda: self.activity_load(max(0, self._act_off - 50))).pack(side="right")
        button(fl, "Older >", lambda: self.activity_load(self._act_off + 50)).pack(side="right", padx=4)
        self.act_tv = ttk.Treeview(f, columns=("when", "event", "original", "effective", "run", "frame"), show="headings",
                                   style="EW.Treeview")
        for col, text, w in (("when", "When", 130), ("event", "Event", 330), ("original", "Original attribution", 170),
                             ("effective", "Effective (corrected)", 170), ("run", "Run", 150), ("frame", "Frame", 120)):
            self.act_tv.heading(col, text=text)
            self.act_tv.column(col, width=w, minwidth=70)
        self.act_tv.grid(row=1, column=0, sticky="nsew")
        self.act_status = tk.Label(f, text="", bg=T.BG, fg=T.TEXT_MUTED, font=T.FONT_SMALL, anchor="w")
        self.act_status.grid(row=2, column=0, sticky="ew")
        self._act_off = 0
        self._act_gen = 0

    def _activity_refresh_filters(self) -> None:
        items = ["All items"] + [f"{p['name']}  ({p['item_id']})" for p in self.c.library]
        self.act_item.configure(values=items)
        preset = getattr(self, "_activity_item_preset", None)
        if preset:
            self.act_item.set(next((x for x in items if x.endswith(f"({preset})")), "All items"))
            self._activity_item_preset = None
        elif not self.act_item.get():
            self.act_item.set("All items")
        runs = ["All runs"] + [f"{_wall(r['started_wall'])}  ({r['run_id']})" for r in self.c.runs]
        self.act_run.configure(values=runs)
        if not self.act_run.get():
            self.act_run.set("All runs")

    def activity_load(self, offset: int) -> None:
        from ..review.activity import page
        it, rn, kd = self.act_item.get(), self.act_run.get(), self.act_kind.get()
        item = it[it.rfind("(") + 1:-1] if "(" in it else None
        run = rn[rn.rfind("(") + 1:-1] if "(" in rn else None
        kinds = None if kd in ("", "All events") else [kd]
        self._act_off = max(0, offset)
        self._act_gen += 1
        store, cor, off = self.c.engine.store, self.c.corrections, self._act_off
        self.act_status.configure(text="Loading saved activity...")
        self.c.tasks.submit("activity", ("activity", self._act_gen),
                            lambda: page(store, cor, item_id=item, run_id=run, kinds=kinds, offset=off, limit=50))

    def _activity_on_page(self, c) -> None:
        if c.token != ("activity", self._act_gen):
            return
        if not c.ok:
            self.act_status.configure(text=f"Activity could not be read: {c.error}")
            return
        p = c.value
        self.act_tv.delete(*self.act_tv.get_children())
        for r in p["rows"]:
            self.act_tv.insert("", "end", values=(datetime.fromtimestamp(r["wall"]).strftime("%Y-%m-%d %H:%M:%S"), r["label"],
                                                  r["original_name"] or r["original_item"],
                                                  (r["effective_name"] + " (corrected)") if r["corrected"] else r["effective_name"],
                                                  r["run_id"], r["frame"] or ""))
        live = " Live events of a running session are saved when it finalizes." if self.c.phase != PHASE_IDLE else ""
        self.act_status.configure(text=("No saved activity for these filters." if p["status"] == "empty" else
                                        f"Showing {p['offset'] + 1}-{p['offset'] + len(p['rows'])} of {p['total']} saved events.")
                                  + live)

    # =================================================================================================================
    # Same item corrections
    # =================================================================================================================
    def _build_corrections(self, f) -> None:
        f.grid_columnconfigure(0, weight=1)
        c = card(f, "Same item: combine two profiles' HISTORY (references and recognition are not changed)")
        c.grid(row=0, column=0, sticky="ew")
        row = tk.Frame(c, bg=T.SURFACE)
        row.pack(fill="x")
        self.corr_a = ttk.Combobox(row, state="readonly", width=34, style="EW.TCombobox")
        self.corr_a.pack(side="left")
        tk.Label(row, text="is the same physical item as", bg=T.SURFACE, fg=T.TEXT_MUTED).pack(side="left", padx=6)
        self.corr_b = ttk.Combobox(row, state="readonly", width=34, style="EW.TCombobox")
        self.corr_b.pack(side="left")
        button(row, "Compare", self._corr_preview).pack(side="left", padx=6)
        self.corr_info = tk.Label(c, text="Choose two profiles and Compare.", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_SMALL,
                                  justify="left", anchor="w", wraplength=900)
        self.corr_info.pack(fill="x", pady=T.SPACE_2)
        row2 = tk.Frame(c, bg=T.SURFACE)
        row2.pack(fill="x")
        self.corr_primary = tk.StringVar(value="a")
        ttk.Radiobutton(row2, text="Keep the first as primary", variable=self.corr_primary, value="a",
                        style="EW.TRadiobutton").pack(side="left")
        ttk.Radiobutton(row2, text="Keep the second as primary", variable=self.corr_primary, value="b",
                        style="EW.TRadiobutton").pack(side="left", padx=8)
        self.corr_ack = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="I reviewed the conflicts", variable=self.corr_ack, style="EW.TCheckbutton").pack(side="left", padx=8)
        self.b_corr_save = button(row2, "Save correction", self._corr_apply, "primary")
        self.b_corr_save.pack(side="right")
        h = card(f, "Correction history for the first profile (Undo appends a reversal)")
        h.grid(row=1, column=0, sticky="nsew", pady=T.SPACE_2)
        f.grid_rowconfigure(1, weight=1)
        self.corr_hist = ttk.Treeview(h, columns=("when", "kind", "detail", "status"), show="headings", style="EW.Treeview",
                                      height=8, selectmode="browse")
        for col, text, w in (("when", "When", 130), ("kind", "Kind", 90), ("detail", "Detail", 380), ("status", "Status", 330)):
            self.corr_hist.heading(col, text=text)
            self.corr_hist.column(col, width=w, minwidth=60)
        self.corr_hist.pack(fill="both", expand=True)
        hb = tk.Frame(h, bg=T.SURFACE)
        hb.pack(fill="x", pady=(T.SPACE_1, 0))
        button(hb, "Undo selected correction", self._corr_undo).pack(side="right")
        button(hb, "Reconcile selected (same operation)", self._corr_reconcile).pack(side="right", padx=6)
        # comparison/save/retry/reconcile state lives in the Tk-free CorrectionFlow (EW-LAB-03 R1/R2)
        self.flow = CorrectionFlow(self.c.corrections, self.c.tasks.submit)
        self._hist_ops: dict[str, str] = {}             # history row -> operation id

    def _corr_refresh_profiles(self) -> None:
        vals = [f"{p['name']}  ({p['item_id']})" for p in self.c.library]
        self.corr_a.configure(values=vals)
        self.corr_b.configure(values=vals)
        preset = getattr(self, "_corr_preset", None)
        if preset:
            self.corr_a.set(next((x for x in vals if x.endswith(f"({preset})")), ""))
            self._corr_preset = None
        self._corr_history()

    @staticmethod
    def _id_of(v: str) -> str | None:
        return v[v.rfind("(") + 1:-1] if "(" in v else None

    def _corr_preview(self) -> None:
        a, b = self._id_of(self.corr_a.get()), self._id_of(self.corr_b.get())
        if not a or not b:
            self.c.say("Choose both profiles.", "warning")
            return
        self.flow.compare(a, b)
        self.corr_info.configure(text="Comparing the two effective groups...")

    def _corr_on_preview(self, c) -> None:
        text = self.flow.on_preview(c.token, c.ok, c.value, c.error)
        if text is not None:                            # None: an older comparison's result, ignored
            self.corr_info.configure(text=text)

    def _corr_apply(self) -> None:
        notice = self.flow.save(self.corr_primary.get(), bool(self.corr_ack.get()))
        if notice:
            self.c.say(notice, "warning")
        if self.flow.request is not None:
            set_enabled(self.b_corr_save, False)

    def _corr_on_apply(self, c) -> None:
        """Report exactly what the durable store (or the classified failure) says - see app.correction_flow."""
        set_enabled(self.b_corr_save, True)
        if not c.ok:                                    # the flow wraps exceptions; this is a task-runner failure
            self.c.say(f"Outcome unknown: {c.error}. Use Reconcile on the operation in the history.", "warning")
        else:
            msg, tone = self.flow.on_result(c.value)
            self.c.say(msg, tone)
        self._corr_history()

    def _corr_history(self) -> None:
        a = self._id_of(self.corr_a.get())
        if not a:
            self.corr_hist.delete(*self.corr_hist.get_children())
            return
        cor = self.c.corrections
        self.c.tasks.submit("corr_history", ("corr_hist", a), lambda: cor.history(a))

    def _corr_on_history(self, c) -> None:
        if not c.ok:
            return
        self.corr_hist.delete(*self.corr_hist.get_children())
        self._hist_ops = {}
        for r in c.value:
            p = r["payload_obj"]
            detail = (f"{p.get('a')} + {p.get('b')} -> primary {p.get('primary')}" if r["kind"] == "same_item" else
                      f"Undo of {p.get('reverses')}")
            self._hist_ops[r["correction_id"]] = r["op_id"]
            self.corr_hist.insert("", "end", iid=r["correction_id"], values=(_wall(r["wall"]), r["kind"], detail,
                                                                              r.get("status_text", "")))

    def _corr_reconcile(self) -> None:
        sel = self.corr_hist.selection()
        op = self._hist_ops.get(sel[0]) if sel else self.flow.retry_op()
        if not op:
            self.c.say("Select a correction or an Undo in the history to reconcile.", "warning")
            return
        self.flow.reconcile(op)

    def _corr_undo(self) -> None:
        sel = self.corr_hist.selection()
        if not sel:
            self.c.say("Select a Same item correction to undo.", "warning")
            return
        self.flow.undo(sel[0])                           # a retry of this Undo reuses its operation id


    # =================================================================================================================
    # Settings (Pending / Applied / Running; Health details)
    # =================================================================================================================
    def _build_settings(self, f) -> None:
        f.grid_columnconfigure(1, weight=1)
        cats = tk.Listbox(f, bg=T.SURFACE, fg=T.TEXT, selectbackground=T.ACCENT_SOFT, highlightthickness=0, bd=0,
                          font=T.FONT_BODY, width=24, exportselection=False)
        for x in ("Recording & rules", "Inference (GPU)", "Health: recognition calibration", "Run diagnostics", "About"):
            cats.insert("end", x)
        cats.grid(row=0, column=0, sticky="ns")
        cats.selection_set(0)
        cats.bind("<<ListboxSelect>>", lambda e: self._settings_render())
        self.set_cats = cats
        body = card(f)
        body.grid(row=0, column=1, sticky="nsew", padx=(T.SPACE_2, 0))
        f.grid_rowconfigure(0, weight=1)
        self.set_body = body
        self.set_text = tk.Label(body, text="", bg=T.SURFACE, fg=T.TEXT, font=T.FONT_MONO_SMALL, justify="left",
                                 anchor="nw", wraplength=760)
        self.set_text.pack(fill="both", expand=True)

    def _settings_render(self) -> None:
        s = self.set_cats.curselection()
        cat = self.set_cats.get(s[0]) if s else "Recording & rules"
        c = self.c
        if cat == "Recording & rules":
            st = c.settings_status()
            al = c.allowance_status()
            lines = ["Edit these in Display Assets > Configure. Pending = on screen, Applied = the saved configuration "
                     "the next Start uses, Running = what the active run acknowledged.", "",
                     f"{'setting':22} {'pending':>8} {'applied':>8} {'running':>8}"]
            for k, v in st.items():
                run = "-" if v["running"] is None else ("ON" if v["running"] else "OFF")
                lines.append(f"{k:22} {'ON' if v['pending'] else 'OFF':>8} {'ON' if v['applied'] else 'OFF':>8} {run:>8}"
                             + ("   (not applied yet)" if v["dirty"] else ""))
            lines.append(f"{'review allowance (s)':22} {al['pending']:>8} {al['applied']:>8} "
                         f"{'-' if al['running'] is None else al['running']:>8}" + ("   (not applied yet)" if al["dirty"] else ""))
            rs = c.running_setup
            lines += ["", f"Running setup: {rs['name']} v{rs['version']}" if rs else "Running setup: none",
                      f"Saved setup on screen: {c.setup.name} v{c.setup.version}" if c.setup is not None else
                      "Saved setup on screen: not saved yet"]
            txt = "\n".join(lines)
        elif cat == "Inference (GPU)":
            r = c.worker_receipt or {}
            txt = (f"Requested device: cuda:0 (fp32). There is no CPU substitute.\nWorker: {c.worker_status}\n"
                   f"Running device: {r.get('device')}  dtype: {r.get('dtype')}  torch: {r.get('torch')}  CUDA: {r.get('cuda')}\n"
                   f"GPU: {r.get('gpu_name')}  VRAM free after load: {r.get('vram_free_mib_after_load')} MiB\n"
                   f"Prompt-free model sha256: {r.get('model_sha256')}\nVisual-prompt model sha256: {r.get('vp_model_sha256')}\n"
                   "The worker starts on the first discovery or Start and stops when the application closes.")
        elif cat == "Health: recognition calibration":
            g = c.gate()
            txt = (f"Descriptor: {g['descriptor']}\nP (min genuine 5th percentile): {g['P']}\nN (max distinct-object distractor): {g['N']}\n"
                   f"Midpoint M: {g['M']:.4f}   reject below {g['reject_below']:.4f}   support above {g['support_above']:.4f}\n"
                   f"Calibration id: {g['calibration_id']}   qualified on its development set: {g['qualified']}\n\n"
                   "Only strong evidence (above the support level) can confirm an item; weak evidence is shown as Match "
                   "uncertain and never counts as absence. The calibration was measured on generated development clips. "
                   "It is NOT a statement about this scene: setup does not test motion, removal or return, and similar "
                   "items can stay uncertain.")
        elif cat == "Run diagnostics":
            sn = self._snapshot
            txt = "No run is active." if not sn or c.phase == PHASE_IDLE else (
                f"Phase {sn.phase} step {sn.step}\nRates: {sn.rates}\nOutputs: {sn.outputs}\nSource: "
                f"{ {k: v for k, v in (sn.source or {}).items() if k != 'decoder'} }\nWorker: "
                f"{ {k: v for k, v in (sn.worker or {}).items() if k != 'receipt'} }\nDiscards: {c.engine.session.discards}\n"
                f"Displayed packets: {self.metrics.total_painted}  UI metrics: {self.metrics.path}")
            txt += (f"\nStale completions discarded: {len(c.discarded_completions)}"
                    f"\nScreen update failures after durable saves: {len(c.ui_failures)}")
        else:
            txt = ("ECHO Watch Lab - development build (AGPL-3.0-only). Not a commercial release.\n"
                   "Supported source: one local prerecorded video replayed at its natural pace (no camera).\n"
                   "Notices: LICENSE, NOTICE.md, THIRD_PARTY_NOTICES.md and docs/donor-notices in the Lab folder.")
        self._set_lbl(self.set_text, txt)

    # =================================================================================================================
    # exit
    # =================================================================================================================
    def _yoloe_session_busy(self) -> bool:
        return self.yc is not None and self.yc.session.busy

    def request_exit(self) -> None:
        st = self.c.request_exit()
        if self._yoloe_session_busy():
            # the YOLOE test is stopped (bounded) and its evidence finalized before the window closes
            self.yc.session.stop("application close")
            self.c.closing = True
            st = "closing"
        if st == "ready":
            self._finish_exit()
            return
        if self._force_armed:
            self.metrics.event("force_close", phase=self.c.phase)
            logger.warning("force close requested while finalizing; outputs may be incomplete")
            self._finish_exit(force=True)
            return
        self._force_armed = True
        self.c.say("Closing: finalizing recordings and the decision trace first. Close again to force (outputs may be "
                   "incomplete).", "warning")

    def _finish_exit(self, force: bool = False) -> None:
        if getattr(self, "_shutdown_started", False):
            return
        self._shutdown_started = True
        self._review_stop_play()
        self.c.say("Closing...", "neutral")
        import threading

        def work():
            rec = {}
            try:
                rec = self.c.shutdown() if not force else {"forced": True}
            except Exception as e:  # noqa: BLE001
                rec = {"error": str(e)}
            if self.yc is not None:
                try:
                    rec["yoloe"] = self.yc.shutdown(wait_s=0.0 if force else 30.0)
                except Exception as e:  # noqa: BLE001
                    rec["yoloe"] = {"error": str(e)}
            self.metrics.event("shutdown", **{k: str(v) for k, v in rec.items()})
            self._shutdown_done = True
        threading.Thread(target=work, name="shutdown", daemon=True).start()
        self._wait_shutdown(time.monotonic())

    def _wait_shutdown(self, t0: float) -> None:
        if getattr(self, "_shutdown_done", False) or time.monotonic() - t0 > 45:
            self.prep.shutdown(1.0)
            self.review_worker.shutdown(1.0)
            if self.yoloe_page is not None:
                self.yoloe_page.shutdown()
            self.metrics.close()
            self.root.destroy()
            return
        self.root.after(100, lambda: self._wait_shutdown(t0))
