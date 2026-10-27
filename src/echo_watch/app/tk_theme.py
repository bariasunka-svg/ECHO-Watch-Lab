# SPDX-License-Identifier: AGPL-3.0-only
"""ECHO Watch navy/cyan design tokens and private ttk styles (Tk presentation only).

Colour, typography and spacing tokens are ADOPTED (subset, values unchanged) from the frozen PatronSight reference
src/patronsight/app/dashboard_theme.py (archive Patsight_PS27D_C3_reference_source.zip; LICENSE: 'Original PatSight
contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only). NOT adopted: the PIL-rendered rounded image elements
(decor.py) and the people/service styles. Difference: the donor never called ``theme_use`` because a second review
window shared its interpreter; the Lab workbench owns its interpreter alone, so it selects ``clam`` (whose elements
honour these colours) and defines private ``EW.*`` style names.
State colours follow the ECHO item states (Visible green, Checking amber, Not visible red, Match uncertain violet,
Cannot assess / Paused / Offline / Stopped grey). EW-LAB-05: surface/accent values follow the owner's Display Assets
references (docs/DISPLAY_ASSETS_UX_20261026.md: navy #061525/#081c2c, cards #0b2435, borders #244254, cyan #20d9e9,
text #e8f2f8 / #a2b6c8), Segoe UI 10-11 pt body, 18-22 pt headings, 8/12/16/24 spacing, ~36-40 px controls. Colour is
never the only signal: every state also has text.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

BG = "#061525"
RAIL = "#04101d"
SURFACE = "#0b2435"
SURFACE_ALT = "#123247"
SURFACE_SUNKEN = "#081c2c"
BORDER = "#244254"
BORDER_STRONG = "#2f5a70"
ACCENT = "#20d9e9"
ACCENT_PRESSED = "#12b8c7"
ACCENT_TEXT = "#03222a"
ACCENT_SOFT = "#0d3b4c"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
PENDING = "#a78bfa"
TEXT = "#e8f2f8"
TEXT_MUTED = "#a2b6c8"
TEXT_DIM = "#6f879b"
VIDEO_BG = "#081c2c"
OUTLINE_MUTED = "#7f98aa"

TONES: dict[str, tuple[str, str]] = {
    "neutral": ("#17324a", TEXT_MUTED),
    "accent": ("#0b3a4a", ACCENT),
    "success": ("#0d3321", "#4ade80"),
    "warning": ("#3b2a08", "#fbbf24"),
    "danger": ("#3d1111", "#f87171"),
    "pending": ("#2a1f4a", "#c4b5fd"),
}

STATE_TONE = {"visible": "success", "checking": "warning", "not_seen": "danger", "uncertain": "pending",
              "cannot_assess": "neutral", "paused": "neutral", "offline": "neutral", "stopped": "neutral",
              "saved": "accent"}
STATE_BOX = {"visible": "#4ade80", "checking": "#fbbf24"}

FONT_FAMILY = "Segoe UI"
FONT_FAMILY_STRONG = "Segoe UI Semibold"
MONO_FAMILY = "Consolas"
FONT_BRAND = (FONT_FAMILY_STRONG, 15)
FONT_H1 = (FONT_FAMILY_STRONG, 20)
FONT_H1_COMPACT = (FONT_FAMILY_STRONG, 16)
FONT_SUB = (FONT_FAMILY, 11)
FONT_TITLE = (FONT_FAMILY_STRONG, 13)
FONT_H2 = (FONT_FAMILY_STRONG, 12)
FONT_COUNT = (FONT_FAMILY_STRONG, 18)
FONT_BODY = (FONT_FAMILY, 10)
FONT_BODY_BOLD = (FONT_FAMILY, 10, "bold")
FONT_SMALL = (FONT_FAMILY, 9)
FONT_SMALL_BOLD = (FONT_FAMILY, 9, "bold")
FONT_MICRO = (FONT_FAMILY, 8)
FONT_NAV = (FONT_FAMILY, 10)
FONT_NAV_ACTIVE = (FONT_FAMILY_STRONG, 10)
FONT_MONO_SMALL = (MONO_FAMILY, 9)

SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5 = 4, 8, 12, 16, 24
NAV_WIDTH = 196
RIGHT_RAIL_WIDTH = 336
CONTROL_PADY = 9               # with 10 pt text this gives ~36-40 px tall buttons at 100%


def install(root: tk.Misc) -> None:
    """Private EW.* styles in the CURRENT theme (never theme_use)."""
    s = ttk.Style(root)
    try:
        s.theme_use("clam")       # this interpreter belongs to the workbench alone; clam honours every colour below
    except tk.TclError:
        pass
    root.configure(background=BG)
    root.option_add("*Font", FONT_BODY)
    s.configure("EW.TFrame", background=BG)
    s.configure("EW.Card.TFrame", background=SURFACE)
    s.configure("EW.TLabel", background=BG, foreground=TEXT, font=FONT_BODY)
    s.configure("EW.Card.TLabel", background=SURFACE, foreground=TEXT, font=FONT_BODY)
    s.configure("EW.Muted.TLabel", background=SURFACE, foreground=TEXT_MUTED, font=FONT_SMALL)
    s.configure("EW.Title.TLabel", background=BG, foreground=TEXT, font=FONT_TITLE)
    s.configure("EW.H2.TLabel", background=SURFACE, foreground=TEXT, font=FONT_H2)
    s.configure("EW.TButton", background=SURFACE_ALT, foreground=TEXT, bordercolor=BORDER_STRONG, focuscolor=ACCENT,
                padding=(10, 5), font=FONT_BODY)
    s.map("EW.TButton", background=[("disabled", SURFACE), ("pressed", BORDER_STRONG), ("active", BORDER_STRONG)],
          foreground=[("disabled", TEXT_DIM)])
    s.configure("EW.Primary.TButton", background=ACCENT, foreground=ACCENT_TEXT, padding=(12, 5), font=FONT_BODY_BOLD)
    s.map("EW.Primary.TButton", background=[("disabled", SURFACE), ("pressed", ACCENT_PRESSED), ("active", ACCENT_PRESSED)],
          foreground=[("disabled", TEXT_DIM)])
    s.configure("EW.Danger.TButton", background="#3d1111", foreground="#fca5a5", padding=(10, 5))
    s.map("EW.Danger.TButton", background=[("disabled", SURFACE), ("active", "#5b1717")], foreground=[("disabled", TEXT_DIM)])
    s.configure("EW.TCheckbutton", background=SURFACE, foreground=TEXT, font=FONT_BODY)
    s.map("EW.TCheckbutton", background=[("active", SURFACE)])
    s.configure("EW.TRadiobutton", background=SURFACE, foreground=TEXT, font=FONT_BODY)
    s.configure("EW.TEntry", fieldbackground=SURFACE_SUNKEN, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER_STRONG)
    s.configure("EW.TCombobox", fieldbackground=SURFACE_SUNKEN, foreground=TEXT, background=SURFACE_ALT)
    s.configure("EW.Treeview", background=SURFACE_SUNKEN, fieldbackground=SURFACE_SUNKEN, foreground=TEXT, rowheight=32,
                bordercolor=BORDER, font=FONT_SMALL)
    s.map("EW.Treeview", background=[("selected", ACCENT_SOFT)], foreground=[("selected", TEXT)])
    s.configure("EW.Treeview.Heading", background=SURFACE_ALT, foreground=TEXT_MUTED, font=FONT_SMALL_BOLD, relief="flat")
    s.configure("EW.Vertical.TScrollbar", background=SURFACE_ALT, troughcolor=SURFACE_SUNKEN, bordercolor=BORDER,
                arrowcolor=TEXT_MUTED)
    s.configure("EW.Horizontal.TScale", background=SURFACE, troughcolor=SURFACE_SUNKEN)
