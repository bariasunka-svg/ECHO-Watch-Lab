# SPDX-License-Identifier: AGPL-3.0-only
"""ECHO Watch Lab: the PatronSight-style native desktop workbench for remembered display items.

engine/  the single managed session owner, paired result packets, item policy, recording and decision trace
worker/  the isolated GPU inference child (env/gpu only)
review/  read-only saved-run projection, activity paging and append-only corrections
app/     Tk presentation (app/tk_*) over pure, GUI-free view-model/command state (app/viewmodel.py)
"""

__version__ = "0.3.0-lab"
