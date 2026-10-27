# SPDX-License-Identifier: AGPL-3.0-only
"""env/ui/pythonw.exe -I -m echo_watch : the owner manual launch of the native workbench (see app.launch)."""
import sys

from .app.launch import main

sys.exit(main())
