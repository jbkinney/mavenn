# Set matplotlib to use a non-GUI backend before any test or test helper imports it.
# Avoids Tcl/Tk errors on headless CI (e.g. Windows hosted runners where tk.tcl
# is missing or broken).
import os

os.environ.setdefault("MPLBACKEND", "Agg")
