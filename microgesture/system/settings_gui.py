"""Settings GUI panel — sensitivity, scroll, right-click mode.

Opens as a separate tkinter window from the tray menu.
All changes apply immediately via callbacks.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class SettingsPanel:
    """Floating settings window for real-time parameter adjustment."""

    def __init__(
        self,
        *,
        cursor_sensitivity: float = 0.8,
        scroll_sensitivity: float = 2.0,
        scroll_deadzone: float = 0.03,
        right_click_mode: str = "fist_tap",
        shadow_threshold: float = 0.90,
        on_cursor_sensitivity: Callable[[float], None] | None = None,
        on_scroll_sensitivity: Callable[[float], None] | None = None,
        on_scroll_deadzone: Callable[[float], None] | None = None,
        on_right_click: Callable[[str], None] | None = None,
        on_shadow_threshold: Callable[[float], None] | None = None,
    ):
        self._root = tk.Tk()
        self._root.title("MicroGesture Settings")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Callbacks
        self._cb_cursor = on_cursor_sensitivity
        self._cb_scroll = on_scroll_sensitivity
        self._cb_dz = on_scroll_deadzone
        self._cb_rc = on_right_click
        self._cb_shadow = on_shadow_threshold

        # ── Main frame ──────────────────────────────────────────────────
        f = ttk.Frame(self._root, padding=(16, 12, 16, 12))
        f.grid(row=0, column=0, sticky="nsew")

        row = 0

        # ── Cursor sensitivity ──────────────────────────────────────────
        ttk.Label(f, text="光标灵敏度").grid(row=row, column=0, sticky="w", pady=(0, 2))
        self._cursor_var = tk.DoubleVar(value=cursor_sensitivity)
        self._cursor_scale = ttk.Scale(
            f, from_=0.3, to=3.0, variable=self._cursor_var,
            orient=tk.HORIZONTAL, length=260, command=self._on_cursor_change,
        )
        self._cursor_scale.grid(row=row + 1, column=0, sticky="ew")
        self._cursor_label = ttk.Label(f, text=f"{cursor_sensitivity:.1f}")
        self._cursor_label.grid(row=row + 1, column=1, padx=(8, 0))
        row += 2

        # ── Scroll sensitivity ─────────────────────────────────────────
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2,
                                                     sticky="ew", pady=6)
        row += 1

        ttk.Label(f, text="滚动灵敏度").grid(row=row, column=0, sticky="w")
        self._scroll_var = tk.DoubleVar(value=scroll_sensitivity)
        self._scroll_scale = ttk.Scale(
            f, from_=1.0, to=20.0, variable=self._scroll_var,
            orient=tk.HORIZONTAL, length=260, command=self._on_scroll_change,
        )
        self._scroll_scale.grid(row=row + 1, column=0, sticky="ew")
        self._scroll_label = ttk.Label(f, text=f"{scroll_sensitivity:.1f}")
        self._scroll_label.grid(row=row + 1, column=1, padx=(8, 0))
        row += 2

        # Scroll deadzone
        ttk.Label(f, text="滚动死区").grid(row=row, column=0, sticky="w")
        self._dz_var = tk.DoubleVar(value=scroll_deadzone)
        self._dz_scale = ttk.Scale(
            f, from_=0.005, to=0.10, variable=self._dz_var,
            orient=tk.HORIZONTAL, length=260, command=self._on_dz_change,
        )
        self._dz_scale.grid(row=row + 1, column=0, sticky="ew")
        self._dz_label = ttk.Label(f, text=f"{scroll_deadzone:.3f}")
        self._dz_label.grid(row=row + 1, column=1, padx=(8, 0))
        row += 2

        # ── Right-click mode ────────────────────────────────────────────
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2,
                                                     sticky="ew", pady=6)
        row += 1

        ttk.Label(f, text="右键模式").grid(row=row, column=0, sticky="w")
        self._rc_var = tk.StringVar(value=right_click_mode)
        rc_frame = ttk.Frame(f)
        rc_frame.grid(row=row + 1, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(rc_frame, text="Fist + Tap", variable=self._rc_var,
                        value="fist_tap", command=self._on_rc_change).pack(side=tk.LEFT)
        ttk.Radiobutton(rc_frame, text="Two Finger", variable=self._rc_var,
                        value="two_finger", command=self._on_rc_change).pack(side=tk.LEFT, padx=(12, 0))
        row += 2

        # ── ONNX shadow threshold ───────────────────────────────────────
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2,
                                                     sticky="ew", pady=6)
        row += 1

        ttk.Label(f, text="ONNX 接管阈值").grid(row=row, column=0, sticky="w")
        self._shadow_var = tk.DoubleVar(value=shadow_threshold)
        self._shadow_scale = ttk.Scale(
            f, from_=0.50, to=1.0, variable=self._shadow_var,
            orient=tk.HORIZONTAL, length=260, command=self._on_shadow_change,
        )
        self._shadow_scale.grid(row=row + 1, column=0, sticky="ew")
        self._shadow_label = ttk.Label(f, text=f"{shadow_threshold:.0%}")
        self._shadow_label.grid(row=row + 1, column=1, padx=(8, 0))
        row += 2

        # ── Close button ───────────────────────────────────────────────
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2,
                                                     sticky="ew", pady=6)
        row += 1
        ttk.Button(f, text="关闭", command=self._on_close).grid(
            row=row, column=0, columnspan=2, pady=(4, 0))

        # Size hints
        self._root.geometry("320x480")
        self._window_open = True

    # ── callbacks ────────────────────────────────────────────────────────

    def _on_cursor_change(self, val):
        v = round(float(val), 1)
        self._cursor_label.config(text=f"{v:.1f}")
        if self._cb_cursor:
            self._cb_cursor(v)

    def _on_scroll_change(self, val):
        v = round(float(val), 1)
        self._scroll_label.config(text=f"{v:.1f}")
        if self._cb_scroll:
            self._cb_scroll(v)

    def _on_dz_change(self, val):
        v = round(float(val), 3)
        self._dz_label.config(text=f"{v:.3f}")
        if self._cb_dz:
            self._cb_dz(v)

    def _on_rc_change(self):
        if self._cb_rc:
            self._cb_rc(self._rc_var.get())

    def _on_shadow_change(self, val):
        v = round(float(val), 2)
        self._shadow_label.config(text=f"{v:.0%}")
        if self._cb_shadow:
            self._cb_shadow(v)

    def _on_close(self):
        self._window_open = False
        self._root.destroy()

    # ── lifecycle ────────────────────────────────────────────────────────

    def show(self):
        """Show the window (non-blocking)."""
        self._root.deiconify()
        self._root.lift()

    @property
    def is_open(self):
        return self._window_open


def open_settings_panel(
    cursor_sensitivity: float = 0.8,
    scroll_sensitivity: float = 2.0,
    scroll_deadzone: float = 0.03,
    right_click_mode: str = "fist_tap",
    shadow_threshold: float = 0.90,
    on_cursor_sensitivity: Callable | None = None,
    on_scroll_sensitivity: Callable | None = None,
    on_scroll_deadzone: Callable | None = None,
    on_right_click: Callable | None = None,
    on_shadow_threshold: Callable | None = None,
) -> SettingsPanel:
    """Create and show the settings panel. Returns panel for lifecycle management."""
    panel = SettingsPanel(
        cursor_sensitivity=cursor_sensitivity,
        scroll_sensitivity=scroll_sensitivity,
        scroll_deadzone=scroll_deadzone,
        right_click_mode=right_click_mode,
        shadow_threshold=shadow_threshold,
        on_cursor_sensitivity=on_cursor_sensitivity,
        on_scroll_sensitivity=on_scroll_sensitivity,
        on_scroll_deadzone=on_scroll_deadzone,
        on_right_click=on_right_click,
        on_shadow_threshold=on_shadow_threshold,
    )
    panel.show()
    return panel
