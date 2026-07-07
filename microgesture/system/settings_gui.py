"""Settings GUI — tabbed panels for Mouse / Gesture / DTW.

Opens from tray menu. Changes apply immediately on slider release.
Save button persists all values to config.json.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class SettingsPanel:
    """Tabbed settings window for real-time parameter adjustment."""

    def __init__(
        self,
        *,
        # Mouse
        cursor_sensitivity: float = 0.8,
        cursor_deadzone: float = 0.003,
        cursor_tap_deadzone: float = 0.012,
        scroll_sensitivity: float = 2.0,
        scroll_deadzone: float = 0.03,
        # Gesture
        right_click_mode: str = "fist_tap",
        shadow_threshold: float = 0.90,
        tap_threshold: float = 0.3,
        tap_min_bend: float = 0.15,
        # DTW
        dtw_motion_threshold: float = 0.005,
        dtw_still_frames: int = 10,
        dtw_min_frames: int = 15,
        dtw_match_threshold: float = 8.0,
        dtw_cooldown_frames: int = 90,
        # Callbacks
        on_cursor_sensitivity: Callable[[float], None] | None = None,
        on_cursor_deadzone: Callable[[float], None] | None = None,
        on_cursor_tap_deadzone: Callable[[float], None] | None = None,
        on_scroll_sensitivity: Callable[[float], None] | None = None,
        on_scroll_deadzone: Callable[[float], None] | None = None,
        on_right_click: Callable[[str], None] | None = None,
        on_shadow_threshold: Callable[[float], None] | None = None,
        on_tap_threshold: Callable[[float], None] | None = None,
        on_tap_min_bend: Callable[[float], None] | None = None,
        on_dtw_motion: Callable[[float], None] | None = None,
        on_dtw_still: Callable[[int], None] | None = None,
        on_dtw_min: Callable[[int], None] | None = None,
        on_dtw_threshold: Callable[[float], None] | None = None,
        on_dtw_cooldown: Callable[[int], None] | None = None,
        on_save: Callable[[], None] | None = None,
    ):
        self._root = tk.Tk()
        self._root.title("MicroGesture Settings")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_save = on_save
        self._window_open = True

        notebook = ttk.Notebook(self._root)
        notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        # ── Build tabs ──────────────────────────────────────────────────
        self._mouse_tab(notebook, cursor_sensitivity, cursor_deadzone,
                        cursor_tap_deadzone, scroll_sensitivity, scroll_deadzone,
                        on_cursor_sensitivity, on_cursor_deadzone,
                        on_cursor_tap_deadzone,
                        on_scroll_sensitivity, on_scroll_deadzone)

        self._gesture_tab(notebook, right_click_mode, shadow_threshold,
                          tap_threshold, tap_min_bend,
                          on_right_click, on_shadow_threshold,
                          on_tap_threshold, on_tap_min_bend)

        self._dtw_tab(notebook, dtw_motion_threshold, dtw_still_frames,
                      dtw_min_frames, dtw_match_threshold, dtw_cooldown_frames,
                      on_dtw_motion, on_dtw_still, on_dtw_min,
                      on_dtw_threshold, on_dtw_cooldown)

        # ── Bottom buttons ──────────────────────────────────────────────
        btn_frame = ttk.Frame(self._root)
        btn_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))
        ttk.Button(btn_frame, text="Save to Config", command=self._on_save_click).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Close", command=self._on_close).pack(side=tk.LEFT)

        self._root.geometry("360x440")
        # Daemon thread for tkinter event loop (avoids pystray conflict)
        import threading
        threading.Thread(target=self._root.mainloop, daemon=True).start()

    # ── helpers ──────────────────────────────────────────────────────────

    def _make_slider(self, parent, label_text, from_val, to_val, initial,
                     callback=None, fmt=".1f", row=0):
        """Create a labeled slider row. Returns (var, label_widget)."""
        var = tk.DoubleVar(value=initial)
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w",
                                                 padx=4, pady=(4, 0))
        scale = ttk.Scale(parent, from_=from_val, to=to_val, variable=var,
                          orient=tk.HORIZONTAL, length=200,
                          command=lambda v, cb=callback, f=fmt: self._on_scale(cb, v, f))
        scale.grid(row=row + 1, column=0, sticky="ew", padx=4)
        label = ttk.Label(parent, text=f"{initial:{fmt}}")
        label.grid(row=row + 1, column=1, padx=(4, 0))
        return var, label

    def _on_scale(self, callback, val, fmt):
        if callback:
            try:
                callback(round(float(val), 3 if '.' in fmt else 0))
            except (ValueError, TypeError):
                pass

    def _make_radio(self, parent, label_text, var, choices, callback, row=0):
        """Create labeled radio group."""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w",
                                                 padx=4, pady=(8, 0))
        frame = ttk.Frame(parent)
        frame.grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=4)
        for text, value in choices:
            ttk.Radiobutton(frame, text=text, variable=var,
                            value=value, command=lambda cb=callback: cb()).pack(
                side=tk.LEFT, padx=(0, 8))

    # ── Mouse tab ────────────────────────────────────────────────────────

    def _mouse_tab(self, nb, cs, cd, ctd, ss, sd, on_cs, on_cd, on_ctd, on_ss, on_sd):
        f = ttk.Frame(nb, padding=(8, 8, 8, 8))
        nb.add(f, text="Mouse")

        row = 0
        _, self._cs_label = self._make_slider(
            f, "Cursor Sensitivity", 0.3, 3.0, cs, on_cs, ".1f", row)
        row += 2
        _, self._cd_label = self._make_slider(
            f, "Cursor Deadzone", 0.0, 0.02, cd, on_cd, ".3f", row)
        row += 2
        _, self._ctd_label = self._make_slider(
            f, "Cursor Tap Deadzone", 0.0, 0.05, ctd, on_ctd, ".3f", row)
        row += 2
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        _, self._ss_label = self._make_slider(
            f, "Scroll Sensitivity", 0.5, 20.0, ss, on_ss, ".1f", row)
        row += 2
        _, self._sd_label = self._make_slider(
            f, "Scroll Deadzone", 0.005, 0.10, sd, on_sd, ".3f", row)

    # ── Gesture tab ──────────────────────────────────────────────────────

    def _gesture_tab(self, nb, rc, sh, tt, tmb, on_rc, on_sh, on_tt, on_tmb):
        f = ttk.Frame(nb, padding=(8, 8, 8, 8))
        nb.add(f, text="Gesture")

        row = 0
        self._rc_var = tk.StringVar(value=rc)
        self._make_radio(f, "Right Click Mode", self._rc_var,
                         [("Fist + Tap", "fist_tap"), ("Two Finger", "two_finger")],
                         lambda: on_rc and on_rc(self._rc_var.get()), row)
        row += 2

        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        _, self._sh_label = self._make_slider(
            f, "ONNX Shadow Threshold", 0.5, 1.0, sh, on_sh, ".0%", row)
        row += 2

        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        _, self._tt_label = self._make_slider(
            f, "Tap Threshold", 0.1, 1.0, tt, on_tt, ".2f", row)
        row += 2
        _, self._tmb_label = self._make_slider(
            f, "Tap Min Bend", 0.05, 0.5, tmb, on_tmb, ".2f", row)

    # ── DTW tab ──────────────────────────────────────────────────────────

    def _dtw_tab(self, nb, dm, ds, dmi, dmt, dc,
                 on_dm, on_ds, on_dmi, on_dmt, on_dc):
        f = ttk.Frame(nb, padding=(8, 8, 8, 8))
        nb.add(f, text="DTW")

        row = 0
        _, self._dm_label = self._make_slider(
            f, "Motion Threshold", 0.001, 0.02, dm, on_dm, ".4f", row)
        row += 2
        _, self._ds_label = self._make_slider(
            f, "Still Frames", 3, 30, float(ds), on_ds, ".0f", row)
        row += 2
        _, self._dmi_label = self._make_slider(
            f, "Min Record Frames", 5, 60, float(dmi), on_dmi, ".0f", row)
        row += 2
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        _, self._dmt_label = self._make_slider(
            f, "Match Threshold", 1.0, 30.0, dmt, on_dmt, ".1f", row)
        row += 2
        _, self._dc_label = self._make_slider(
            f, "Cooldown Frames", 10, 300, float(dc), on_dc, ".0f", row)

    # ── actions ──────────────────────────────────────────────────────────

    def _on_save_click(self):
        if self._on_save:
            self._on_save()

    def _on_close(self):
        self._window_open = False
        self._root.destroy()

    def show(self):
        self._root.deiconify()
        self._root.lift()

    @property
    def is_open(self):
        return self._window_open


# ── factory ────────────────────────────────────────────────────────────────


def open_settings_panel(
    cursor_sensitivity: float = 0.8,
    cursor_deadzone: float = 0.003,
    cursor_tap_deadzone: float = 0.012,
    scroll_sensitivity: float = 2.0,
    scroll_deadzone: float = 0.03,
    right_click_mode: str = "fist_tap",
    shadow_threshold: float = 0.90,
    tap_threshold: float = 0.3,
    tap_min_bend: float = 0.15,
    dtw_motion_threshold: float = 0.005,
    dtw_still_frames: int = 10,
    dtw_min_frames: int = 15,
    dtw_match_threshold: float = 8.0,
    dtw_cooldown_frames: int = 90,
    on_cursor_sensitivity: Callable | None = None,
    on_cursor_deadzone: Callable | None = None,
    on_cursor_tap_deadzone: Callable | None = None,
    on_scroll_sensitivity: Callable | None = None,
    on_scroll_deadzone: Callable | None = None,
    on_right_click: Callable | None = None,
    on_shadow_threshold: Callable | None = None,
    on_tap_threshold: Callable | None = None,
    on_tap_min_bend: Callable | None = None,
    on_dtw_motion: Callable | None = None,
    on_dtw_still: Callable | None = None,
    on_dtw_min: Callable | None = None,
    on_dtw_threshold: Callable | None = None,
    on_dtw_cooldown: Callable | None = None,
    on_save: Callable | None = None,
) -> SettingsPanel:
    panel = SettingsPanel(
        cursor_sensitivity=cursor_sensitivity,
        cursor_deadzone=cursor_deadzone,
        cursor_tap_deadzone=cursor_tap_deadzone,
        scroll_sensitivity=scroll_sensitivity,
        scroll_deadzone=scroll_deadzone,
        right_click_mode=right_click_mode,
        shadow_threshold=shadow_threshold,
        tap_threshold=tap_threshold,
        tap_min_bend=tap_min_bend,
        dtw_motion_threshold=dtw_motion_threshold,
        dtw_still_frames=dtw_still_frames,
        dtw_min_frames=dtw_min_frames,
        dtw_match_threshold=dtw_match_threshold,
        dtw_cooldown_frames=dtw_cooldown_frames,
        on_cursor_sensitivity=on_cursor_sensitivity,
        on_cursor_deadzone=on_cursor_deadzone,
        on_cursor_tap_deadzone=on_cursor_tap_deadzone,
        on_scroll_sensitivity=on_scroll_sensitivity,
        on_scroll_deadzone=on_scroll_deadzone,
        on_right_click=on_right_click,
        on_shadow_threshold=on_shadow_threshold,
        on_tap_threshold=on_tap_threshold,
        on_tap_min_bend=on_tap_min_bend,
        on_dtw_motion=on_dtw_motion,
        on_dtw_still=on_dtw_still,
        on_dtw_min=on_dtw_min,
        on_dtw_threshold=on_dtw_threshold,
        on_dtw_cooldown=on_dtw_cooldown,
        on_save=on_save,
    )
    panel.show()
    return panel
