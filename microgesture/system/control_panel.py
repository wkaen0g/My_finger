"""Unified control panel — settings, gestures, diagnostics in one window.

Uses a shared Tk root from main.py. All sub-windows are Toplevel.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable


class ControlPanel:
    """Tabbed control panel with Mouse, Gesture, DTW, and System tabs."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        # ── System ──
        fps: float = 0.0,
        inference_source: str = "rule",
        onnx_conf: float = 0.0,
        tracking_enabled: bool = True,
        camera_connected: bool = True,
        dtw_state: str = "IDLE",
        dtw_buffer: int = 0,
        template_count: int = 0,
        # ── Mouse ──
        cursor_sensitivity: float = 0.8,
        cursor_deadzone: float = 0.003,
        cursor_tap_deadzone: float = 0.012,
        scroll_sensitivity: float = 2.0,
        scroll_deadzone: float = 0.03,
        # ── Gesture ──
        right_click_mode: str = "fist_tap",
        shadow_threshold: float = 0.90,
        tap_threshold: float = 0.3,
        tap_min_bend: float = 0.15,
        # ── DTW ──
        dtw_motion_threshold: float = 0.005,
        dtw_still_frames: int = 10,
        dtw_min_frames: int = 15,
        dtw_match_threshold: float = 8.0,
        dtw_cooldown_frames: int = 90,
        # ── Callbacks ──
        on_toggle_tracking: Callable[[], None] | None = None,
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
        on_register_gesture: Callable[[str, str, dict], None] | None = None,
        on_delete_gesture: Callable[[str], None] | None = None,
        on_save: Callable[[], None] | None = None,
        get_templates: Callable[[], list[dict]] | None = None,
        is_training: Callable[[], bool] | None = None,
    ):
        self._root = root
        self._on_save = on_save
        self._on_toggle_tracking = on_toggle_tracking
        self._on_register = on_register_gesture
        self._on_delete = on_delete_gesture
        self._get_templates = get_templates
        self._is_training = is_training
        self._window_open = True

        win = tk.Toplevel(root)
        win.title("MicroGesture Control Panel")
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._win = win

        # ── Notebook ────────────────────────────────────────────────────
        nb = ttk.Notebook(win)
        nb.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        self._system_tab(nb, fps, inference_source, onnx_conf,
                         tracking_enabled, camera_connected,
                         dtw_state, dtw_buffer, template_count)

        self._slider_tab(nb, "Mouse", [
            ("Cursor Sensitivity", 0.3, 3.0, cursor_sensitivity, on_cursor_sensitivity, ".1f"),
            ("Cursor Deadzone", 0.0, 0.02, cursor_deadzone, on_cursor_deadzone, ".3f"),
            ("Tap Deadzone", 0.0, 0.05, cursor_tap_deadzone, on_cursor_tap_deadzone, ".3f"),
            ("Scroll Sensitivity", 0.5, 20.0, scroll_sensitivity, on_scroll_sensitivity, ".1f"),
            ("Scroll Deadzone", 0.005, 0.10, scroll_deadzone, on_scroll_deadzone, ".3f"),
        ])

        self._gesture_tab(nb, right_click_mode, shadow_threshold,
                          tap_threshold, tap_min_bend,
                          on_right_click, on_shadow_threshold,
                          on_tap_threshold, on_tap_min_bend)

        self._dtw_tab(nb, dtw_motion_threshold, dtw_still_frames,
                      dtw_min_frames, dtw_match_threshold, dtw_cooldown_frames,
                      on_dtw_motion, on_dtw_still, on_dtw_min,
                      on_dtw_threshold, on_dtw_cooldown)

        # ── Bottom bar ──────────────────────────────────────────────────
        bar = ttk.Frame(win)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))
        ttk.Button(bar, text="Save Config", command=self._on_save_click).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(bar, text="Close", command=self._on_close).pack(side=tk.LEFT)

        win.geometry("520x620")

        # ── Poll for live stats ─────────────────────────────────────────
        self._poll()

    # ── helpers ──────────────────────────────────────────────────────────

    def _make_slider(self, parent, label_text, from_val, to_val, initial,
                     callback, fmt, row):
        var = tk.DoubleVar(value=initial)
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w",
                                                 padx=4, pady=(4, 0))
        lbl = ttk.Label(parent, text=f"{initial:{fmt}}")
        ttk.Scale(parent, from_=from_val, to=to_val, variable=var,
                  orient=tk.HORIZONTAL, length=300,
                  command=lambda v, cb=callback, l=lbl, f=fmt:
                  self._on_slider(cb, v, l, f)
                  ).grid(row=row + 1, column=0, sticky="ew", padx=4)
        lbl.grid(row=row + 1, column=1, padx=(4, 0))

    @staticmethod
    def _on_slider(callback, val, lbl, fmt):
        v = float(val)
        lbl.configure(text=f"{v:{fmt}}")
        if callback:
            # Integer callbacks get rounded values
            is_int = ".0" in fmt
            callback(int(round(v)) if is_int else round(v, 3))

    # ── System tab ───────────────────────────────────────────────────────

    def _system_tab(self, nb, fps, source, conf, tracking, camera,
                    dtw_st, dtw_buf, tpl_count):
        f = ttk.Frame(nb, padding=(8, 8, 8, 8))
        nb.add(f, text="System")

        row = 0
        # Status grid
        self._sys_labels: dict[str, ttk.Label] = {}
        for label, var_name, fmt_str in [
            ("FPS", "fps", "{:.0f}"),
            ("Inference", "source", "{}"),
            ("ONNX Confidence", "conf", "{:.0%}"),
            ("Camera", "camera", "{}"),
            ("Tracking", "tracking", "{}"),
            ("DTW State", "dtw_state", "{}"),
            ("DTW Buffer", "dtw_buffer", "{}"),
            ("Templates", "templates", "{}"),
        ]:
            ttk.Label(f, text=label + ":").grid(row=row, column=0, sticky="w",
                                                 padx=4, pady=1)
            lbl = ttk.Label(f, text="--", font=("", 9, "bold"))
            lbl.grid(row=row, column=1, sticky="w", padx=4, pady=1)
            self._sys_labels[var_name] = lbl
            row += 1

        row += 1
        self._track_btn = ttk.Button(f, text="Toggle Tracking",
                                     command=self._on_toggle_tracking)
        self._track_btn.grid(row=row, column=0, columnspan=2, pady=(8, 0))

    def _update_system_stats(self):
        """Called from pipeline to push live stats (via polling callback)."""
        pass  # stats pushed via _poll refresh callback

    # ── Gesture tab ──────────────────────────────────────────────────────

    def _gesture_tab(self, nb, rc, sh, tt, tmb,
                     on_rc, on_sh, on_tt, on_tmb):
        f = ttk.Frame(nb, padding=(8, 8, 8, 8))
        nb.add(f, text="Gesture")

        row = 0
        self._rc_var = tk.StringVar(value=rc)
        ttk.Label(f, text="Right Click Mode:").grid(row=row, column=0, sticky="w",
                                                     padx=4, pady=(8, 0))
        rcf = ttk.Frame(f)
        rcf.grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Radiobutton(rcf, text="Fist + Tap", variable=self._rc_var,
                        value="fist_tap", command=lambda: on_rc and on_rc("fist_tap")
                        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(rcf, text="Two Finger", variable=self._rc_var,
                        value="two_finger", command=lambda: on_rc and on_rc("two_finger")
                        ).pack(side=tk.LEFT)
        row += 2

        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        for lbl, frm, to, val, cb, fmt in [
            ("ONNX Threshold", 0.5, 1.0, sh, on_sh, ".0%"),
            ("Tap Threshold", 0.1, 1.0, tt, on_tt, ".2f"),
            ("Tap Min Bend", 0.05, 0.5, tmb, on_tmb, ".2f"),
        ]:
            self._make_slider(f, lbl, frm, to, val, cb, fmt, row)
            row += 2

    # ── Slider helper tab ────────────────────────────────────────────────

    def _slider_tab(self, nb, title, sliders):
        f = ttk.Frame(nb, padding=(8, 8, 8, 8))
        nb.add(f, text=title)
        row = 0
        for lbl, frm, to, val, cb, fmt in sliders:
            self._make_slider(f, lbl, frm, to, val, cb, fmt, row)
            row += 2

    # ── DTW tab ──────────────────────────────────────────────────────────

    def _dtw_tab(self, nb, dm, ds, dmi, dmt, dc,
                 on_dm, on_ds, on_dmi, on_dmt, on_dc):
        f = ttk.Frame(nb, padding=(8, 8, 8, 8))
        nb.add(f, text="DTW")

        row = 0
        for lbl, frm, to, val, cb, fmt in [
            ("Motion Threshold", 0.001, 0.02, dm, on_dm, ".4f"),
            ("Still Frames", 3, 30, float(ds), on_ds, ".0f"),
            ("Min Frames", 5, 60, float(dmi), on_dmi, ".0f"),
        ]:
            self._make_slider(f, lbl, frm, to, val, cb, fmt, row)
            row += 2

        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        for lbl, frm, to, val, cb, fmt in [
            ("Match Threshold", 1.0, 30.0, dmt, on_dmt, ".1f"),
            ("Cooldown Frames", 10, 300, float(dc), on_dc, ".0f"),
        ]:
            self._make_slider(f, lbl, frm, to, val, cb, fmt, row)
            row += 2

        # ── Template list ────────────────────────────────────────────
        ttk.Separator(f, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Label(f, text="Templates:").grid(row=row, column=0, sticky="w",
                                              padx=4, pady=(0, 2))
        row += 1

        cols = ("name", "action")
        self._tree = ttk.Treeview(f, columns=cols, show="headings", height=4)
        self._tree.heading("name", text="Name")
        self._tree.heading("action", text="Action")
        self._tree.column("name", width=120)
        self._tree.column("action", width=120)
        self._tree.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4)
        row += 1

        btnf = ttk.Frame(f)
        btnf.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        self._reg_btn = ttk.Button(btnf, text="Register New",
                                   command=self._on_register_click)
        self._reg_btn.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btnf, text="Delete", command=self._on_delete_click).pack(
            side=tk.LEFT)
        row += 1

        self._train_status = ttk.Label(f, text="", foreground="blue")
        self._train_status.grid(row=row, column=0, columnspan=2, sticky="w",
                                padx=4, pady=(4, 0))

        self._refresh_templates()

    def _refresh_templates(self):
        if self._tree is None:
            return
        for item in self._tree.get_children():
            self._tree.delete(item)
        if self._get_templates:
            for t in self._get_templates():
                action = t.get("action", {})
                action_str = ""
                if action.get("type") == "key_combo":
                    mods = "+".join(action.get("modifiers", []))
                    key = action.get("key", "")
                    action_str = f"{mods}+{key}" if mods else key
                self._tree.insert("", tk.END, values=(t["name"], action_str))

    def _on_register_click(self):
        if self._on_register:
            import time
            name = f"gesture_{time.strftime('%H%M%S')}"
            self._on_register(name, name, {"type": "key_combo", "modifiers": ["ctrl"], "key": "s"})
            self._reg_btn.configure(state="disabled", text="训练中...")
            self._train_status.configure(text="准备... (请保持静止)")
            self._poll_training()

    def _poll_training(self):
        """Check if training is complete, update status, re-enable button."""
        if not self._window_open:
            return
        if hasattr(self, '_is_training') and self._is_training():
            # Still training — poll again
            self._win.after(500, self._poll_training)
        else:
            # Training complete
            self._reg_btn.configure(state="normal", text="Register New")
            self._train_status.configure(text="Training complete!")
            self._refresh_templates()

    def _on_delete_click(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a gesture to delete.")
            return
        name = self._tree.item(sel[0], "values")[0]
        if messagebox.askyesno("Confirm", f"Delete '{name}'?"):
            if self._on_delete:
                self._on_delete(name)
            self._refresh_templates()

    def _on_toggle_tracking(self):
        if self._on_toggle_tracking:
            self._on_toggle_tracking()

    # ── polling ──────────────────────────────────────────────────────────

    def _poll(self):
        if not self._window_open:
            return
        self._win.after(200, self._poll)

    # ── actions ──────────────────────────────────────────────────────────

    def _on_save_click(self):
        if self._on_save:
            self._on_save()
        self._refresh_templates()

    def _on_close(self):
        self._window_open = False
        self._win.destroy()

    @property
    def is_open(self):
        return self._window_open


def open_control_panel(
    root: tk.Tk,
    **kwargs,
) -> ControlPanel:
    return ControlPanel(root, **kwargs)
