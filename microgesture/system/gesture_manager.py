"""Gesture template manager — view, delete, register custom DTW gestures."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable


class GestureManager:
    """Tkinter window for managing DTW gesture templates."""

    def __init__(
        self,
        *,
        get_templates: Callable[[], list[dict]],
        get_state: Callable[[], str],
        get_buffer: Callable[[], int],
        on_register: Callable[[str, str, dict], None] | None = None,
        on_delete: Callable[[str], None] | None = None,
        on_refresh: Callable[[], None] | None = None,
    ):
        self._get_templates = get_templates
        self._get_state = get_state
        self._get_buffer = get_buffer
        self._on_register = on_register
        self._on_delete = on_delete
        self._on_refresh = on_refresh

        self._root = tk.Tk()
        self._root.title("Gesture Manager")
        self._root.resizable(True, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._window_open = True

        f = ttk.Frame(self._root, padding=(12, 8, 12, 8))
        f.grid(row=0, column=0, sticky="nsew")

        # ── Status bar ──────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="IDLE")
        ttk.Label(f, textvariable=self._status_var, font=("", 9, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        # ── Template list ───────────────────────────────────────────────
        cols = ("name", "label", "frames", "action")
        self._tree = ttk.Treeview(f, columns=cols, show="headings", height=8)
        self._tree.heading("name", text="Name")
        self._tree.heading("label", text="Label")
        self._tree.heading("frames", text="Frames")
        self._tree.heading("action", text="Action")
        self._tree.column("name", width=100)
        self._tree.column("label", width=80)
        self._tree.column("frames", width=60)
        self._tree.column("action", width=100)
        self._tree.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 6))

        # ── Buttons ─────────────────────────────────────────────────────
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        ttk.Button(btn_frame, text="Register New", command=self._on_register_click).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Delete", command=self._on_delete_click).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Refresh", command=self._refresh).pack(
            side=tk.LEFT)

        # ── Register form ───────────────────────────────────────────────
        reg = ttk.LabelFrame(f, text="Register New Gesture", padding=(8, 4, 8, 4))
        reg.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 6))

        ttk.Label(reg, text="Name:").grid(row=0, column=0, sticky="w")
        self._name_entry = ttk.Entry(reg, width=15)
        self._name_entry.grid(row=0, column=1, sticky="w", padx=(4, 8))
        self._name_entry.insert(0, "my_gesture")

        ttk.Label(reg, text="Label:").grid(row=0, column=2, sticky="w")
        self._label_entry = ttk.Entry(reg, width=12)
        self._label_entry.grid(row=0, column=3, sticky="w", padx=(4, 8))

        ttk.Label(reg, text="Key:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._key_combo = ttk.Combobox(reg, width=20,
                                       values=["ctrl+c", "ctrl+v", "ctrl+z", "ctrl+s",
                                               "win+d", "win+e", "win+r",
                                               "alt+tab", "alt+f4",
                                               "f5", "f11", "enter", "space"])
        self._key_combo.grid(row=1, column=1, columnspan=3, sticky="w",
                             padx=(4, 8), pady=(4, 0))
        self._key_combo.set("ctrl+s")

        ttk.Button(reg, text="Start Recording",
                   command=self._on_start_register).grid(
            row=2, column=0, columnspan=4, pady=(8, 0))

        self._root.geometry("420x480")
        self._refresh()
        self._poll()
        # Daemon thread for tkinter event loop (avoids pystray conflict)
        import threading
        threading.Thread(target=self._root.mainloop, daemon=True).start()

    # ── refresh ──────────────────────────────────────────────────────────

    def _refresh(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for t in self._get_templates():
            action = t.get("action", {})
            action_str = ""
            if action.get("type") == "key_combo":
                mods = "+".join(action.get("modifiers", []))
                key = action.get("key", "")
                action_str = f"{mods}+{key}" if mods else key
            frames = len(t.get("sequence", []))
            self._tree.insert("", tk.END, values=(
                t["name"], t["label"], frames, action_str))

    def _poll(self):
        """Update status bar with live DTW state."""
        if not self._window_open:
            return
        state = self._get_state()
        buf = self._get_buffer()
        self._status_var.set(f"DTW: {state}  |  Buffer: {buf}")
        if self._on_refresh:
            self._on_refresh()
        self._root.after(500, self._poll)

    # ── actions ──────────────────────────────────────────────────────────

    def _on_register_click(self):
        sel = self._tree.selection()
        if sel:
            name = self._tree.item(sel[0], "values")[0]
            self._name_entry.delete(0, tk.END)
            self._name_entry.insert(0, f"{name}_v2")

    def _on_delete_click(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a gesture to delete.")
            return
        name = self._tree.item(sel[0], "values")[0]
        if messagebox.askyesno("Confirm", f"Delete '{name}'?"):
            if self._on_delete:
                self._on_delete(name)
            self._refresh()

    def _on_start_register(self):
        name = self._name_entry.get().strip()
        label = self._label_entry.get().strip() or name
        if not name:
            messagebox.showwarning("Missing Name", "Enter a gesture name.")
            return
        # Parse key combo
        combo_text = self._key_combo.get().strip()
        modifiers, key = self._parse_combo(combo_text)

        if self._on_register:
            self._on_register(name, label, {
                "type": "key_combo",
                "modifiers": modifiers,
                "key": key,
            })
            messagebox.showinfo("Recording",
                f"Gesture '{label}' registration started.\n"
                f"Hold your hand still, then perform the gesture, then pause.\n"
                f"Repeat 3 times.")

    @staticmethod
    def _parse_combo(text: str) -> tuple[list[str], str]:
        parts = text.lower().split("+")
        mods = [p for p in parts[:-1] if p in ("ctrl", "alt", "shift", "win")]
        key = parts[-1] if parts else ""
        return mods, key

    def _on_close(self):
        self._window_open = False
        self._root.destroy()

    # ── lifecycle ────────────────────────────────────────────────────────

    def show(self):
        self._root.deiconify()
        self._root.lift()

    @property
    def is_open(self):
        return self._window_open


def open_gesture_manager(
    get_templates: Callable[[], list[dict]],
    get_state: Callable[[], str],
    get_buffer: Callable[[], int],
    on_register: Callable | None = None,
    on_delete: Callable | None = None,
) -> GestureManager:
    return GestureManager(
        get_templates=get_templates,
        get_state=get_state,
        get_buffer=get_buffer,
        on_register=on_register,
        on_delete=on_delete,
    )
