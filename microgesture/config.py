"""Configuration manager with JSON hot-reload via watchdog."""

import json
import logging
import os
import time as _time
from pathlib import Path
from threading import Lock, Event
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent / "config.json"


class Config:
    """Thread-safe config holder with file-watch reload."""

    def __init__(self, path: Path | None = None, watch: bool = True):
        self._path = Path(path) if path else _DEFAULT_CONFIG
        self._lock = Lock()
        self._data: dict[str, Any] = {}
        self._stop_event = Event()
        self._observer = None
        self._watch = watch
        self.reload()
        if self._watch:
            self._start_watch()

    def reload(self) -> None:
        """Load config from disk with retry for atomic-write safety."""
        for attempt in range(3):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    new_data = json.load(f)
                with self._lock:
                    self._data = new_data
                logger.info("Config loaded from %s", self._path)
                return
            except json.JSONDecodeError:
                if attempt < 2:
                    logger.debug("Config read retry %d/2", attempt + 1)
                    _time.sleep(0.05 * (attempt + 1))
                else:
                    logger.exception("Failed to load config from %s", self._path)  # last attempt
            except Exception:
                logger.exception("Failed to load config from %s", self._path)
                break  # non-JSON errors (permission, missing) — don't retry
        with self._lock:
            if not self._data:
                self._data = {}

    def _start_watch(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            config_self = self  # capture outer Config reference
            _last_reload = 0.0

            class _ReloadHandler(FileSystemEventHandler):
                def on_modified(self, event):
                    nonlocal _last_reload
                    if not event.src_path.endswith(config_self._path.name):
                        return
                    # Debounce: os.replace fires multiple events; reload at most once per 0.5s
                    now = _time.monotonic()
                    if now - _last_reload < 0.5:
                        return
                    _last_reload = now
                    logger.info("Config file changed, reloading...")
                    config_self.reload()

            self._observer = Observer()
            self._observer.schedule(
                _ReloadHandler(), str(self._path.parent), recursive=False
            )
            self._observer.start()
            logger.info("Config file watcher started for %s", self._path)
        except Exception:
            logger.warning("Failed to start config file watcher (watchdog may not be installed)")

    def _stop_watch(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)

    def get(self, *keys: str, default: Any = None) -> Any:
        with self._lock:
            node = self._data
            for k in keys:
                if isinstance(node, dict):
                    node = node.get(k)
                else:
                    return default
            return node if node is not None else default

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def set(self, *keys: str, value: Any) -> None:
        """Thread-safe nested setter. Creates intermediate dicts as needed."""
        with self._lock:
            node = self._data
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    node[k] = {}
                node = node[k]
            node[keys[-1]] = value

    def save(self) -> None:
        """Persist current config atomically to the JSON file."""
        import tempfile

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".config_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                with self._lock:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(self._path))
            logger.info("Config saved to %s", self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def as_dict(self) -> dict:
        """Return a deep copy of all config data."""
        import copy

        with self._lock:
            return copy.deepcopy(self._data)


_config_instance: Config | None = None


def get_config(path: Path | None = None, watch: bool = True) -> Config:
    global _config_instance
    new_path = Path(path) if path else _DEFAULT_CONFIG
    if (
        _config_instance is None
        or _config_instance._path != new_path
        or _config_instance._watch != watch
    ):
        if _config_instance is not None:
            _config_instance._stop_watch()
        _config_instance = Config(path, watch=watch)
    return _config_instance
