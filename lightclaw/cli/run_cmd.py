from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

import click

from lightclaw.config.utils import BASE_DIR, read_last_api, write_last_api
from lightclaw.constant import LOG_LEVEL_ENV


# Lazy-import aliases — defined here so unit tests can patch them.
def _load_config():
    from lightclaw.config import load_config as _lc

    return _lc()


@click.command("run")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host",
)
@click.option(
    "--port",
    default=8088,
    type=int,
    show_default=True,
    help="Bind port",
)
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev only)")
@click.option(
    "--from-configfile",
    is_flag=True,
    help="Read host and port from lightclaw.json instead of using defaults.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    show_default=True,
    help="Worker processes",
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(
        ["critical", "error", "warning", "info", "debug", "trace"],
        case_sensitive=False,
    ),
    show_default=True,
    help="Log level",
)
@click.option(
    "--hide-access-paths",
    multiple=True,
    default=("/dashboard/messages",),
    show_default=True,
    help="Path substrings to hide from uvicorn access log (repeatable).",
)
@click.option(
    "--ssl",
    is_flag=True,
    help="Enable HTTPS. Auto-generates a self-signed certificate if --ssl-certfile is not provided.",
)
@click.option(
    "--ssl-certfile",
    default=None,
    help="Path to TLS certificate file (PEM). Used together with --ssl.",
)
@click.option(
    "--ssl-keyfile",
    default=None,
    help="Path to TLS private key file (PEM). Used together with --ssl.",
)
def app_cmd(
    host: str,
    port: int,
    reload: bool,
    from_configfile: bool,
    workers: int,
    log_level: str,
    hide_access_paths: tuple[str, ...],
    ssl: bool,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
) -> None:
    """Run LightClaw FastAPI app."""
    import uvicorn

    _PID_FILE = Path(BASE_DIR) / "lightclaw.pid"

    def _pid_alive(pid: int) -> bool:
        """Check if a process with given PID is a running lightclaw instance."""
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_text()
            return "lightclaw" in cmdline
        except (FileNotFoundError, PermissionError):
            return False

    if _PID_FILE.exists():
        try:
            old_pid = int(_PID_FILE.read_text().strip())
            if old_pid != os.getpid() and _pid_alive(old_pid):
                click.echo(
                    f"Error: Another LightClaw instance is already running (PID {old_pid}).\n"
                    f"Stop it first with: lightclaw stop\n"
                    f"Or remove stale lock with: rm -f {_PID_FILE}"
                )
                raise SystemExit(1)
        except (ValueError, SystemExit):
            raise
        except Exception:
            pass  # corrupted PID file, overwrite it

    _PID_FILE.write_text(str(os.getpid()))

    def _cleanup_pid():
        try:
            if _PID_FILE.exists() and int(_PID_FILE.read_text().strip()) == os.getpid():
                _PID_FILE.unlink()
        except Exception:
            pass

    atexit.register(_cleanup_pid)

    from lightclaw.utils.logging import SuppressPathAccessLogFilter, setup_logger

    if from_configfile:
        saved = read_last_api()
        if saved:
            host, port, ssl, ssl_certfile, ssl_keyfile = saved

    if ssl and not ssl_certfile:
        from lightclaw.cli.ssl_utils import generate_self_signed_cert

        ssl_dir = Path(BASE_DIR) / "ssl"
        _cert = ssl_dir / "self_signed.crt"
        _key = ssl_dir / "self_signed.key"
        if not _cert.exists() or not _key.exists():
            generate_self_signed_cert(_cert, _key, host)
            click.echo(f"Self-signed certificate generated: {_cert}")
        ssl_certfile = str(_cert)
        ssl_keyfile = str(_key)

    write_last_api(host, port, ssl=ssl, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
    os.environ[LOG_LEVEL_ENV] = log_level
    setup_logger(log_level)
    if log_level in ("debug", "trace"):
        from lightclaw.cli.main import log_init_timings

        log_init_timings()

    paths = [p for p in hide_access_paths if p]
    if paths:
        logging.getLogger("uvicorn.access").addFilter(
            SuppressPathAccessLogFilter(paths),
        )

    uvicorn.run(
        "lightclaw.app._app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        ssl_certfile=ssl_certfile or None,
        ssl_keyfile=ssl_keyfile or None,
    )
