"""Dreaming pipeline CLI commands (lightclaw dream ...)."""

from __future__ import annotations

import click

from lightclaw.cli.http import authed_client


def _server_base_url(ctx: click.Context) -> str:
    host = ctx.obj.get("host", "127.0.0.1")
    port = ctx.obj.get("port", 8088)
    return f"http://{host}:{port}"


@click.group("dream")
def dream_group() -> None:
    """Manage dreaming pipeline (3-phase nightly memory consolidation)."""


@dream_group.command("status")
@click.pass_context
def dream_status(ctx: click.Context) -> None:
    """Show dreaming pipeline status."""
    url = _server_base_url(ctx)
    try:
        with authed_client(url) as c:
            resp = c.get("/cron/dreaming/status")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise click.ClickException(str(e)) from e
    click.echo("Dreaming Pipeline Status")
    click.echo("=" * 40)
    for key, val in data.items():
        click.echo(f"  {key}: {val}")


@dream_group.command("run")
@click.option("--dry-run/--no-dry-run", default=False, help="Preview only, no files written")
@click.pass_context
def dream_run(ctx: click.Context, dry_run: bool) -> None:
    """Run dreaming pipeline immediately."""
    url = _server_base_url(ctx)
    try:
        with authed_client(url) as c:
            resp = c.post(
                "/cron/dreaming/run",
                params={"dry_run": "true" if dry_run else "false"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Status: {data.get('status')}")
    click.echo(f"Dry run: {data.get('dry_run')}")
    if data.get("message"):
        click.echo(data["message"])
    report = data.get("report_text", "")
    if report:
        click.echo("\n" + report)
    else:
        click.echo("Check logs for progress (dreaming runs asynchronously).")


@dream_group.command("light-ingest")
@click.option("--dry-run/--no-dry-run", default=False, help="Preview only, no files written")
@click.pass_context
def dream_light_ingest(ctx: click.Context, dry_run: bool) -> None:
    """Run light-only ingest (Light Phase only, no REM/Deep, no memory writes)."""
    url = _server_base_url(ctx)
    try:
        with authed_client(url) as c:
            resp = c.post(
                "/cron/dreaming/light-ingest",
                params={"dry_run": "true" if dry_run else "false"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Status: {data.get('status')}")
    click.echo(f"Dry run: {data.get('dry_run')}")
    click.echo(f"Candidates total: {data.get('candidates_total', 'N/A')}")
    click.echo(f"Candidates kept: {data.get('candidates_kept', 'N/A')}")
    if data.get("errors"):
        click.echo(f"Errors: {', '.join(data['errors'])}")


@dream_group.command("report")
@click.option("-n", "--lines", default=80, type=int, help="Number of lines to show")
def dream_report(lines: int) -> None:
    """Show latest DREAMS.md."""
    from lightclaw.constant import WORKING_DIR

    path = WORKING_DIR / "DREAMS.md"
    if not path.is_file():
        click.echo("No DREAMS.md found.")
        return
    text = path.read_text(encoding="utf-8")
    all_lines = text.splitlines()
    if len(all_lines) <= lines:
        click.echo(text)
    else:
        click.echo("\n".join(all_lines[-lines:]))
        click.echo(f"\n... ({len(all_lines)} lines total, showing last {lines})")
