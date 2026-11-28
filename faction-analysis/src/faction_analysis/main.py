# pattern: Imperative Shell
"""Entry point: lock, telemetry, the polling loop and the manual commands."""

import argparse
import logging
import os
import signal
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType

from faction_analysis import measure, model_store, pipeline
from faction_analysis.config import AppConfig
from faction_analysis.db import FactionAnalysisDb
from faction_analysis.lock import AlreadyRunning, InstanceLock
from faction_analysis.promotion import IntentUnreadable
from faction_analysis.telemetry import setup_telemetry

logger = logging.getLogger(__name__)
EXIT_REFUSED = 3
LOCK_NAME = 'faction-analysis.lock'
_shutdown = threading.Event()


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    logger.info('signal received, stopping after this poll', extra={'signal': signum})
    _shutdown.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m faction_analysis.main', description=__doc__.splitlines()[0])
    parser.add_argument('--once', action='store_true', help='run one poll and exit')
    parser.add_argument('--kind', choices=['weekly', 'full'], help='force the kind of run for this poll')
    parser.add_argument('--dry-run', action='store_true', help='compute everything and write nothing')
    parser.add_argument('--measure', action='store_true', help='run the production measurement instead')
    return parser


SECRET_FILE = Path('/run/secrets/typesafe_api_key')


def api_key(environ: Mapping[str, str] | None = None, secret_file: Path = SECRET_FILE) -> str:
    """The TypeSafe key, from the environment or from the secret file a compose service mounts.

    Either source is enough. The environment wins when both are present. The key is never logged.
    """
    from_environment = (environ if environ is not None else os.environ).get('TYPESAFE_API_KEY', '').strip()
    if from_environment:
        return from_environment
    try:
        return secret_file.read_text().strip()
    except OSError:
        return ''


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--measure' in arguments:
        return measure.main(arguments[arguments.index('--measure') + 1 :])
    args = build_parser().parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    config = AppConfig.from_env()
    telemetry = setup_telemetry(config.telemetry)
    model_dir = Path(config.storage.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    instance = InstanceLock(model_dir / LOCK_NAME)
    try:
        instance.acquire()
    except AlreadyRunning as exc:
        logger.error('%s', exc)
        telemetry.shutdown()
        return EXIT_REFUSED
    client: FactionAnalysisDb | None = None
    try:
        client = FactionAnalysisDb(config.clickhouse, config.query)
        repo: pipeline.RunRepository = pipeline.recording(client) if args.dry_run else client
        context = pipeline.RunContext(
            config,
            repo,
            model_dir,
            trainer=pipeline.default_trainer(api_key()),
            telemetry=telemetry,
            dry_run=args.dry_run,
        )
        forced = args.kind
        if args.once or forced is not None or args.dry_run:
            outcome = pipeline.poll(context, kind=forced)
            _log_outcome(outcome)
            return outcome.exit_code
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
        while not _shutdown.is_set():
            outcome = pipeline.poll(context)
            _log_outcome(outcome)
            if outcome.status in {'stopped', 'unresolved'}:
                return outcome.exit_code
            _shutdown.wait(config.schedule.poll_interval_seconds)
        return 0
    except IntentUnreadable as exc:
        logger.error('the promotion-intent file cannot be read; nothing is cleaned up: %s', exc)
        return 1
    except model_store.ModelStoreError as exc:
        logger.error('the model store is not usable: %s', exc)
        return 1
    finally:
        if client is not None:
            client.close()
        telemetry.shutdown()
        instance.release()


def _log_outcome(outcome: pipeline.RunOutcome) -> None:
    logger.info(
        'poll finished',
        extra={'status': outcome.status, 'stage': outcome.stage, 'exit_code': outcome.exit_code},
    )


if __name__ == '__main__':
    raise SystemExit(main())
