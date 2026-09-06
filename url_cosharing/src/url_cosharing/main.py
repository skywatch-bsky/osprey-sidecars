# pattern: Imperative Shell
from __future__ import annotations

import logging
import re
import signal
import time
from datetime import date, datetime

from url_cosharing.analyzer import (
    cluster_core,
    compute_evolution,
    compute_temporal_metrics,
)
from url_cosharing.config import AppConfig, Exclusions, load_exclusions, load_exclusions_with_fallback
from url_cosharing.db import CosharingDb, MembershipRow, RunMetadata
from url_cosharing.dismantling import dismantle
from url_cosharing.queries import (
    fetch_excluded_shares_count_query,
    fetch_historical_membership_query,
    fetch_member_timestamps_query,
    fetch_raw_account_count_query,
    fetch_url_shares_query,
)
from url_cosharing.similarity import similarity_network
from url_cosharing.telemetry import (
    TelemetryHandles,
    noop_telemetry,
    record_failure,
    record_run_metrics,
    set_run_attributes,
    setup_telemetry,
    stage_span,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('url_cosharing')

_shutdown = False


def _sanitize_did(did: str) -> str:
    """Remove non-allowed characters from DID for defence-in-depth SQL safety.

    DIDs should match pattern did:plc:[a-z0-9]+. Keep only these characters.
    """
    return re.sub(r'[^a-z0-9:.]', '', did)


def _latest_previous_membership(history_rows: list[MembershipRow]) -> dict[str, frozenset[str]]:
    """Build previous membership from each cluster's most recent snapshot only.

    Unioning rows across the whole evolution window would accumulate departed
    members, depressing Jaccard similarity against today's membership and
    misclassifying normal churn as births, deaths, or splits.
    """
    latest_run_date: dict[str, date] = {}
    for row in history_rows:
        current = latest_run_date.get(row.cluster_id)
        if current is None or row.run_date > current:
            latest_run_date[row.cluster_id] = row.run_date

    members: dict[str, set[str]] = {}
    for row in history_rows:
        if row.run_date == latest_run_date[row.cluster_id]:
            members.setdefault(row.cluster_id, set()).add(row.did)

    return {cluster_id: frozenset(dids) for cluster_id, dids in members.items()}


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    logger.info(f'received signal {signum}, shutting down after current cycle')
    _shutdown = True


def run_cycle(
    db: CosharingDb,
    config: AppConfig,
    run_date: date | None = None,
    telemetry: TelemetryHandles | None = None,
    exclusions: Exclusions = Exclusions.empty(),
) -> None:
    """Compute and persist one day's detection results.

    run_date defaults to today (the daemon path). Passing a past date
    recomputes that day: the detection window is the window_days days ending
    the day before run_date, and existing rows for run_date are overwritten
    via the idempotent delete-then-insert below.

    exclusions are applied at the SQL eligibility layer (inside the share
    query, before df/eligibility/mono-URL computation). When non-empty, the
    suppressed share-row count is fetched and recorded alongside the applied
    revision in the run metadata audit columns.
    """
    if run_date is None:
        run_date = date.today()
    telemetry = telemetry or noop_telemetry()
    analysis = config.analysis
    run_start = time.monotonic()

    with telemetry.tracer.start_as_current_span(
        'url_cosharing.run_cycle',
        attributes={'run_date': run_date.isoformat(), 'window_days': analysis.window_days},
        record_exception=False,
        set_status_on_exception=False,
    ) as root_span:
        try:
            logger.info('fetching url shares')
            with stage_span(telemetry, 'url_cosharing.fetch_url_shares'):
                rows = db.fetch_url_shares(fetch_url_shares_query(analysis, run_date, exclusions))
            logger.info(f'fetched {len(rows)} share rows')
            if not rows:
                logger.warning(
                    'fetched 0 share rows; verify source data volume and URL eligibility '
                    'filters (min_url_sharers, max_url_df_fraction)'
                )

            with stage_span(telemetry, 'url_cosharing.fetch_raw_account_count'):
                accounts_raw = db.fetch_raw_account_count(fetch_raw_account_count_query(analysis, run_date))

            excluded_shares_suppressed = 0
            if not exclusions.is_empty():
                with stage_span(telemetry, 'url_cosharing.fetch_excluded_shares_count'):
                    excluded_shares_suppressed = db.fetch_excluded_shares_count(
                        fetch_excluded_shares_count_query(analysis, run_date, exclusions)
                    )
                logger.info(f'exclusions suppressed {excluded_shares_suppressed} share rows')

            with stage_span(telemetry, 'url_cosharing.build_similarity_network'):
                network = similarity_network(rows, analysis.edge_epsilon)
            set_run_attributes(
                root_span,
                {
                    'accounts_raw': accounts_raw,
                    'accounts_eligible': network.accounts_eligible,
                    'urls_eligible': network.urls_eligible,
                    'graph_edges': network.graph_edges,
                },
            )
            logger.info(
                f'similarity network: {network.accounts_eligible}/{accounts_raw} accounts, '
                f'{network.urls_eligible} urls, {network.graph_edges} edges'
            )

            with stage_span(telemetry, 'url_cosharing.dismantle'):
                result = dismantle(
                    network.graph,
                    analysis.edge_quantile_grid,
                    analysis.centrality_quantile_grid,
                    analysis.density_floor,
                    analysis.max_flagged_fraction,
                    analysis.max_flagged_accounts,
                    analysis.min_cluster_size,
                    logger,
                )

            with stage_span(telemetry, 'url_cosharing.cluster_core'):
                clusters = cluster_core(
                    result.core, network.matrix, network.tfidf, analysis.resolution, analysis.min_cluster_size
                )
            logger.info(f'found {len(clusters)} clusters (knee_found={result.knee_found})')

            logger.info('gathering member timestamps')
            all_dids = set()
            for cluster in clusters:
                all_dids.update(cluster.members)

            with stage_span(telemetry, 'url_cosharing.fetch_member_timestamps'):
                if all_dids:
                    sanitized_dids = [_sanitize_did(did) for did in sorted(all_dids)]
                    dids_placeholder = ','.join(f"'{did}'" for did in sanitized_dids)
                    timestamps_query = fetch_member_timestamps_query(analysis, dids_placeholder, run_date)
                    timestamp_rows = db.fetch_member_timestamps(timestamps_query)

                    member_timestamps: dict[str, list[datetime]] = {}
                    for row in timestamp_rows:
                        if row.did not in member_timestamps:
                            member_timestamps[row.did] = []
                        member_timestamps[row.did].append(row.ts)
                else:
                    member_timestamps = {}

            logger.info('computing temporal metrics')
            with stage_span(telemetry, 'url_cosharing.compute_temporal_metrics'):
                timestamped_clusters = [compute_temporal_metrics(cluster, member_timestamps) for cluster in clusters]

            logger.info('gathering historical membership')
            with stage_span(telemetry, 'url_cosharing.fetch_historical_membership'):
                history_query = fetch_historical_membership_query(analysis, run_date)
                history_rows = db.fetch_historical_membership(history_query)

            previous_membership = _latest_previous_membership(history_rows)

            logger.info('computing evolution')
            with stage_span(telemetry, 'url_cosharing.compute_evolution'):
                events = compute_evolution(
                    clusters,
                    previous_membership,
                    run_date,
                    analysis.jaccard_threshold,
                )

            logger.info('clearing stale data for today (idempotent re-run guard)')
            with stage_span(telemetry, 'url_cosharing.delete_stale_run_date'):
                db.delete_run_date(analysis.clusters_table, run_date)
                db.delete_run_date(analysis.membership_table, run_date)
                db.delete_run_date(analysis.runs_table, run_date)

            logger.info('persisting results')
            cluster_rows = [
                (run_date, ts_cluster, event)
                for ts_cluster, event in zip(timestamped_clusters, events)
                if event.evolution_type != 'death'
            ]
            with stage_span(telemetry, 'url_cosharing.persist_clusters'):
                db.insert_clusters(analysis.clusters_table, cluster_rows)
            logger.info(f'wrote {len(cluster_rows)} cluster results to {analysis.clusters_table}')

            membership_rows = []
            for ts_cluster, event in zip(timestamped_clusters, events):
                if event.evolution_type != 'death':
                    for did in ts_cluster.members:
                        membership_rows.append((run_date, event.cluster_id, did))

            with stage_span(telemetry, 'url_cosharing.persist_membership'):
                db.insert_membership(analysis.membership_table, membership_rows)
            logger.info(f'wrote {len(membership_rows)} membership rows to {analysis.membership_table}')

            run_metadata = RunMetadata(
                run_date=run_date,
                window_days=analysis.window_days,
                accounts_raw=accounts_raw,
                accounts_eligible=network.accounts_eligible,
                urls_eligible=network.urls_eligible,
                graph_edges=network.graph_edges,
                edge_quantile=result.edge_quantile,
                centrality_quantile=result.centrality_quantile,
                min_component_density=result.min_component_density,
                knee_found=result.knee_found,
                guardrail_triggered=result.guardrail_triggered,
                flagged_accounts=result.core.vcount(),
                cluster_count=len(cluster_rows),
                excluded_domains_count=len(exclusions.excluded_domains),
                excluded_dids_count=len(exclusions.excluded_dids),
                exclusions_hash=exclusions.content_hash,
                excluded_shares_suppressed=excluded_shares_suppressed,
            )
            with stage_span(telemetry, 'url_cosharing.persist_run_metadata'):
                db.insert_run(analysis.runs_table, run_metadata)
            set_run_attributes(
                root_span,
                {
                    'knee_found': run_metadata.knee_found,
                    'guardrail_triggered': run_metadata.guardrail_triggered,
                    'flagged_accounts': run_metadata.flagged_accounts,
                    'cluster_count': run_metadata.cluster_count,
                },
            )
            record_run_metrics(telemetry, run_metadata, time.monotonic() - run_start)
            logger.info(f'wrote run metadata to {analysis.runs_table}')
        except Exception as exc:
            record_failure(telemetry, 'run_cycle', exc)
            root_span.set_attribute('error.type', type(exc).__name__)
            raise


def _load_startup_exclusions(exclusions_file: str | None) -> Exclusions:
    """Load exclusions once at startup, failing fast on a bad file.

    The env var being set but the file missing/invalid is a deployment error:
    exit non-zero rather than silently running with (or without) exclusions
    the operator believes are applied.
    """
    if exclusions_file is None:
        return Exclusions.empty()
    try:
        return load_exclusions(exclusions_file)
    except (OSError, ValueError) as exc:
        logger.error(f'failed to load exclusions from {exclusions_file}: {exc}')
        raise SystemExit(1) from exc


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    config = AppConfig.from_env()
    logger.info(f'starting url-cosharing detector (interval={config.analysis.interval_seconds}s)')
    logger.info(
        f'window_days={config.analysis.window_days}, min_unique_urls={config.analysis.min_unique_urls}, '
        f'min_url_sharers={config.analysis.min_url_sharers}, density_floor={config.analysis.density_floor}, '
        f'max_flagged_fraction={config.analysis.max_flagged_fraction}, '
        f'max_flagged_accounts={config.analysis.max_flagged_accounts}, resolution={config.analysis.resolution}, '
        f'min_cluster_size={config.analysis.min_cluster_size}, jaccard_threshold={config.analysis.jaccard_threshold}'
    )
    previous_exclusions = _load_startup_exclusions(config.analysis.exclusions_file)
    if previous_exclusions.is_empty():
        logger.info('exclusions: none configured')
    else:
        logger.info(
            f'exclusions loaded: {len(previous_exclusions.excluded_domains)} domains, '
            f'{len(previous_exclusions.excluded_dids)} dids (hash={previous_exclusions.content_hash[:12]})'
        )

    telemetry = setup_telemetry(config.telemetry)
    db = CosharingDb(config.clickhouse)

    try:
        while not _shutdown:
            try:
                # Per-cycle hot reload: edits to the exclusions file take
                # effect next cycle. A mid-flight broken edit retains
                # last-known-good (auditable via the recorded hash); the
                # error is logged and the fallback only covers file
                # breakage, not analysis failures.
                exclusions, reload_error = load_exclusions_with_fallback(
                    config.analysis.exclusions_file, previous_exclusions
                )
                if reload_error is not None:
                    logger.error(reload_error)
                else:
                    previous_exclusions = exclusions
                run_cycle(db, config, telemetry=telemetry, exclusions=exclusions)
            except Exception:
                logger.exception('error during analysis cycle')

            if not _shutdown:
                logger.info(f'sleeping {config.analysis.interval_seconds}s until next cycle')
                time.sleep(config.analysis.interval_seconds)
    finally:
        db.close()
        telemetry.shutdown()
        logger.info('shutdown complete')


if __name__ == '__main__':
    main()
