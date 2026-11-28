from pathlib import Path

from faction_analysis import main


def test_the_api_key_comes_from_the_environment_first(tmp_path: Path) -> None:
    secret = tmp_path / 'typesafe_api_key'
    secret.write_text(' from-file \n')
    assert main.api_key({'TYPESAFE_API_KEY': ' from-env '}, secret) == 'from-env'
    assert main.api_key({}, secret) == 'from-file'
    # No environment variable and no secret file: no key, and the model refresh buys no labels.
    assert main.api_key({}, tmp_path / 'missing') == ''
    assert main.api_key({'TYPESAFE_API_KEY': '   '}, tmp_path / 'missing') == ''


def test_the_command_line_flags() -> None:
    parser = main.build_parser()
    args = parser.parse_args(['--once', '--kind', 'full', '--dry-run'])
    assert (args.once, args.kind, args.dry_run, args.measure) == (True, 'full', True, False)
    args = parser.parse_args([])
    assert (args.once, args.kind, args.dry_run, args.measure) == (False, None, False, False)
