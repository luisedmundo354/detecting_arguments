"""Thin wrapper retained for backwards compatibility with previous entrypoint."""

from iaa_scores.cli import main


if __name__ == "__main__":  # pragma: no cover
    main()
