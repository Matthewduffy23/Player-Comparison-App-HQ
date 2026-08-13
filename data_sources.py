"""Shared data loading for Player-Comparison-App-HQ.

Reads the SAME season-split files every other Streamlit app in the ecosystem
reads — player files named `<season>WORLDFULL.csv` and team files named
`WORLDTEAMS<season>.csv` — rather than carrying its own bundled dataset.

Ordering is by the season parsed from the FILENAME, never st_mtime. The split
scripts write seasons newest-first and shutil.copy2 preserves those source
timestamps, so mtime order is the exact INVERSE of season order and every app
that sorted by it defaulted to the oldest season available.

Team loading is included here now, ahead of the Teams comparison UI, so that
stage has its data layer ready and both sides go through one implementation.
"""

import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

PLAYER_GLOB = "*WORLDFULL.csv"
TEAM_GLOB = "WORLDTEAMS*.csv"

# The unique key for a player-season row. Player+Season is NOT enough: a player
# who transfers mid-season has one row per club, and continental competitions
# add further rows for the SAME club and season. R. Durosinmi has four rows in
# 2025-26 alone (Plzen/Czech 1., Pisa/Italy 1., Europa League, CL Qualifiers),
# all sharing Wyscout ID -299334 — so neither the name nor the ID identifies a
# row on its own.
ROW_KEY = ["Player", "Team", "League", "Season"]


def season_key(name):
    """First 4-digit year anywhere in the filename, for ordering.

    Handles both naming patterns:
        2026-27WORLDFULL.csv   -> 2026   (player files: year leads)
        WORLDTEAMS2026-27.csv  -> 2026   (team files: year follows a prefix)
    Anything without a 4-digit year returns -1 and sorts last, so a stray file
    can never take the default slot.
    """
    m = re.search(r"(\d{4})", str(name))
    return int(m.group(1)) if m else -1


def season_label(name):
    """Human season label from a filename: '2026-27WORLDFULL.csv' -> '2026-27'.

    Falls back to the bare year, then to the filename stem, so this never
    raises on an unexpected name.
    """
    s = str(name)
    m = re.search(r"(\d{4}-\d{2})", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4})", s)
    return m.group(1) if m else Path(s).stem


def sort_by_season(paths, newest_first=True):
    """Order paths by parsed season. Accepts Paths or strings."""
    return sorted(
        paths,
        key=lambda p: season_key(getattr(p, "name", str(p))),
        reverse=newest_first,
    )


def default_search_paths():
    """Where to look for season files.

    Both the working directory (how Streamlit Cloud runs the app) and this
    file's own directory (how it behaves when launched from elsewhere), deduped
    by resolved path so a file is never discovered twice.
    """
    here = Path(__file__).resolve().parent
    out, seen = [], set()
    for d in (Path.cwd(), here):
        try:
            r = d.resolve()
        except OSError:
            continue
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def discover(pattern, search_paths=None):
    """Season files matching `pattern`, newest season first, deduped."""
    paths, seen = [], set()
    for d in (search_paths or default_search_paths()):
        for p in Path(d).glob(pattern):
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                paths.append(p)
    return sort_by_season(paths)


def discover_player_files(search_paths=None):
    return discover(PLAYER_GLOB, search_paths)


def discover_team_files(search_paths=None):
    """Team files. The player glob (*WORLDFULL.csv) and the team glob
    (WORLDTEAMS*.csv) cannot collide, so no cross-filtering is needed."""
    return discover(TEAM_GLOB, search_paths)


@st.cache_data(show_spinner=False)
def _read_one(path_str, mtime, size):
    """Read a single season file.

    mtime/size are part of the cache key ONLY so that a file replaced on disk
    (a refresh cycle rewriting the season) invalidates the entry — they are
    deliberately not used for ordering anywhere.
    """
    return pd.read_csv(path_str)


def load_files(paths):
    """Concatenate season files into one frame.

    Guarantees a 'Season' column: the files carry one, but if a hand-made
    upload does not, the season parsed from the filename is used so downstream
    keying never sees NaN. '__source_file' is kept for provenance/debugging.
    """
    frames = []
    for p in paths:
        p = Path(p)
        try:
            stat = p.stat()
            df = _read_one(str(p), stat.st_mtime, stat.st_size)
        except Exception as e:  # noqa: BLE001 - surfaced in the UI, not swallowed
            st.warning(f"Could not read {p.name}: {e}")
            continue
        df = df.copy()
        if "Season" not in df.columns or df["Season"].isna().all():
            df["Season"] = season_label(p.name)
        df["__source_file"] = p.name
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def read_uploads(uploaded):
    """Read one or more Streamlit UploadedFile objects into a single frame.

    Session-only by construction: nothing here writes to disk, so an upload can
    never overwrite or corrupt the real season files.
    """
    if not uploaded:
        return None
    if not isinstance(uploaded, (list, tuple)):
        uploaded = [uploaded]
    frames = []
    for f in uploaded:
        try:
            df = pd.read_csv(io.BytesIO(f.getvalue()))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not read {f.name}: {e}")
            continue
        if "Season" not in df.columns or df["Season"].isna().all():
            df["Season"] = season_label(f.name)
        df["__source_file"] = f"upload:{f.name}"
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def season_source_picker(kind, key_prefix, default_count=2, search_paths=None):
    """Season multiselect + always-visible upload widget.

    Returns (DataFrame or None, list-of-labels describing what was loaded).

    The uploader sits ALONGSIDE the selector rather than behind an
    "if no files found" branch — when files exist on disk the fallback form is
    unreachable, which is exactly how ad-hoc CSVs became impossible to load in
    the other apps. Uploads are additive: they extend the on-disk seasons
    rather than replacing them, so an ad-hoc file can be compared against real
    data in the same session.

    Only the SELECTED seasons are read. Each season is roughly 57 MB in memory,
    so defaulting to every file on disk would be ~500 MB before any work
    happens.
    """
    finder = discover_player_files if kind == "players" else discover_team_files
    files = finder(search_paths)
    noun = "player" if kind == "players" else "team"

    chosen_paths = []
    if files:
        labels = [f"{season_label(p.name)}  ({p.name})" for p in files]
        by_label = dict(zip(labels, files))
        picked = st.multiselect(
            f"{noun.capitalize()} seasons to load",
            labels,
            default=labels[:default_count],
            key=f"{key_prefix}_{kind}_seasons",
            help="Ordered newest season first, parsed from the filename.",
        )
        chosen_paths = [by_label[l] for l in picked]
    else:
        st.info(
            f"No {noun} season files found (looking for "
            f"`{PLAYER_GLOB if kind == 'players' else TEAM_GLOB}`). Upload one below."
        )

    uploaded = st.file_uploader(
        f"Or upload a {noun} CSV",
        type=["csv"],
        accept_multiple_files=True,
        key=f"{key_prefix}_{kind}_upload",
        help="Session only — never written to disk, never touches the real season files.",
    )

    parts, described = [], []
    disk_df = load_files(chosen_paths) if chosen_paths else None
    if disk_df is not None:
        parts.append(disk_df)
        described += [season_label(p.name) for p in chosen_paths]
    up_df = read_uploads(uploaded)
    if up_df is not None:
        parts.append(up_df)
        described += [f"upload:{f.name}" for f in uploaded]

    if not parts:
        return None, []
    if len(parts) == 1:
        return parts[0], described
    return pd.concat(parts, ignore_index=True, sort=False), described
