
#!/usr/bin/env python3
"""
Elo Talk Ranker (Tkinter)

Purpose
-------
Interactive, bias-aware Elo-style ranking for conference talks based on repeated
side-by-side comparisons of 2, 3, or 4 talks at a time.

Key bias-mitigation features implemented
---------------------------------------
- Balanced exposure: prioritizes talks with fewer appearances.
- Novelty: discourages repeatedly comparing the same pairs.
- Connectivity: mild preference for mixing tracks (optional metadata).
- Position bias reduction: randomizes on-screen ordering each round.
- Stronger signal per screen: captures a *full ordering* (1..m) by default.
- Adaptive K: larger updates early, smaller as evidence accumulates.
- Undo: correct accidental submissions without contaminating the ladder.
- Audit log + autosave: reproducible and resumable sessions.

Input CSV columns (required)
----------------------------
ID, Title, Speaker, Abstract, Track

Usage
-----
python elo_talk_ranker.py
# or
python elo_talk_ranker.py --csv /path/to/talks.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
from tkinter.scrolledtext import ScrolledText


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Talk:
    talk_id: str
    title: str
    speaker: str
    abstract: str
    track: str = ""


# -----------------------------
# Elo engine (multi-opponent)
# -----------------------------

class EloEngine:
    """
    Multi-opponent Elo update for m-way rankings.

    We treat a ranking as implicit pairwise outcomes, compute each item's
    actual score s_i as the average of its pairwise results against the other
    displayed items, compute expected e_i as the average expected win prob
    against the others, then update once per item:

        R_i <- R_i + K_i * (s_i - e_i)

    K_i is adaptive (shrinks with accumulated evidence), and is mildly scaled
    with sqrt(m-1) so 4-way screens (more information) can move slightly more
    than 2-way screens.
    """

    def __init__(
        self,
        talk_ids: List[str],
        base_rating: float = 1500.0,
        scale: float = 400.0,
        base_k: float = 40.0,
        k_decay_pairs: float = 30.0,
        k_min: float = 8.0,
    ) -> None:
        self.base_rating = float(base_rating)
        self.scale = float(scale)
        self.base_k = float(base_k)
        self.k_decay_pairs = float(k_decay_pairs)
        self.k_min = float(k_min)

        self.ratings: Dict[str, float] = {tid: self.base_rating for tid in talk_ids}

        # Rounds = comparison screens participated in
        self.rounds_seen: Dict[str, int] = {tid: 0 for tid in talk_ids}
        # Pairwise-equivalent comparisons: sum of (m-1) per screen per talk
        self.pairwise_seen: Dict[str, int] = {tid: 0 for tid in talk_ids}

        # Pair co-appearance counts to discourage repeatedly matching the same pairs
        self.pair_counts: Dict[Tuple[str, str], int] = {}

        # How many submitted comparison screens total
        self.rounds_done: int = 0

        # History for undo + audit trail
        self.history: List[Dict[str, Any]] = []

        # Abstained talks (skip in comparisons, empty in ranking)
        self.abstained_ids: Set[str] = set()

    def expected_win_prob(self, r_a: float, r_b: float) -> float:
        # Standard Elo logistic with base-10
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / self.scale))

    def _k_for(self, talk_id: str, m: int) -> float:
        pairs = self.pairwise_seen.get(talk_id, 0)
        k = self.base_k / (1.0 + (pairs / max(1e-9, self.k_decay_pairs)))
        k = max(self.k_min, k)
        # mild information scaling by number of opponents
        k *= math.sqrt(max(1, m - 1))
        return k

    def compute_expected_scores(self, ids: List[str]) -> Dict[str, float]:
        m = len(ids)
        expected: Dict[str, float] = {tid: 0.0 for tid in ids}
        for i, a in enumerate(ids):
            r_a = self.ratings[a]
            acc = 0.0
            for j, b in enumerate(ids):
                if i == j:
                    continue
                acc += self.expected_win_prob(r_a, self.ratings[b])
            expected[a] = acc / (m - 1)
        return expected

    @staticmethod
    def _pair_key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def update_from_ranks(
        self,
        ids: List[str],
        ranks: Dict[str, int],
        *,
        allow_ties: bool = False,
    ) -> None:
        """
        ids: talk IDs shown in this round (order does not matter).
        ranks: mapping talk_id -> integer rank (1 = best). If allow_ties=True,
               equal ranks produce ties.
        """
        m = len(ids)
        if m < 2:
            raise ValueError("Need at least 2 talks for a comparison.")
        for tid in ids:
            if tid not in ranks:
                raise ValueError(f"Missing rank for talk_id={tid}")

        # Validate ranks
        values = [int(ranks[tid]) for tid in ids]
        if not allow_ties:
            if sorted(values) != list(range(1, m + 1)):
                raise ValueError(f"Ranks must be a permutation of 1..{m}.")
        else:
            # at least one rank value must exist; gaps are fine
            if any(v < 1 or v > m for v in values):
                raise ValueError(f"Ranks must be within 1..{m} when ties are enabled.")

        # Snapshot for undo/audit
        before_ratings = {tid: self.ratings[tid] for tid in ids}
        before_rounds = {tid: self.rounds_seen[tid] for tid in ids}
        before_pairs = {tid: self.pairwise_seen[tid] for tid in ids}

        # Actual scores s_i from implicit pairwise outcomes
        actual: Dict[str, float] = {tid: 0.0 for tid in ids}
        for i, a in enumerate(ids):
            wins = 0.0
            for j, b in enumerate(ids):
                if i == j:
                    continue
                ra = ranks[a]
                rb = ranks[b]
                if ra < rb:
                    wins += 1.0
                elif ra == rb:
                    wins += 0.5
                else:
                    wins += 0.0
            actual[a] = wins / (m - 1)

        expected = self.compute_expected_scores(ids)

        # Apply update
        for tid in ids:
            k = self._k_for(tid, m)
            self.ratings[tid] += k * (actual[tid] - expected[tid])

        # Update exposure counts
        for tid in ids:
            self.rounds_seen[tid] += 1
            self.pairwise_seen[tid] += (m - 1)

        # Update pair co-appearance counts
        for i in range(m):
            for j in range(i + 1, m):
                a, b = ids[i], ids[j]
                key = self._pair_key(a, b)
                self.pair_counts[key] = self.pair_counts.get(key, 0) + 1

        self.rounds_done += 1

        record = {
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "ids": list(ids),
            "ranks": {tid: int(ranks[tid]) for tid in ids},
            "before_ratings": before_ratings,
            "after_ratings": {tid: self.ratings[tid] for tid in ids},
            "before_rounds_seen": before_rounds,
            "after_rounds_seen": {tid: self.rounds_seen[tid] for tid in ids},
            "before_pairwise_seen": before_pairs,
            "after_pairwise_seen": {tid: self.pairwise_seen[tid] for tid in ids},
        }
        self.history.append(record)

    def undo_last(self) -> bool:
        """Undo the most recent submitted comparison screen."""
        if not self.history:
            return False
        record = self.history.pop()
        ids = record["ids"]

        # Restore ratings and counts
        for tid, val in record["before_ratings"].items():
            self.ratings[tid] = float(val)
        for tid, val in record["before_rounds_seen"].items():
            self.rounds_seen[tid] = int(val)
        for tid, val in record["before_pairwise_seen"].items():
            self.pairwise_seen[tid] = int(val)

        # Roll back pair_counts (subtract one for every pair in ids)
        m = len(ids)
        for i in range(m):
            for j in range(i + 1, m):
                key = self._pair_key(ids[i], ids[j])
                if key in self.pair_counts:
                    self.pair_counts[key] -= 1
                    if self.pair_counts[key] <= 0:
                        del self.pair_counts[key]

        self.rounds_done = max(0, self.rounds_done - 1)
        return True

    def ranked_ids(self) -> List[str]:
        return sorted(self.ratings.keys(), key=lambda tid: self.ratings[tid], reverse=True)


# -----------------------------
# Tooltips & Hints
# -----------------------------

class ToolTip:
    """
    Shows a small tooltip window near the mouse cursor after a delay,
    and updates a status/hint line in the main application immediately.
    """

    def __init__(self, widget: tk.Widget, text: str, app: 'TalkRankerApp') -> None:
        self.widget = widget
        self.text = text
        self.app = app
        self.tip_window: Optional[tk.Toplevel] = None
        self.id: Optional[str] = None
        self.x = self.y = 0

        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)

    def enter(self, event: Optional[tk.Event] = None) -> None:
        self.schedule()
        self.app.set_hint(self.text)

    def leave(self, event: Optional[tk.Event] = None) -> None:
        self.unschedule()
        self.hidetip()
        self.app.clear_hint()

    def schedule(self) -> None:
        self.unschedule()
        self.id = self.widget.after(600, self.showtip)

    def unschedule(self) -> None:
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, event: Optional[tk.Event] = None) -> None:
        x, y, cx, cy = self.widget.bbox("insert")  # type: ignore
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", "8", "normal"),
        )
        label.pack(ipadx=1)

    def hidetip(self) -> None:
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


class ComparisonScheduler:
    """
    Selects the next set of m talks to compare with an emphasis on:
    - balanced exposure (low rounds_seen)
    - novelty (avoid repeating pairs)
    - mild cross-track mixing to improve graph connectivity
    - occasional exploration to prevent local pockets
    """

    def __init__(
        self,
        talks: Dict[str, Talk],
        engine: EloEngine,
        *,
        explore_rate: float = 0.35,
        pool_delta_rounds: int = 1,
        rating_closeness_scale: float = 200.0,
        track_same_penalty: float = 0.95,
        track_mix_bonus: float = 1.05,
        avoid_repeat_last: bool = True,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.talks = talks
        self.engine = engine
        self.explore_rate = float(explore_rate)
        self.pool_delta_rounds = int(pool_delta_rounds)
        self.rating_closeness_scale = float(rating_closeness_scale)
        self.track_same_penalty = float(track_same_penalty)
        self.track_mix_bonus = float(track_mix_bonus)
        self.avoid_repeat_last = bool(avoid_repeat_last)
        self.rng = rng or random.Random()

        self._last_set_frozenset: Optional[frozenset[str]] = None

    def _avg_pair_count(self, tid: str, selected: List[str]) -> float:
        if not selected:
            return 0.0
        acc = 0.0
        for other in selected:
            key = (tid, other) if tid < other else (other, tid)
            acc += self.engine.pair_counts.get(key, 0)
        return acc / len(selected)

    def choose_next_set(self, m: int) -> List[str]:
        all_ids = list(self.talks.keys())

        # Filter out abstained
        active_ids = [tid for tid in all_ids if tid not in self.engine.abstained_ids]

        if m > len(active_ids):
            raise ValueError("Not enough active talks to display that many at once.")

        explore_mode = (self.rng.random() < self.explore_rate)

        rounds_seen = self.engine.rounds_seen
        pairwise_seen = self.engine.pairwise_seen
        ratings = self.engine.ratings

        min_rounds = min(rounds_seen.values()) if rounds_seen else 0
        max_rounds = max(rounds_seen.values()) if rounds_seen else 0

        # Pool: talks with lowest exposure (within delta)
        pool = [tid for tid in active_ids if rounds_seen[tid] <= min_rounds + self.pool_delta_rounds]
        if len(pool) < m:
            # widen pool deterministically
            pool = sorted(active_ids, key=lambda tid: rounds_seen[tid])[:max(m, min(50, len(active_ids)))]

        # Pick a seed from low-exposure pool, with some randomness for coverage
        seed = self.rng.choice(pool)

        selected: List[str] = [seed]
        seed_rating = ratings[seed]

        # Select remaining talks with weighted sampling
        while len(selected) < m:
            candidates = [tid for tid in active_ids if tid not in selected]
            weights: List[float] = []
            for tid in candidates:
                # Exposure: prefer those with fewer rounds
                exposure_term = 1.0 + (max_rounds - rounds_seen[tid])

                # Novelty: prefer low repeated co-appearances with selected
                avg_pair = self._avg_pair_count(tid, selected)
                novelty_term = 1.0 / (1.0 + avg_pair)  # in (0,1]

                # Uncertainty: prefer talks with fewer pairwise comparisons
                u = 1.0 / math.sqrt(1.0 + pairwise_seen[tid])
                uncertainty_term = 1.0 + u  # in (1,2]

                # Rating closeness (only in exploit mode)
                rating_term = 1.0
                if not explore_mode:
                    diff = abs(ratings[tid] - seed_rating)
                    rating_term = 1.0 / (1.0 + (diff / max(1e-9, self.rating_closeness_scale)))

                # Mild track mixing preference
                track_term = 1.0
                t_track = (self.talks[tid].track or "").strip()
                if t_track:
                    same_track = any(((self.talks[x].track or "").strip() == t_track) for x in selected)
                    track_term = self.track_same_penalty if same_track else self.track_mix_bonus

                w = exposure_term * novelty_term * uncertainty_term * rating_term * track_term
                weights.append(max(1e-6, w))

            pick = self.rng.choices(candidates, weights=weights, k=1)[0]
            selected.append(pick)

        # Randomize display order to reduce position bias
        self.rng.shuffle(selected)

        if self.avoid_repeat_last:
            current_set = frozenset(selected)
            if self._last_set_frozenset is not None and current_set == self._last_set_frozenset:
                # re-sample a few times to avoid immediate repeats
                for _ in range(50):
                    self.rng.shuffle(selected)
                    if frozenset(selected) != self._last_set_frozenset:
                        break
            self._last_set_frozenset = frozenset(selected)

        return selected


# -----------------------------
# Persistence
# -----------------------------

def state_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".elo_state.json")


def save_state(csv_path: Path, talks: Dict[str, Talk], engine: EloEngine, app_cfg: Dict[str, Any]) -> None:
    path = state_path_for(csv_path)
    pair_counts_ser = {f"{a}||{b}": c for (a, b), c in engine.pair_counts.items()}

    data = {
        "version": 1,
        "csv_file": str(csv_path),
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "engine": {
            "base_rating": engine.base_rating,
            "scale": engine.scale,
            "base_k": engine.base_k,
            "k_decay_pairs": engine.k_decay_pairs,
            "k_min": engine.k_min,
            "ratings": engine.ratings,
            "rounds_seen": engine.rounds_seen,
            "pairwise_seen": engine.pairwise_seen,
            "pair_counts": pair_counts_ser,
            "rounds_done": engine.rounds_done,
            "history": engine.history[-5000:],  # cap size defensively
            "abstained_ids": list(engine.abstained_ids),
        },
        "app_cfg": app_cfg,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_state(csv_path: Path) -> Optional[Dict[str, Any]]:
    path = state_path_for(csv_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# -----------------------------
# CSV I/O
# -----------------------------

def load_talks_csv(csv_path: Path) -> Dict[str, Talk]:
    required = ["ID", "Title", "Speaker", "Abstract", "Track"]
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing = [c for c in required if c not in headers]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {headers}")

        talks: Dict[str, Talk] = {}
        for row in reader:
            tid = (row.get("ID") or "").strip()
            if not tid:
                continue
            if tid in talks:
                raise ValueError(f"Duplicate ID in CSV: {tid}")

            talks[tid] = Talk(
                talk_id=tid,
                title=(row.get("Title") or "").strip(),
                speaker=(row.get("Speaker") or "").strip(),
                abstract=(row.get("Abstract") or "").strip(),
                track=(row.get("Track") or "").strip(),
            )
    if not talks:
        raise ValueError("No talks loaded from CSV.")
    return talks


def export_rankings_csv(
    out_path: Path,
    talks: Dict[str, Talk],
    engine: EloEngine,
    scale_min: float = 1.0,
    scale_max: float = 10.0,
    round_ranking: bool = False,
    columns: Optional[List[str]] = None,
    sort_by: str = "rank",  # "rank" or "id"
) -> None:
    if sort_by == "id":
        # Try numeric sort if IDs look like integers
        try:
             ranked = sorted(engine.ratings.keys(), key=lambda x: int(x))
        except ValueError:
             ranked = sorted(engine.ratings.keys())
    else:
        # Default to rank: Sort by rating (descending).
        # Exclude abstained from the ranking list order basically?
        # Or put abstained at bottom?
        # Let's get "valid" ranked IDs first
        valid = [tid for tid in engine.ranked_ids() if tid not in engine.abstained_ids]

        # Abstained at bottom, sorted logic? or just appended
        abstained = sorted([tid for tid in engine.abstained_ids if tid in engine.ratings], key=lambda x: engine.ratings[x], reverse=True)
        # Note: abstained talks still have ratings technically, but we treat them as unranked output.

        ranked = valid + abstained

    # Calculate min/max elo for scaling (always needed for score calc even if score not exported?)
    # We calculate it based on ALL ranked items to stay consistent
    all_ratings = [engine.ratings[tid] for tid in engine.ratings]
    min_elo = min(all_ratings) if all_ratings else 0
    max_elo = max(all_ratings) if all_ratings else 1

    # Define available column getters
    # Place is rank index (1..N) if sorted by rank, else it might be confusing?
    # Usually "Place" implies ranking order. If sorted by ID, "Place" is just row number?
    # Let's keep "Place" as the position in the current list being exported.

    # Headers map
    # We will just use the list of strings provided in `columns`.
    # But we need to know how to map a column name to value.

    col_map = {
        "Place": lambda i, tid, t, r, s: i if tid not in engine.abstained_ids else "",
        "Elo": lambda i, tid, t, r, s: (f"{r:.0f}" if round_ranking else f"{r:.2f}") if tid not in engine.abstained_ids else "",
        "Score": lambda i, tid, t, r, s: (f"{s:.0f}" if round_ranking else f"{s:.2f}") if tid not in engine.abstained_ids else "",
        "ID": lambda i, tid, t, r, s: tid,
        "Speaker": lambda i, tid, t, r, s: t.speaker,
        "Title": lambda i, tid, t, r, s: t.title,
        "Track": lambda i, tid, t, r, s: t.track,
    }

    # Default columns if None
    if columns is None:
        columns = ["ID", "Place", "Elo", "Score", "Speaker", "Title"]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for i, tid in enumerate(ranked, start=1):
            t = talks[tid]
            rating = engine.ratings[tid]

            # Recalculate 'i' if abstained?
            # Ideally "Place" should skip abstained?
            # If sorted by rank, active talks get 1..N. Abstained get nothing.
            # If sorted by ID, Place should probably still reflect their rank order if they were valid?
            # But requirement says: "show them with empty ranks".
            # So if abstained, Place is "".

            # To get correct "Place" for valid talks even if mixed or sorted by ID:
            # We can pre-calculate the rank of every valid ID.
            pass

    # Pre-calculate ranks for valid items
    valid_ranked = [tid for tid in engine.ranked_ids() if tid not in engine.abstained_ids]
    rank_map = {tid: i for i, tid in enumerate(valid_ranked, start=1)}

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for tid in ranked:
            t = talks[tid]
            rating = engine.ratings[tid]

            # Place
            place = rank_map.get(tid, "")

            # Calc score
            if max_elo > min_elo:
                 norm = (rating - min_elo) / (max_elo - min_elo)
                 score = scale_min + norm * (scale_max - scale_min)
            else:
                 score = (scale_min + scale_max) / 2.0

            row = []
            for col in columns:
                if col in col_map:
                    val = col_map[col](place, tid, t, rating, score)
                    row.append(val)
                else:
                    row.append("")

            writer.writerow(row)


# -----------------------------
# UI
# -----------------------------

class TalkRankerApp(tk.Tk):
    def __init__(self, csv_path: Optional[Path] = None) -> None:
        super().__init__()
        self.title("Elo Talk Ranker")
        self.geometry("1500x1000")

        self.csv_path: Optional[Path] = None
        self.talks: Dict[str, Talk] = {}
        self.engine: Optional[EloEngine] = None
        self.scheduler: Optional[ComparisonScheduler] = None

        # Session-level configuration (can be changed during session)
        self.compare_size = tk.IntVar(value=3)
        self.target_appearances_per_talk = tk.IntVar(value=10)
        self.explore_rate = tk.DoubleVar(value=0.5)
        self.allow_ties = tk.BooleanVar(value=False)

        # Display controls (bias-related: can anonymize)
        self.show_speaker = tk.BooleanVar(value=True)
        self.show_abstract = tk.BooleanVar(value=True)
        self.round_ranking = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="")
        self.hint_var = tk.StringVar(value="")

        # Export Configuration
        self.export_cols: Dict[str, tk.BooleanVar] = {
            "Place": tk.BooleanVar(value=True),
            "Elo": tk.BooleanVar(value=True),
            "Score": tk.BooleanVar(value=True),
            "ID": tk.BooleanVar(value=True),
            "Speaker": tk.BooleanVar(value=True),
            "Title": tk.BooleanVar(value=True),
            "Track": tk.BooleanVar(value=True),
        }
        self.export_sort = tk.StringVar(value="rank") # "rank" or "id"

        # Font customization
        # self.font_title_family = tk.StringVar(value="Helvetica")
        self.font_title_family = tk.StringVar(value="Arimo")
        self.font_title_size = tk.IntVar(value=12)
        self.font_title_spacing = tk.IntVar(value=2)  # Extra line spacing for title
        # self.font_speaker_family = tk.StringVar(value="Helvetica")
        self.font_speaker_family = tk.StringVar(value="Arimo")
        self.font_speaker_size = tk.IntVar(value=11)
        # self.font_abstract_family = tk.StringVar(value="Times")
        self.font_abstract_family = tk.StringVar(value="Arimo")
        self.font_abstract_size = tk.IntVar(value=12)
        self.font_abstract_spacing = tk.IntVar(value=2)  # Extra line spacing

        # Current set
        self.current_ids: List[str] = []
        self.rank_vars: List[tk.StringVar] = []
        self.rank_combos: List[ttk.Combobox] = []

        # Grid widgets
        self.title_lbls: List[tk.Label] = []
        self.speaker_lbls: List[tk.Label] = []
        self.abs_boxes: List[ScrolledText] = []

        # Build UI

        self.scale_min = tk.IntVar(value=1)
        self.scale_max = tk.IntVar(value=9)
        self.show_current_rank = tk.BooleanVar(value=False)

        self.ranking_window_geometry: Optional[str] = None
        self.abstain_window_geometry: Optional[str] = None
        self.ranking_win: Optional[tk.Toplevel] = None

        # Build UI
        self._build_menu()
        self._build_controls()
        self._build_comparison_area()
        self._build_status_bar()

        # Load initial CSV if provided
        if csv_path is not None:
            self.load_csv(csv_path)
        else:
            # Prompt user to open a CSV
            self.after(100, self._prompt_open_csv)

        # Autosave on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Keyboard bindings (global)
        # Main number keys
        self.bind_all("<Key-1>", lambda e: self._on_rank_shortcut(0))
        self.bind_all("<Key-2>", lambda e: self._on_rank_shortcut(1))
        self.bind_all("<Key-3>", lambda e: self._on_rank_shortcut(2))
        self.bind_all("<Key-4>", lambda e: self._on_rank_shortcut(3))
        # Numpad keys
        self.bind_all("<KP_1>", lambda e: self._on_rank_shortcut(0))
        self.bind_all("<KP_2>", lambda e: self._on_rank_shortcut(1))
        self.bind_all("<KP_3>", lambda e: self._on_rank_shortcut(2))
        self.bind_all("<KP_4>", lambda e: self._on_rank_shortcut(3))

        self.bind_all("<Shift-Return>", self.on_submit)
        self.bind_all("<Escape>", self._on_clear_ranks)

    # ---- Tooltip / Hint Support ----

    def set_hint(self, text: str) -> None:
        self.hint_var.set(text)

    def clear_hint(self) -> None:
        self.hint_var.set("")

    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        ToolTip(widget, text, self)

    # ---- UI scaffolding ----

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Talks CSV…", command=self._prompt_open_csv)
        file_menu.add_command(label="Export Ranking CSV…", command=self.on_export)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Show Current Ranking…", command=self.on_show_rankings)
        view_menu.add_separator()
        view_menu.add_command(label="Font Settings…", command=self.on_font_settings)
        menubar.add_cascade(label="View", menu=view_menu)

        action_menu = tk.Menu(menubar, tearoff=0)
        action_menu.add_command(label="Undo Last Comparison", command=self.on_undo)
        action_menu.add_command(label="Skip / New Set", command=self.on_skip)
        action_menu.add_separator()
        action_menu.add_command(label="Manage Abstentions...", command=self.on_manage_abstentions)
        menubar.add_cascade(label="Actions", menu=action_menu)

        self.config(menu=menubar)

    def _build_controls(self) -> None:
        frm = ttk.Frame(self, padding=10)
        frm.pack(side=tk.TOP, fill=tk.X)

        # Row 0: Settings + Slider + Buttons

        # Comparison size
        ttk.Label(frm, text="Talks per comparison:").grid(row=0, column=0, sticky="w")
        size_cb = ttk.Combobox(frm, width=5, state="readonly", values=[2, 3, 4], textvariable=self.compare_size)
        size_cb.grid(row=0, column=1, sticky="w", padx=(5, 15))
        size_cb.bind("<<ComboboxSelected>>", lambda e: self._rebuild_comparison_panels())
        self._add_tooltip(size_cb, "Number of talks to compare at once (2-4).")

        # Target appearances per talk
        ttk.Label(frm, text="Target appearances per talk:").grid(row=0, column=2, sticky="w")
        tgt = ttk.Spinbox(frm, from_=1, to=999, width=6, textvariable=self.target_appearances_per_talk, command=self._update_status)
        tgt.grid(row=0, column=3, sticky="w", padx=(5, 15))
        self._add_tooltip(tgt, "Target number of times each talk should be shown to you.")

        # Explore rate (Compact)
        ttk.Label(frm, text="Exploration rate:").grid(row=0, column=4, sticky="w")
        # Fixed length=150 to make it shorter as requested
        er = ttk.Scale(frm, from_=0.0, to=0.9, variable=self.explore_rate, length=150, command=lambda v: self._on_explore_change())
        er.grid(row=0, column=5, sticky="w", padx=(5, 5))
        self._add_tooltip(er, "Exploration rate: Higher values show more random/less-seen talks vs. close-ranking pairs.")

        self._explore_label = ttk.Label(frm, text=f"{self.explore_rate.get():.2f}")
        self._explore_label.grid(row=0, column=6, sticky="w", padx=(0, 15))

        # Buttons (Moved to Row 0, Right aligned)
        btns = ttk.Frame(frm)
        btns.grid(row=0, column=7, sticky="e")
        # Push buttons to the right
        frm.columnconfigure(7, weight=1)

        b_sub = ttk.Button(btns, text="Submit ranking", command=self.on_submit)
        b_sub.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(b_sub, "Confirm and save the current ranking.")

        b_skip = ttk.Button(btns, text="Skip / New set", command=self.on_skip)
        b_skip.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(b_skip, "Discard current set and show new talks (no rating update).")

        b_undo = ttk.Button(btns, text="Undo", command=self.on_undo)
        b_undo.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(b_undo, "Revert the last submitted comparison.")

        b_show = ttk.Button(btns, text="Show ranking", command=self.on_show_rankings)
        b_show.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(b_show, "Display current leaderboard and statistics.")

        b_abs = ttk.Button(btns, text="Manage Abstentions", command=self.on_manage_abstentions)
        b_abs.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(b_abs, "Manage excluded/abstained talks.")

        b_ex = ttk.Button(btns, text="Export CSV", command=self.on_export)
        b_ex.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(b_ex, "Save current rankings to a CSV file.")

        # Row 1: Checkboxes (Display options) & Scale Settings

        row1 = ttk.Frame(frm)
        row1.grid(row=1, column=0, columnspan=8, sticky="ew", pady=(10, 0))

        # Group 1: Display Options (Left)
        disp_opts = ttk.Frame(row1)
        disp_opts.pack(side=tk.LEFT)

        chk_spk = ttk.Checkbutton(disp_opts, text="Show Speaker", variable=self.show_speaker, command=self._render_current_set)
        chk_spk.pack(side=tk.LEFT, padx=(0, 10))
        self._add_tooltip(chk_spk, "Toggle visibility of speaker names.")

        chk_abs = ttk.Checkbutton(disp_opts, text="Show Abstract", variable=self.show_abstract, command=self._render_current_set)
        chk_abs.pack(side=tk.LEFT, padx=(0, 10))
        self._add_tooltip(chk_abs, "Toggle visibility of talk abstracts.")

        chk_ties = ttk.Checkbutton(disp_opts, text="Allow ties", variable=self.allow_ties)
        chk_ties.pack(side=tk.LEFT, padx=(0, 10))
        self._add_tooltip(chk_ties, "Allow assigning the same rank to multiple talks.")

        chk_rank = ttk.Checkbutton(disp_opts, text="Show Rank", variable=self.show_current_rank, command=self._render_current_set)
        chk_rank.pack(side=tk.LEFT, padx=(0, 10))
        self._add_tooltip(chk_rank, "Show current Elo rating and score below abstract.")

        chk_round = ttk.Checkbutton(disp_opts, text="Round Ranking", variable=self.round_ranking, command=self._render_current_set)
        chk_round.pack(side=tk.LEFT, padx=(0, 10))
        self._add_tooltip(chk_round, "Round Elo and Score to whole numbers (display & export).")

        # Group 2: Scale Settings (Right)
        scale_opts = ttk.Frame(row1)
        scale_opts.pack(side=tk.RIGHT)

        ttk.Label(scale_opts, text="Scale:").pack(side=tk.LEFT, padx=(10, 2))

        sc_min = ttk.Spinbox(scale_opts, from_=0, to=9999, width=5, textvariable=self.scale_min)
        sc_min.pack(side=tk.LEFT, padx=2)
        self._add_tooltip(sc_min, "Minimum value for scaled score (e.g. 1).")

        ttk.Label(scale_opts, text="-").pack(side=tk.LEFT, padx=2)

        sc_max = ttk.Spinbox(scale_opts, from_=0, to=9999, width=5, textvariable=self.scale_max)
        sc_max.pack(side=tk.LEFT, padx=(2, 0))
        self._add_tooltip(sc_max, "Maximum value for scaled score (e.g. 10).")

    def _build_comparison_area(self) -> None:
        self.compare_container = ttk.Frame(self, padding=10)
        self.compare_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Help footer (packed BOTTOM so it stays at the bottom)
        help_frame = ttk.Frame(self.compare_container, padding=(0, 5))
        help_frame.pack(side=tk.BOTTOM, fill=tk.X)
        help_lbl = ttk.Label(
            help_frame,
            text="Shortcuts: Press '1', '2'... to assign rank to talk in the panel of corresponding number in the order of pressing (1. press - talk X is best, 2. press - talk X is second best, etc.) • 'Shift+Enter' to Submit • 'Esc' to Clear",
            foreground="black",
            font=("TkDefaultFont", 10, "italic")
        )
        help_lbl.pack(anchor=tk.CENTER)

        self.panels_frame = ttk.Frame(self.compare_container)
        self.panels_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._rebuild_comparison_panels()

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self, padding=(10, 6))
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Left side: Status
        ttk.Label(bar, textvariable=self.status_var).pack(side=tk.LEFT)

        # Right side: Hints
        ttk.Label(bar, textvariable=self.hint_var, foreground="gray").pack(side=tk.RIGHT)

        self._update_status()

    # ---- CSV/session handling ----

    def _prompt_open_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Talks CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.load_csv(Path(path))

    def load_csv(self, csv_path: Path) -> None:
        try:
            talks = load_talks_csv(csv_path)
        except Exception as e:
            messagebox.showerror("Failed to load CSV", str(e))
            return

        self.csv_path = csv_path
        self.talks = talks

        # Initialize engine and scheduler
        engine = EloEngine(list(talks.keys()))
        self.engine = engine

        self.scheduler = ComparisonScheduler(
            talks=talks,
            engine=engine,
            explore_rate=self.explore_rate.get(),
        )

        # Try to restore state
        state = load_state(csv_path)
        if state and state.get("engine", {}).get("ratings"):
            if messagebox.askyesno("Resume session?", f"Found saved state for:\n{csv_path}\n\nResume previous session?"):
                self._apply_state(state)
            else:
                # If user chooses not to resume, discard state file to prevent confusion
                # (safe/optional; comment out if undesired)
                pass

        self._rebuild_comparison_panels()
        self.on_skip()  # start with first set
        self._update_status()

    def _apply_state(self, state: Dict[str, Any]) -> None:
        if self.engine is None:
            return
        eng = state.get("engine", {})
        try:
            self.engine.base_rating = float(eng.get("base_rating", self.engine.base_rating))
            self.engine.scale = float(eng.get("scale", self.engine.scale))
            self.engine.base_k = float(eng.get("base_k", self.engine.base_k))
            self.engine.k_decay_pairs = float(eng.get("k_decay_pairs", self.engine.k_decay_pairs))
            self.engine.k_min = float(eng.get("k_min", self.engine.k_min))

            # Restore mappings (only for IDs present)
            for tid in self.talks.keys():
                if tid in eng.get("ratings", {}):
                    self.engine.ratings[tid] = float(eng["ratings"][tid])
                if tid in eng.get("rounds_seen", {}):
                    self.engine.rounds_seen[tid] = int(eng["rounds_seen"][tid])
                if tid in eng.get("pairwise_seen", {}):
                    self.engine.pairwise_seen[tid] = int(eng["pairwise_seen"][tid])

            self.engine.pair_counts.clear()
            pc = eng.get("pair_counts", {})
            for k, v in pc.items():
                if "||" in k:
                    a, b = k.split("||", 1)
                    if a in self.talks and b in self.talks:
                        key = (a, b) if a < b else (b, a)
                        self.engine.pair_counts[key] = int(v)

            self.engine.rounds_done = int(eng.get("rounds_done", 0))
            self.engine.history = list(eng.get("history", []))
            self.engine.abstained_ids = set(eng.get("abstained_ids", []))

            # Restore UI config if available
            cfg = state.get("app_cfg", {})
            if "compare_size" in cfg:
                self.compare_size.set(int(cfg["compare_size"]))
            if "target_appearances_per_talk" in cfg:
                self.target_appearances_per_talk.set(int(cfg["target_appearances_per_talk"]))
            if "explore_rate" in cfg:
                self.explore_rate.set(float(cfg["explore_rate"]))
            if "allow_ties" in cfg:
                self.allow_ties.set(bool(cfg["allow_ties"]))
            if "show_speaker" in cfg:
                self.show_speaker.set(bool(cfg["show_speaker"]))
            if "show_abstract" in cfg:
                self.show_abstract.set(bool(cfg["show_abstract"]))
            if "round_ranking" in cfg:
                self.round_ranking.set(bool(cfg["round_ranking"]))

            # Restore export settings
            if "export_sort" in cfg:
                self.export_sort.set(cfg["export_sort"])
            exp_cols = cfg.get("export_cols", {})
            for k, v in exp_cols.items():
                if k in self.export_cols:
                    self.export_cols[k].set(bool(v))

            if "font_title_family" in cfg: self.font_title_family.set(cfg["font_title_family"])
            if "font_title_size" in cfg: self.font_title_size.set(int(cfg["font_title_size"]))
            if "font_title_spacing" in cfg: self.font_title_spacing.set(int(cfg["font_title_spacing"]))
            if "font_speaker_family" in cfg: self.font_speaker_family.set(cfg["font_speaker_family"])
            if "font_speaker_size" in cfg: self.font_speaker_size.set(int(cfg["font_speaker_size"]))
            if "font_abstract_family" in cfg: self.font_abstract_family.set(cfg["font_abstract_family"])
            if "font_abstract_size" in cfg: self.font_abstract_size.set(int(cfg["font_abstract_size"]))
            if "font_abstract_spacing" in cfg: self.font_abstract_spacing.set(int(cfg["font_abstract_spacing"]))

            if "scale_min" in cfg: self.scale_min.set(int(cfg["scale_min"]))
            if "scale_max" in cfg: self.scale_max.set(int(cfg["scale_max"]))
            if "show_current_rank" in cfg: self.show_current_rank.set(bool(cfg["show_current_rank"]))

            if "window_geometry" in cfg:
                self.geometry(cfg["window_geometry"])

            if "ranking_window_geometry" in cfg:
                self.ranking_window_geometry = cfg["ranking_window_geometry"]
            if "abstain_window_geometry" in cfg:
                self.abstain_window_geometry = cfg["abstain_window_geometry"]

        except Exception:
            # If state is malformed, ignore safely
            return

        # Ensure scheduler uses updated explore rate
        if self.scheduler is not None:
            self.scheduler.explore_rate = self.explore_rate.get()

        # Update label and other dependents
        self._on_explore_change()

    def _autosave(self) -> None:
        if self.csv_path is None or self.engine is None:
            return
        # Capture ranking window geometry if valid
        if self.ranking_win is not None:
             try:
                 self.ranking_window_geometry = self.ranking_win.geometry()
             except Exception:
                 pass

        app_cfg = {
            "compare_size": int(self.compare_size.get()),
            "target_appearances_per_talk": int(self.target_appearances_per_talk.get()),
            "explore_rate": float(self.explore_rate.get()),
            "allow_ties": bool(self.allow_ties.get()),
            "show_speaker": bool(self.show_speaker.get()),
            "show_abstract": bool(self.show_abstract.get()),
            "round_ranking": bool(self.round_ranking.get()),
            "export_sort": self.export_sort.get(),
            "export_cols": {k: bool(v.get()) for k, v in self.export_cols.items()},
            "font_title_family": self.font_title_family.get(),
            "font_title_size": int(self.font_title_size.get()),
            "font_title_spacing": int(self.font_title_spacing.get()),
            "font_speaker_family": self.font_speaker_family.get(),
            "font_speaker_size": int(self.font_speaker_size.get()),
            "font_abstract_family": self.font_abstract_family.get(),
            "font_abstract_size": int(self.font_abstract_size.get()),
            "font_abstract_spacing": int(self.font_abstract_spacing.get()),
            "scale_min": int(self.scale_min.get()),
            "scale_max": int(self.scale_max.get()),
            "show_current_rank": bool(self.show_current_rank.get()),
            "window_geometry": self.geometry(),
        }
        if self.ranking_window_geometry:
             app_cfg["ranking_window_geometry"] = self.ranking_window_geometry
        if self.abstain_window_geometry:
             app_cfg["abstain_window_geometry"] = self.abstain_window_geometry

        try:
            save_state(self.csv_path, self.talks, self.engine, app_cfg)
        except Exception:
            # Autosave failure should not interrupt session
            pass

    def _on_close(self) -> None:
        self._autosave()
        self.destroy()

    # ---- Comparison panel rendering ----

    # ---- Comparison panel rendering ----

    def _rebuild_comparison_panels(self) -> None:
        # Update scheduler explore rate immediately
        if self.scheduler is not None:
            self.scheduler.explore_rate = self.explore_rate.get()

        # Destroy old panels
        for w in self.panels_frame.winfo_children():
            w.destroy()

        m = int(self.compare_size.get())
        self.rank_vars = []
        self.rank_combos = []
        self.title_boxes = []  # Changed from title_lbls to title_boxes (Text widgets)
        self.speaker_lbls = []
        self.rank_lbls = []  # Labels for showing current rank/score
        self.abs_boxes = []

        # Prepare fonts
        f_title = (self.font_title_family.get(), self.font_title_size.get(), "bold")
        f_speaker = (self.font_speaker_family.get(), self.font_speaker_size.get())
        f_abstract = (self.font_abstract_family.get(), self.font_abstract_size.get())

        # Use a Grid layout for perfect alignment
        # The container itself needs to be grid-managed
        matrix = ttk.Frame(self.panels_frame)
        matrix.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Configure columns (equal width)
        for c in range(m):
            matrix.columnconfigure(c, weight=1, uniform="col")

        # Configure rows
        # 0: Rank controls (fixed)
        # 1: Title (dynamic height, aligned)
        # 2: Speaker (dynamic height, aligned)
        # 3: Abstract (expands)
        matrix.rowconfigure(3, weight=1)

        for idx in range(m):
            # --- Row 0: Rank ---
            rank_var = tk.StringVar(value="")
            self.rank_vars.append(rank_var)

            row0_frame = ttk.Frame(matrix, padding=5)
            row0_frame.grid(row=0, column=idx, sticky="ew")

            ttk.Label(row0_frame, text=f"Talk {idx+1} rank:").pack(side=tk.LEFT)
            rank_cb = ttk.Combobox(
                row0_frame,
                width=5,
                state="readonly",
                values=[str(i) for i in range(1, m + 1)],
                textvariable=rank_var,
            )
            rank_cb.pack(side=tk.LEFT, padx=6)
            self.rank_combos.append(rank_cb)
            self._add_tooltip(rank_cb, "Assign rank 1 (best) to N (worst).")

            # --- Row 1: Title ---
            # Using Text widget to support line spacing
            title_box = tk.Text(
                matrix,
                width=1,  # Width is controlled by grid/resize, this just needs to be small enough to fit
                height=1, # Initial height, updated dynamically
                font=f_title,
                bg="#e0e0e0",
                relief=tk.GROOVE,
                spacing2=self.font_title_spacing.get(),
                wrap=tk.WORD,
                padx=8, pady=8,
                borderwidth=2,
                cursor="arrow"
            )
            title_box.grid(row=1, column=idx, sticky="nsew", padx=4, pady=(5, 5))
            title_box.bind("<Key>", lambda e: "break") # Read-only but selectable
            title_box.bind("<Configure>", self._on_title_configure)
            self.title_boxes.append(title_box)

            # --- Row 2: Speaker ---
            speaker_lbl = tk.Label(
                matrix,
                text="",
                font=f_speaker,
                bg="#f0f0f0",
                relief=tk.GROOVE,
                justify="left",
                anchor="nw",
                padx=8, pady=6
            )
            speaker_lbl.grid(row=2, column=idx, sticky="nsew", padx=4, pady=(0, 5))
            self.speaker_lbls.append(speaker_lbl)

            # --- Row 3: Abstract ---
            abs_frame = ttk.Frame(matrix, padding=4)
            abs_frame.grid(row=3, column=idx, sticky="nsew")

            abs_box = ScrolledText(
                abs_frame,
                height=10,
                wrap=tk.WORD,
                font=f_abstract,
                spacing2=self.font_abstract_spacing.get()
            )
            abs_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            abs_box.configure(state="disabled")
            self.abs_boxes.append(abs_box)

            # Label for Rank/Score (below abstract)
            r_lbl = ttk.Label(abs_frame, text="", foreground="#555555", font=("TkDefaultFont", 9))
            r_lbl.pack(side=tk.BOTTOM, pady=(5,0))
            self.rank_lbls.append(r_lbl)

        # Bind resize event to adjust wraplength dynamically
        matrix.bind("<Configure>", self._on_panel_resize)

        self._render_current_set()
        self._update_status()

        # If we already have a session, fetch a new set sized correctly
        if self.engine is not None and self.scheduler is not None and self.talks:
            self.on_skip()

    # ---- Hotkey handlers ----

    def _on_title_configure(self, event: tk.Event) -> None:
        """Handle resize of title box to adjust height."""
        # Check if widget is destroyed or not valid
        try:
            box = event.widget
            if not isinstance(box, tk.Text):
                return
            self._update_title_height(box, force_update=False)
        except Exception:
            pass

    def _update_title_height(self, box: tk.Text, force_update: bool = False) -> None:
        """Auto-adjust height of title text box based on content."""
        if force_update:
            box.update_idletasks()
        try:
            # Count display lines
            count = box.count("1.0", "end", "displaylines")
            if count:
                lines = int(count[0])
            else:
                lines = 1
        except Exception:
            lines = 1

        # Only configure if changed to avoid loops
        if int(box.cget("height")) != lines:
            box.configure(height=lines)

    def _on_panel_resize(self, event: tk.Event) -> None:
        if not self.title_boxes:
            return
        m = len(self.title_boxes)
        if m == 0:
            return

        # Approximate width per column (total width / m) - padding
        col_width = (event.width / m) - 30
        if col_width < 100:
            col_width = 100

        # For Speakers (Labels), update wraplength
        for lbl in self.speaker_lbls:
            lbl.configure(wraplength=int(col_width))

        # Title boxes update themselves via <Configure> binding

    def _render_current_set(self) -> None:
        m = int(self.compare_size.get())

        # Clear/Reset
        for i in range(len(self.title_boxes)):
            if i < len(self.title_boxes):
                self.title_boxes[i].configure(state="normal")
                self.title_boxes[i].delete("1.0", tk.END)
                self.title_boxes[i].configure(state="disabled")
                self.title_boxes[i].configure(height=1)

            if i < len(self.speaker_lbls): self.speaker_lbls[i].configure(text="")
            if i < len(self.rank_lbls): self.rank_lbls[i].configure(text="")
            if i < len(self.abs_boxes):
                self.abs_boxes[i].configure(state="normal")
                self.abs_boxes[i].delete("1.0", tk.END)
                self.abs_boxes[i].configure(state="disabled")

        if not self.current_ids:
            return

        for idx, tid in enumerate(self.current_ids[:m]):
            t = self.talks.get(tid)
            if t is None:
                continue

            if idx < len(self.title_boxes):
                title_txt = t.title or f"(No title) [{t.talk_id}]"
                box = self.title_boxes[idx]
                box.configure(state="normal")
                box.insert("1.0", title_txt)
                # Force update initially to set correct height before display
                self._update_title_height(box, force_update=True)
                box.configure(state="disabled")

            if idx < len(self.speaker_lbls):
                txt = f"Speaker: {t.speaker}" if self.show_speaker.get() else ""
                self.speaker_lbls[idx].configure(text=txt)

            if idx < len(self.abs_boxes):
                txt = t.abstract if self.show_abstract.get() else ""
                box = self.abs_boxes[idx]
                box.configure(state="normal")
                box.insert("1.0", txt or "(No abstract)")
                box.configure(state="disabled")

            if idx < len(self.rank_lbls):
                if self.show_current_rank.get():
                     # Calculate min/max elo for scaling (re-calculation is cheap enough here)
                    ranked = self.engine.ranked_ids()
                    all_ratings = [self.engine.ratings[x] for x in ranked]
                    min_elo = min(all_ratings) if all_ratings else 0
                    max_elo = max(all_ratings) if all_ratings else 1

                    user_min = float(self.scale_min.get())
                    user_max = float(self.scale_max.get())

                    rating = self.engine.ratings.get(tid, self.engine.base_rating)

                    if max_elo > min_elo:
                        norm = (rating - min_elo) / (max_elo - min_elo)
                        score = user_min + norm * (user_max - user_min)
                    else:
                        score = (user_min + user_max) / 2.0

                    if self.round_ranking.get():
                        self.rank_lbls[idx].configure(text=f"Elo: {rating:.0f} | Score: {score:.0f}")
                    else:
                        self.rank_lbls[idx].configure(text=f"Elo: {rating:.0f} | Score: {score:.2f}")
                else:
                    self.rank_lbls[idx].configure(text="")

        # Reset ranks to blank each new set (forces explicit judgment)
        for rv, cb in zip(self.rank_vars, self.rank_combos):
            rv.set("")
            cb.set("")

    # ---- Actions ----

    def on_skip(self) -> None:
        """Load a new comparison set without updating ratings."""
        if self.scheduler is None or self.engine is None:
            return
        m = int(self.compare_size.get())
        try:
            ids = self.scheduler.choose_next_set(m)
        except Exception as e:
            messagebox.showerror("Cannot pick next set", str(e))
            return
        self.current_ids = ids
        self._render_current_set()
        self._update_status()

    def on_submit(self, event: Optional[tk.Event] = None) -> None:
        if self.engine is None or self.scheduler is None:
            return
        m = int(self.compare_size.get())
        if len(self.current_ids) != m:
            self.on_skip()
            return

        # Collect ranks
        ranks: Dict[str, int] = {}
        for tid, rv in zip(self.current_ids, self.rank_vars):
            v = (rv.get() or "").strip()
            if not v:
                messagebox.showwarning("Missing ranks", "Please assign a rank (1..N) for every displayed talk.")
                return
            try:
                ranks[tid] = int(v)
            except ValueError:
                messagebox.showwarning("Invalid rank", "Ranks must be integers.")
                return

        try:
            self.engine.update_from_ranks(self.current_ids, ranks, allow_ties=self.allow_ties.get())
        except Exception as e:
            messagebox.showerror("Cannot submit ranking", str(e))
            return

        self._autosave()
        self.on_skip()  # next set

    def on_undo(self) -> None:
        if self.engine is None:
            return
        if not self.engine.history:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        ok = self.engine.undo_last()
        if ok:
            self._autosave()
            self._update_status()
            # Show a new set (we do not re-show the previous set by default)
            self.on_skip()

    def on_show_rankings(self) -> None:
        if self.engine is None:
            return

        if self.ranking_win is not None:
             self.ranking_win.lift()
             return

        win = tk.Toplevel(self)
        self.ranking_win = win
        win.title("Current Ranking")

        if self.ranking_window_geometry:
             win.geometry(self.ranking_window_geometry)
        else:
             win.geometry("1300x900")

        def _on_rank_close() -> None:
             self.ranking_window_geometry = win.geometry()
             self.ranking_win = None
             win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_rank_close)

        cols = ("Place", "Rating", "Score", "Rounds", "Pairwise", "ID", "Speaker", "Title", "Track")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, anchor="w", width=140 if c in ("Title", "Speaker") else 80)
        tree.column("Title", width=420)
        tree.column("Speaker", width=180)

        yscroll = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        # Tag config for visibility
        tree.tag_configure("abstained_row", foreground="red")

        ranked = self.engine.ranked_ids()

        # Split into valid and abstained for display
        valid_ranked = [tid for tid in ranked if tid not in self.engine.abstained_ids]
        abstained = [tid for tid in ranked if tid in self.engine.abstained_ids]

        # Calculate min/max only from valid
        valid_ratings = [self.engine.ratings[tid] for tid in valid_ranked]
        min_elo = min(valid_ratings) if valid_ratings else 0
        max_elo = max(valid_ratings) if valid_ratings else 1

        user_min = float(self.scale_min.get())
        user_max = float(self.scale_max.get())

        # Show valid first
        for i, tid in enumerate(valid_ranked, start=1):
            t = self.talks[tid]
            rating = self.engine.ratings[tid]

            # Linear mapping
            if max_elo > min_elo:
                 norm = (rating - min_elo) / (max_elo - min_elo)
                 score = user_min + norm * (user_max - user_min)
            else:
                 score = (user_min + user_max) / 2.0

            r_fmt = f"{rating:.0f}" if self.round_ranking.get() else f"{rating:.2f}"
            s_fmt = f"{score:.0f}" if self.round_ranking.get() else f"{score:.2f}"

            tree.insert(
                "",
                "end",
                values=(
                    i,
                    r_fmt,
                    s_fmt,
                    self.engine.rounds_seen[tid],
                    self.engine.pairwise_seen[tid],
                    tid,
                    t.speaker,
                    t.title,
                    t.track,
                ),
            )

        # Show abstained at bottom
        for tid in abstained:
            t = self.talks[tid]
            tree.insert(
                "",
                "end",
                values=(
                    "",
                    "",
                    "",
                    self.engine.rounds_seen[tid],
                    self.engine.pairwise_seen[tid],
                    tid,
                    t.speaker,
                    t.title,
                    t.track,
                ),
                tags=("abstained",)
            )

        tree.tag_configure("abstained", foreground="gray")

    def on_manage_abstentions(self) -> None:
        if self.engine is None or not self.talks:
            return

        win = tk.Toplevel(self)
        if self.abstain_window_geometry:
            win.geometry(self.abstain_window_geometry)
        else:
            win.geometry("900x700")
            # Center in parent
            x = self.winfo_rootx() + (self.winfo_width() // 2) - 450
            y = self.winfo_rooty() + (self.winfo_height() // 2) - 350
            win.geometry(f"+{x}+{y}")

        def _on_win_close():
            self.abstain_window_geometry = win.geometry()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_win_close)

        # Control frame (Filter + Show Only Abstained)
        ctrl_frame = ttk.Frame(win, padding=5)
        ctrl_frame.pack(fill=tk.X)

        ttk.Label(ctrl_frame, text="Filter:").pack(side=tk.LEFT)
        filter_var = tk.StringVar()
        filter_entry = ttk.Entry(ctrl_frame, textvariable=filter_var)
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        show_only_abstained = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="Show only abstained", variable=show_only_abstained).pack(side=tk.LEFT, padx=5)

        # Treeview
        tree_frame = ttk.Frame(win, padding=5)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("Abstained", "ID", "Speaker", "Title")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")

        tree.heading("Abstained", text="Abstain?", command=lambda: _sort_by("Abstained"))
        tree.heading("ID", text="ID", command=lambda: _sort_by("ID"))
        tree.heading("Speaker", text="Speaker", command=lambda: _sort_by("Speaker"))
        tree.heading("Title", text="Title", command=lambda: _sort_by("Title"))

        tree.column("Abstained", width=80, anchor="center")
        tree.column("ID", width=60, anchor="center")
        tree.column("Speaker", width=200, anchor="w")
        tree.column("Title", width=400, anchor="w")

        yscroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        # Tag config for visibility
        tree.tag_configure("abstained_row", foreground="red")

        # Internal state
        # We work on a copy of abstained_ids
        current_abstained = set(self.engine.abstained_ids)

        # Sorting state
        sort_col = "ID"
        sort_reverse = False

        def _populate(*args):
             # Clear tree
             for item in tree.get_children():
                 tree.delete(item)

             f_text = filter_var.get().lower()
             only_abs = show_only_abstained.get()

             data = []
             all_talks = list(self.talks.values())

             for t in all_talks:
                 is_abs = t.talk_id in current_abstained

                 # Filters
                 if only_abs and not is_abs:
                     continue
                 if f_text:
                     if (f_text not in t.title.lower() and
                         f_text not in t.speaker.lower() and
                         f_text not in t.talk_id):
                         continue

                 symbol = "YES" if is_abs else "-"
                 data.append((symbol, t.talk_id, t.speaker, t.title, is_abs))

             # Sort
             # key index mapping
             key_idx = {"Abstained": 4, "ID": 1, "Speaker": 2, "Title": 3}
             idx = key_idx.get(sort_col, 1)

             # Special sort for ID if numeric
             if sort_col == "ID":
                 try:
                    data.sort(key=lambda x: int(x[idx]), reverse=sort_reverse)
                 except ValueError:
                    data.sort(key=lambda x: x[idx], reverse=sort_reverse)
             else:
                 data.sort(key=lambda x: str(x[idx]).lower(), reverse=sort_reverse)

             for item in data:
                 # item: (symbol, tid, speaker, title, is_abs)
                 # values excludes is_abs
                 tree.insert("", "end", values=item[:4], tags=("abstained_row",) if item[4] else ())

        def _sort_by(col):
            nonlocal sort_col, sort_reverse
            if sort_col == col:
                sort_reverse = not sort_reverse
            else:
                sort_col = col
                sort_reverse = False

            # Update heading arrows (optional aesthetic, skipping for simplicity or adding simple markers)
            for c in cols:
                text = c if c != "Abstained" else "Abstain?"
                suffix = " ▼" if (c == sort_col and not sort_reverse) else " ▲" if (c == sort_col and sort_reverse) else ""
                tree.heading(c, text=text + suffix)

            _populate()

        def _on_click(event):
            region = tree.identify_region(event.x, event.y)
            if region == "heading":
                return

            item_id = tree.identify_row(event.y)
            if not item_id:
                return

            col = tree.identify_column(event.x)
            # col is like '#1', '#2'...

            if col == "#1":
                vals = tree.item(item_id, "values")
                tid = vals[1]

                if tid in current_abstained:
                    current_abstained.remove(tid)
                else:
                    current_abstained.add(tid)

                # New symbol
                new_sym = "YES" if tid in current_abstained else "-"
                # vals is a tuple, need list to modify
                new_vals = list(vals)
                new_vals[0] = new_sym
                # Also update tag for immediate feedback
                if tid in current_abstained:
                    tree.item(item_id, tags=("abstained_row",))
                else:
                    tree.item(item_id, tags=())
                tree.item(item_id, values=new_vals)

        tree.bind("<Button-1>", _on_click)

        # Initial populate
        _populate()

        # Traces
        filter_var.trace_add("write", lambda *args: _populate())
        show_only_abstained.trace_add("write", lambda *args: _populate())

        # Save
        def _save():
            if current_abstained != self.engine.abstained_ids:
                self.engine.abstained_ids = current_abstained
                self._autosave()

                # If current set has abstained talks, skip
                if any(tid in current_abstained for tid in self.current_ids):
                    self.on_skip()

                messagebox.showinfo("Abstentions Updated", f"Marked {len(current_abstained)} talks as abstained.")
            _on_win_close()

        btn_frame = ttk.Frame(win, padding=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(btn_frame, text="Save & Close", command=_save).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Cancel", command=_on_win_close).pack(side=tk.RIGHT, padx=5)

    def on_export(self) -> None:
        if self.engine is None or not self.talks:
            return

        # Create configuration dialog
        win = tk.Toplevel(self)
        win.title("Export Configuration")
        win.geometry("400x450")

        # Center in parent
        x = self.winfo_rootx() + (self.winfo_width() // 2) - 200
        y = self.winfo_rooty() + (self.winfo_height() // 2) - 225
        win.geometry(f"+{x}+{y}")

        # Columns Section
        lf_cols = ttk.LabelFrame(win, text="Included Columns", padding=10)
        lf_cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        col_order = ["ID", "Place", "Elo", "Score", "Speaker", "Title", "Track"]
        for col in col_order:
            if col in self.export_cols:
                ttk.Checkbutton(lf_cols, text=col, variable=self.export_cols[col]).pack(anchor="w", pady=2)

        # Sorting Section
        lf_sort = ttk.LabelFrame(win, text="Sort Order", padding=10)
        lf_sort.pack(fill=tk.X, padx=10, pady=5)

        ttk.Radiobutton(lf_sort, text="By Rank (Best first)", variable=self.export_sort, value="rank").pack(anchor="w", pady=2)
        ttk.Radiobutton(lf_sort, text="By ID", variable=self.export_sort, value="id").pack(anchor="w", pady=2)

        # Buttons
        btn_frame = ttk.Frame(win, padding=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        def _do_export() -> None:
             # Gather columns
             selected = [c for c in col_order if self.export_cols[c].get()]
             if not selected:
                 messagebox.showwarning("No columns", "Please select at least one column to export.", parent=win)
                 return

             out = filedialog.asksaveasfilename(
                 title="Export Ranking CSV",
                 defaultextension=".csv",
                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                 parent=win
             )
             if not out:
                 return

             try:
                export_rankings_csv(
                    Path(out),
                    self.talks,
                    self.engine,
                    scale_min=float(self.scale_min.get()),
                    scale_max=float(self.scale_max.get()),
                    round_ranking=self.round_ranking.get(),
                    columns=selected,
                    sort_by=self.export_sort.get(),
                )
             except Exception as e:
                messagebox.showerror("Export failed", str(e), parent=win)
                return

             self._autosave()  # Persist config
             messagebox.showinfo("Export complete", f"Exported to:\n{out}", parent=win)
             win.destroy()

        ttk.Button(btn_frame, text="Export...", command=_do_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=5)

    def on_font_settings(self) -> None:
        win = tk.Toplevel(self)
        win.title("Font Settings")
        win.geometry("600x350")

        # Gather fonts
        all_fonts = sorted(font.families())


        def _add_row(
            parent: ttk.Frame,
            label: str,
            row: int,
            family_var: tk.StringVar,
            size_var: tk.IntVar,
            spacing_var: Optional[tk.IntVar] = None
        ) -> None:
            ttk.Label(parent, text=label, font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, sticky="w", pady=10)

            f_cb = ttk.Combobox(parent, values=all_fonts, textvariable=family_var, state="readonly", width=25)
            f_cb.grid(row=row, column=1, padx=10)

            s_sp = ttk.Spinbox(parent, from_=6, to=72, textvariable=size_var, width=5)
            s_sp.grid(row=row, column=2, padx=10)

            if spacing_var is not None:
                ttk.Label(parent, text="Spacing:").grid(row=row, column=3, padx=(10, 2))
                sp_box = ttk.Spinbox(parent, from_=0, to=50, textvariable=spacing_var, width=4)
                sp_box.grid(row=row, column=4, padx=2)

        frm = ttk.Frame(win, padding=20)
        frm.pack(fill=tk.BOTH, expand=True)

        _add_row(frm, "Title Font:", 0, self.font_title_family, self.font_title_size, self.font_title_spacing)
        _add_row(frm, "Speaker Font:", 1, self.font_speaker_family, self.font_speaker_size)
        _add_row(frm, "Abstract Font:", 2, self.font_abstract_family, self.font_abstract_size, self.font_abstract_spacing)

        def _apply(close: bool = False) -> None:
            self._rebuild_comparison_panels()
            self._autosave()
            if close:
                win.destroy()

        def _reset() -> None:
            # self.font_title_family.set("Helvetica")
            self.font_title_family.set("Arimo")
            self.font_title_size.set(12)
            # self.font_speaker_family.set("Helvetica")
            self.font_speaker_family.set("Arimo")
            self.font_speaker_size.set(11)
            # self.font_abstract_family.set("Times")
            self.font_abstract_family.set("Arimo")
            self.font_abstract_size.set(12)
            self.font_abstract_spacing.set(2)

        btn_frame = ttk.Frame(win, padding=20)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(btn_frame, text="Reset to Default", command=_reset).pack(side=tk.LEFT)

        ttk.Button(btn_frame, text="Apply & Close", command=lambda: _apply(close=True)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Apply", command=lambda: _apply(close=False)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=5)

    # ---- Hotkey handlers ----

    def _on_rank_shortcut(self, idx: int) -> None:
        m = int(self.compare_size.get())
        if idx < 0 or idx >= m:
            return

        # Safety check if panels are built
        if idx >= len(self.rank_vars) or idx >= len(self.rank_combos):
            return

        # If already ranked, do nothing
        current_val = self.rank_vars[idx].get().strip()
        if current_val:
            return

        # Determine next rank to assign
        assigned_count = sum(1 for rv in self.rank_vars if rv.get().strip())
        next_rank = assigned_count + 1

        if next_rank > m:
            return

        val = str(next_rank)
        # Set both var and widget for safety
        self.rank_vars[idx].set(val)
        self.rank_combos[idx].set(val)

    def _on_clear_ranks(self, event: Optional[tk.Event] = None) -> None:
        for rv, cb in zip(self.rank_vars, self.rank_combos):
            rv.set("")
            cb.set("")

    # ---- Status & config changes ----

    def _on_explore_change(self) -> None:
        self._explore_label.configure(text=f"{self.explore_rate.get():.2f}")
        if self.scheduler is not None:
            self.scheduler.explore_rate = self.explore_rate.get()
        self._update_status()
        self._autosave()

    def _update_status(self) -> None:
        if self.engine is None or not self.talks:
            self.status_var.set("No session loaded.")
            return

        m = int(self.compare_size.get())
        n = len(self.talks)

        tgt = max(1, int(self.target_appearances_per_talk.get()))
        required_appearances = n * tgt

        done_appearances = sum(self.engine.rounds_seen.values())
        # Expected comparisons (rounds) depends on current m
        expected_rounds = math.ceil(required_appearances / m)
        done_rounds = self.engine.rounds_done

        pct = 0.0 if expected_rounds <= 0 else (100.0 * done_rounds / expected_rounds)
        pct = max(0.0, min(100.0, pct))

        num_abstained = len(self.engine.abstained_ids)
        self.status_var.set(
            f"Comparisons expected: {expected_rounds}   "
            f"Comparisons done: {done_rounds}   "
            f"Total talks: {n}   "
            f"Abstained: {num_abstained}   "
            f"Progress: {pct:.1f}%"
        )

    def _render_current_set_and_status(self) -> None:
        self._render_current_set()
        self._update_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tkinter Elo-based talk ranker.")
    parser.add_argument("--csv", type=str, default="", help="Path to talks CSV (columns: ID, Title, Speaker, Abstract, Track)")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    if csv_path is not None and not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    app = TalkRankerApp(csv_path=csv_path)
    app.mainloop()


if __name__ == "__main__":
    main()
