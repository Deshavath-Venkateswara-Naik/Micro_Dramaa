"""Tests for the MicrodramaGenerator service.

The backend imports its packages as `services.X` / `routers.X` (the app runs
from the `backend/` directory), so we ensure that directory is importable here.
"""
import os
import sys

# Make the backend package root (parent of this tests/ dir) importable so that
# `from services.microdrama_generator import MicrodramaGenerator` resolves the
# same way it does when the FastAPI app runs.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services.microdrama_generator import MicrodramaGenerator  # noqa: E402
import re

from hypothesis import given, settings
from hypothesis import strategies as st

# Pattern for the Time_Format "HH:MM:SS.mmm". Hours may exceed two digits for
# very long timelines, but minutes/seconds are always two digits and there are
# always exactly three millisecond digits.
TIME_FORMAT_RE = re.compile(r"^\d{2,}:\d{2}:\d{2}\.\d{3}$")

# Tolerance accounting for the 3-decimal rounding in `f"{s:06.3f}"`: the worst
# case rounding error is 0.0005s; we use 0.0015s to also absorb float noise.
ROUND_TRIP_TOLERANCE = 0.0015


# Feature: microdrama-generator, Property 6: timestamp format round-trip
@settings(max_examples=200)
@given(
    seconds=st.floats(
        min_value=0,
        max_value=359999.999,  # up to ~99:59:59.999
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_timestamp_format_round_trip(seconds):
    """Property 6: Timestamp format round-trip.

    For any non-negative float second value, formatting it to the
    "HH:MM:SS.mmm" string and parsing it back must recover the original value
    within millisecond tolerance, and the formatted string must match the
    Time_Format pattern.

    Validates: Requirements 6.5
    """
    formatted = MicrodramaGenerator._seconds_to_time(seconds)

    # The formatted string must follow the HH:MM:SS.mmm Time_Format.
    assert TIME_FORMAT_RE.match(formatted), f"bad format: {formatted!r}"

    # Round-trip must recover the original value within millisecond tolerance.
    recovered = MicrodramaGenerator._time_to_seconds(formatted)
    assert abs(recovered - seconds) < ROUND_TRIP_TOLERANCE, (
        f"round-trip mismatch: {seconds} -> {formatted!r} -> {recovered}"
    )


# ---------------------------------------------------------------------------
# Unit tests for input loading: `_read_json_strict` (Task 3.2)
#
# These are example-based tests (no hypothesis). They use pytest's `tmp_path`
# fixture so no real storage directory is touched. They cover the scenes and
# dialogue inputs for both the present/valid and missing/invalid-JSON cases,
# asserting that the returned error names the offending file.
#
# Requirements: 1.3, 1.4, 1.5
# ---------------------------------------------------------------------------
import json  # noqa: E402

SCENES_FILENAME = "scenes_and_plot.json"
DIALOGUE_FILENAME = "dialogue_diarization.json"


def _make_generator(tmp_path):
    """Build a MicrodramaGenerator rooted at an isolated tmp directory.

    __init__ attempts to init the genai client; when GCP creds are absent it
    logs a warning and sets `llm_client = None`. Construction still succeeds,
    which is all these I/O tests need.
    """
    return MicrodramaGenerator(output_base_dir=str(tmp_path))


def test_read_json_strict_scenes_present_valid(tmp_path):
    """Scenes file present + valid JSON -> (data, None) with parsed content.

    Validates: Requirements 1.3 (success path counterpart)
    """
    generator = _make_generator(tmp_path)
    scenes_content = {
        "total_scenes": 1,
        "Scenes": [
            {
                "scene_number": 2,
                "start_time": "00:00:02.200",
                "end_time": "00:02:53.233",
            }
        ],
        "plot_of_the_movie": "A long prose plot string.",
    }
    scenes_path = tmp_path / SCENES_FILENAME
    scenes_path.write_text(json.dumps(scenes_content), encoding="utf-8")

    data, error = generator._read_json_strict(scenes_path)

    assert error is None
    assert data == scenes_content


def test_read_json_strict_scenes_absent_names_file(tmp_path):
    """Scenes file absent -> (None, error) naming the scenes file.

    Validates: Requirements 1.3
    """
    generator = _make_generator(tmp_path)
    scenes_path = tmp_path / SCENES_FILENAME  # never created

    data, error = generator._read_json_strict(scenes_path)

    assert data is None
    assert error is not None
    assert SCENES_FILENAME in error


def test_read_json_strict_dialogue_present_valid(tmp_path):
    """Dialogue file present + valid JSON -> (data, None) with parsed content.

    Validates: Requirements 1.4 (success path counterpart)
    """
    generator = _make_generator(tmp_path)
    dialogue_content = {
        "dialogues": [
            {"speaker": "SPEAKER_26", "start": 10.77, "end": 12.57, "text": "Telugu line"}
        ]
    }
    dialogue_path = tmp_path / DIALOGUE_FILENAME
    dialogue_path.write_text(json.dumps(dialogue_content), encoding="utf-8")

    data, error = generator._read_json_strict(dialogue_path)

    assert error is None
    assert data == dialogue_content


def test_read_json_strict_dialogue_absent_names_file(tmp_path):
    """Dialogue file absent -> (None, error) naming the dialogue file.

    Validates: Requirements 1.4
    """
    generator = _make_generator(tmp_path)
    dialogue_path = tmp_path / DIALOGUE_FILENAME  # never created

    data, error = generator._read_json_strict(dialogue_path)

    assert data is None
    assert error is not None
    assert DIALOGUE_FILENAME in error


def test_read_json_strict_scenes_invalid_json_names_file(tmp_path):
    """Invalid JSON in the scenes file -> (None, error) naming that file.

    Validates: Requirements 1.5
    """
    generator = _make_generator(tmp_path)
    scenes_path = tmp_path / SCENES_FILENAME
    scenes_path.write_text("{ this is not valid json ", encoding="utf-8")

    data, error = generator._read_json_strict(scenes_path)

    assert data is None
    assert error is not None
    assert SCENES_FILENAME in error


def test_read_json_strict_dialogue_invalid_json_names_file(tmp_path):
    """Invalid JSON in the dialogue file -> (None, error) naming that file.

    Validates: Requirements 1.5
    """
    generator = _make_generator(tmp_path)
    dialogue_path = tmp_path / DIALOGUE_FILENAME
    dialogue_path.write_text("{ broken: , }", encoding="utf-8")

    data, error = generator._read_json_strict(dialogue_path)

    assert data is None
    assert error is not None
    assert DIALOGUE_FILENAME in error


# ---------------------------------------------------------------------------
# Property test for scene validity partition: `_validate_scenes` (Task 4.2)
#
# Property 1 concerns how `_validate_scenes` partitions an arbitrary list of
# scene records into (a) the valid set used for processing and (b) the
# excluded set whose `scene_number` values are recorded. A record is valid iff
# it has BOTH `start_time` and `end_time` present; otherwise it is excluded.
#
# To make the partition deterministically checkable we assign a UNIQUE
# `scene_number` to every generated record (its index in the list), so we can
# match retained vs excluded records by number without ambiguity.
# ---------------------------------------------------------------------------

# A "present" time value: either a valid "HH:MM:SS.mmm" string or numeric
# seconds. Crucially never None, so presence is unambiguous (the service keys
# exclusion on the value being None).
_time_string_strategy = st.builds(
    lambda h, m, s: f"{h:02d}:{m:02d}:{s:06.3f}",
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=59),
    st.floats(min_value=0, max_value=59.999, allow_nan=False, allow_infinity=False),
)
_time_value_strategy = st.one_of(
    _time_string_strategy,
    st.floats(min_value=0, max_value=100000, allow_nan=False, allow_infinity=False),
)

# Optional free-form fields a scene record may carry. Kept small/simple since
# the partition does not depend on them.
_chars_strategy = st.lists(
    st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=8),
    max_size=4,
)
_text_strategy = st.text(max_size=30)


@st.composite
def _scene_record_strategy(draw):
    """Generate a single scene record that randomly includes start_time and/or
    end_time (and randomly includes optional metadata fields).

    `scene_number` is intentionally NOT set here; it is assigned by index in
    `_scenes_input_strategy` to guarantee uniqueness across the list.
    """
    has_start = draw(st.booleans())
    has_end = draw(st.booleans())

    record = {}
    if has_start:
        record["start_time"] = draw(_time_value_strategy)
    if has_end:
        record["end_time"] = draw(_time_value_strategy)

    # Optional metadata that should not influence the partition.
    if draw(st.booleans()):
        record["characters_present"] = draw(_chars_strategy)
    if draw(st.booleans()):
        record["setting"] = draw(_text_strategy)
    if draw(st.booleans()):
        record["description"] = draw(_text_strategy)
    return record


@st.composite
def _scenes_input_strategy(draw):
    """Wrap a non-empty list of scene records into a Scenes_Input dict.

    Each record is assigned a UNIQUE `scene_number` equal to its index, so the
    valid/excluded partition can be checked deterministically. The list is
    non-empty (min_size=1) so `_validate_scenes` does not short-circuit with the
    missing/empty-`Scenes` structural error and the partition behavior is what
    is exercised.
    """
    records = draw(st.lists(_scene_record_strategy(), min_size=1, max_size=12))
    for idx, rec in enumerate(records):
        rec["scene_number"] = idx
    return {"Scenes": records, "plot_of_the_movie": draw(st.text(max_size=50))}


# A single generator instance reused across all generated examples.
# `_validate_scenes` is pure (no filesystem / no LLM access), so the base dir
# value is irrelevant and we avoid re-initializing the genai client per example.
_PARTITION_GENERATOR = MicrodramaGenerator(output_base_dir=os.path.join(BACKEND_DIR, "_pbt_partition_tmp"))


# Feature: microdrama-generator, Property 1: scene validity partition
@settings(max_examples=200)
@given(scenes_input=_scenes_input_strategy())
def test_scene_validity_partition(scenes_input):
    """Property 1: Scene validity partition.

    For any Scenes_Input with an arbitrary list of scene records (each with a
    unique scene_number), `_validate_scenes` must:
      - retain every record that has BOTH start_time and end_time (its
        scene_number appears among the valid scenes), and
      - exclude every record missing either time (its scene_number appears in
        excluded_scene_numbers and NOT among the valid scenes).
    The valid and excluded scene-number sets must form a complete, disjoint
    partition over all records.

    Validates: Requirements 2.2
    """
    valid_scenes, excluded_scene_numbers, error = _PARTITION_GENERATOR._validate_scenes(scenes_input)

    # Non-empty Scenes array => no structural error; the partition applies.
    assert error is None

    records = scenes_input["Scenes"]

    # Expected partition computed directly from the presence of both times.
    expected_valid = {
        rec["scene_number"]
        for rec in records
        if rec.get("start_time") is not None and rec.get("end_time") is not None
    }
    expected_excluded = {
        rec["scene_number"]
        for rec in records
        if rec.get("start_time") is None or rec.get("end_time") is None
    }

    actual_valid = {vs["scene_number"] for vs in valid_scenes}
    actual_excluded = set(excluded_scene_numbers)

    # scene_numbers are unique, so there should be no collapsing into a set.
    assert len(actual_valid) == len(valid_scenes)
    assert len(actual_excluded) == len(excluded_scene_numbers)

    # Every fully-timed record retained; every under-timed record excluded.
    assert actual_valid == expected_valid
    assert actual_excluded == expected_excluded

    # Partition is disjoint and complete over all generated records.
    all_numbers = {rec["scene_number"] for rec in records}
    assert actual_valid.isdisjoint(actual_excluded)
    assert actual_valid | actual_excluded == all_numbers


# ---------------------------------------------------------------------------
# Unit tests for the missing/empty Scenes array: `_validate_scenes` (Task 4.3)
#
# These are example-based tests (NO hypothesis). They cover the structural
# failure modes where the Scenes_Input does not carry a non-empty `Scenes`
# list. In every such case `_validate_scenes` must short-circuit and return
# ([], [], error) with a non-empty, descriptive error that mentions the
# `Scenes` array. The `_make_generator` helper (defined above) builds a
# generator rooted at pytest's `tmp_path` so no real storage is touched.
#
# Requirements: 2.1
# ---------------------------------------------------------------------------


def test_validate_scenes_missing_key_returns_descriptive_error(tmp_path):
    """Case A: Scenes_Input with NO `Scenes` key -> ([], [], error).

    The error must be a non-empty descriptive string that mentions the
    `Scenes` array, and both returned lists must be empty.

    Validates: Requirements 2.1
    """
    generator = _make_generator(tmp_path)
    scenes_input = {"plot_of_the_movie": "A long prose plot string."}

    valid_scenes, excluded_scene_numbers, error = generator._validate_scenes(scenes_input)

    assert valid_scenes == []
    assert excluded_scene_numbers == []
    assert error  # truthy / non-empty
    assert isinstance(error, str)
    assert "Scenes" in error


def test_validate_scenes_empty_list_returns_descriptive_error(tmp_path):
    """Case B: Scenes_Input with an empty `Scenes` list -> ([], [], error).

    An empty array carries no scene records, so the same structural error is
    returned with both lists empty.

    Validates: Requirements 2.1
    """
    generator = _make_generator(tmp_path)
    scenes_input = {"Scenes": []}

    valid_scenes, excluded_scene_numbers, error = generator._validate_scenes(scenes_input)

    assert valid_scenes == []
    assert excluded_scene_numbers == []
    assert error  # truthy / non-empty
    assert isinstance(error, str)
    assert "Scenes" in error


def test_validate_scenes_non_list_value_returns_descriptive_error(tmp_path):
    """Optional: `Scenes` present but not a list -> ([], [], error).

    A non-list `Scenes` value cannot be partitioned into scene records, so the
    structural failure path applies just like the missing/empty cases.

    Validates: Requirements 2.1
    """
    generator = _make_generator(tmp_path)
    scenes_input = {"Scenes": "oops"}

    valid_scenes, excluded_scene_numbers, error = generator._validate_scenes(scenes_input)

    assert valid_scenes == []
    assert excluded_scene_numbers == []
    assert error  # truthy / non-empty
    assert isinstance(error, str)
    assert "Scenes" in error


# ---------------------------------------------------------------------------
# Property test for timeline bounds correctness: `_compute_timeline_bounds`
# (Task 4.5)
#
# Property 2 concerns the pure min/max computation `_compute_timeline_bounds`
# performs over a non-empty set of valid scenes. Each valid scene carries float
# `start_sec`/`end_sec` keys (as produced by `_validate_scenes`). The bounds are
# defined purely as the minimum `start_sec` and the maximum `end_sec` across the
# scenes; the relationship between an individual scene's start and end is
# irrelevant, so we deliberately do NOT require `start_sec <= end_sec`.
#
# Because the method performs no arithmetic (only `min`/`max` selection over the
# exact float values handed in), the result must equal the expected min/max
# EXACTLY, so we assert exact equality.
# ---------------------------------------------------------------------------

# Finite, non-NaN float seconds spanning a wide but bounded timeline (0 to
# ~100 hours). Bounds keep generation fast while still exercising large values.
_SEC_FLOAT_STRATEGY = st.floats(
    min_value=0,
    max_value=360000,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _valid_scene_strategy(draw):
    """Generate a single normalized valid-scene dict.

    Only `start_sec`/`end_sec` (finite floats) matter for the bounds
    computation; `scene_number` and other metadata are included to mirror the
    real ValidScene shape but are never inspected by `_compute_timeline_bounds`.
    Note: `start_sec` and `end_sec` are drawn independently, so a scene may have
    `start_sec > end_sec` — Property 2 must hold regardless.
    """
    return {
        "scene_number": draw(st.integers(min_value=0, max_value=1000)),
        "start_sec": draw(_SEC_FLOAT_STRATEGY),
        "end_sec": draw(_SEC_FLOAT_STRATEGY),
    }


# Feature: microdrama-generator, Property 2: timeline bounds correctness
@settings(max_examples=200)
@given(valid_scenes=st.lists(_valid_scene_strategy(), min_size=1, max_size=20))
def test_timeline_bounds_correctness(valid_scenes):
    """Property 2: Timeline bounds correctness.

    For any non-empty set of valid scene records (each with float `start_sec`
    and `end_sec`), `_compute_timeline_bounds` must return
    `(min(start_sec), max(end_sec))` across those records exactly.

    Validates: Requirements 2.4
    """
    earliest_start_sec, latest_end_sec = _PARTITION_GENERATOR._compute_timeline_bounds(valid_scenes)

    expected_earliest = min(scene["start_sec"] for scene in valid_scenes)
    expected_latest = max(scene["end_sec"] for scene in valid_scenes)

    # No arithmetic is performed (pure min/max selection over the same floats),
    # so exact equality must hold.
    assert earliest_start_sec == expected_earliest
    assert latest_end_sec == expected_latest


# ---------------------------------------------------------------------------
# Property test for prompt completeness: `_build_prompt` (Task 5.2)
#
# Property 3 concerns whether the prompt built for the Gemini_LLM contains
# every grounding signal: the plot text, each valid scene's
# scene_number/start_time/end_time/characters_present/setting/description, each
# dialogue line's speaker/text, the formatted Timeline_Bounds, and the content
# language token (Telugu).
#
# `_build_prompt` serializes the scene and dialogue payloads with
# `json.dumps(..., ensure_ascii=False)`, so JSON escaping is the only thing that
# could break a naive substring check. To verify completeness faithfully WITHOUT
# fighting JSON escaping, the generated text fields are drawn from "safe"
# alphabets that JSON never rewrites: ASCII letters/digits (plus space for prose
# fields and a few real Telugu letters), and crucially NO double-quote,
# backslash, or control characters. With `ensure_ascii=False`, such values are
# emitted verbatim, so `value in prompt` is a reliable completeness check.
#
# The `plot` is inserted into the prompt via an f-string (not JSON-encoded), so
# it may be arbitrary text — but it must be non-whitespace, because
# `_build_prompt` substitutes a fallback string when `plot.strip()` is empty.
# ---------------------------------------------------------------------------

# A handful of real Telugu letters (vowels + consonants) to exercise the
# Unicode path under `ensure_ascii=False`. These are non-ASCII letters that JSON
# leaves untouched, so substring checks remain reliable.
_TELUGU_LETTERS = "అఆఇఈఉఊఎఏకఖగఘచజటడణతదనపబమయరలవశషసహ"

# Token fields (characters_present entries, dialogue speaker): letters, digits,
# underscore, and Telugu — deliberately NO spaces, mirroring labels like
# "SPEAKER_03" and avoiding all-whitespace tokens.
_TOKEN_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_"
    + _TELUGU_LETTERS
)

# Prose fields (setting, description, dialogue text): same safe set plus spaces
# so multi-word values are exercised. Still free of quotes/backslashes/control
# characters, so JSON serialization is a no-op on these strings.
_TEXT_ALPHABET = _TOKEN_ALPHABET + " "

_prompt_token_strategy = st.text(alphabet=_TOKEN_ALPHABET, min_size=1, max_size=12)
_prompt_text_strategy = st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=30)

# Finite, non-NaN second values for scene/dialogue times. Bounded to keep the
# generated prompts small and fast across 100+ examples.
_PROMPT_SEC_STRATEGY = st.floats(
    min_value=0,
    max_value=360000,
    allow_nan=False,
    allow_infinity=False,
)

# Non-whitespace plot text. `st.text(min_size=1)` with a strip filter guarantees
# `_build_prompt` uses the plot verbatim rather than its empty-plot fallback.
_prompt_plot_strategy = st.text(min_size=1).filter(lambda s: s.strip())


@st.composite
def _prompt_valid_scene_strategy(draw):
    """Generate a single normalized ValidScene dict as `_build_prompt` expects.

    Carries float `start_sec`/`end_sec`, a (possibly empty) list of token-like
    characters, and prose `setting`/`description`. `scene_number` is assigned by
    index in `_prompt_valid_scenes_strategy` to keep numbers unique.
    """
    return {
        "start_sec": draw(_PROMPT_SEC_STRATEGY),
        "end_sec": draw(_PROMPT_SEC_STRATEGY),
        "characters_present": draw(st.lists(_prompt_token_strategy, max_size=4)),
        "setting": draw(_prompt_text_strategy),
        "description": draw(_prompt_text_strategy),
    }


@st.composite
def _prompt_valid_scenes_strategy(draw):
    """A NON-EMPTY list of ValidScene dicts with unique `scene_number`s."""
    scenes = draw(st.lists(_prompt_valid_scene_strategy(), min_size=1, max_size=6))
    for idx, scene in enumerate(scenes):
        scene["scene_number"] = idx
    return scenes


@st.composite
def _prompt_dialogue_line_strategy(draw):
    """A single Dialogue_Line dict with token `speaker` and prose `text`."""
    return {
        "speaker": draw(_prompt_token_strategy),
        "start": draw(_PROMPT_SEC_STRATEGY),
        "end": draw(_PROMPT_SEC_STRATEGY),
        "text": draw(_prompt_text_strategy),
    }


# Feature: microdrama-generator, Property 3: prompt completeness
@settings(max_examples=150, deadline=None)
@given(
    plot=_prompt_plot_strategy,
    valid_scenes=_prompt_valid_scenes_strategy(),
    # min_size=0 so both the scenes-only (empty) and with-dialogue branches are
    # exercised across examples.
    dialogues=st.lists(_prompt_dialogue_line_strategy(), min_size=0, max_size=6),
)
def test_prompt_completeness(plot, valid_scenes, dialogues):
    """Property 3: Prompt completeness.

    For any non-whitespace plot string, non-empty valid scene set, and dialogue
    set, `_build_prompt` must produce a prompt that contains:
      - the plot text verbatim,
      - the content-language token "Telugu",
      - the formatted Timeline_Bounds (earliest + latest),
      - for every valid scene: its scene_number, its formatted start_time and
        end_time, its setting, its description, and every character string,
      - when dialogue is present: every line's speaker and text (and when it is
        empty: the scenes-only note).

    Validates: Requirements 3.1, 3.2, 3.3, 3.4
    """
    bounds = _PARTITION_GENERATOR._compute_timeline_bounds(valid_scenes)
    prompt = _PARTITION_GENERATOR._build_prompt(plot, valid_scenes, dialogues, bounds)

    # --- Plot text (Req 3.1) ---
    # Non-whitespace plot is inserted verbatim via f-string (no JSON encoding).
    assert plot in prompt, "plot text missing from prompt"

    # --- Content language token (Req 3.4) ---
    assert "Telugu" in prompt, "content-language token 'Telugu' missing"

    # --- Formatted Timeline_Bounds (Req 3.4) ---
    earliest_sec, latest_sec = bounds
    earliest_fmt = MicrodramaGenerator._seconds_to_time(float(earliest_sec))
    latest_fmt = MicrodramaGenerator._seconds_to_time(float(latest_sec))
    assert earliest_fmt in prompt, f"earliest bound {earliest_fmt!r} missing"
    assert latest_fmt in prompt, f"latest bound {latest_fmt!r} missing"

    # --- Every valid scene's required fields (Req 3.2) ---
    for scene in valid_scenes:
        assert str(scene["scene_number"]) in prompt, (
            f"scene_number {scene['scene_number']} missing"
        )

        start_fmt = MicrodramaGenerator._seconds_to_time(float(scene["start_sec"]))
        end_fmt = MicrodramaGenerator._seconds_to_time(float(scene["end_sec"]))
        assert start_fmt in prompt, f"scene start_time {start_fmt!r} missing"
        assert end_fmt in prompt, f"scene end_time {end_fmt!r} missing"

        assert scene["setting"] in prompt, "scene setting missing"
        assert scene["description"] in prompt, "scene description missing"

        for character in scene["characters_present"]:
            assert character in prompt, f"character {character!r} missing"

    # --- Dialogue lines (Req 3.3) / scenes-only note (Req 2.3) ---
    if dialogues:
        for line in dialogues:
            assert line["speaker"] in prompt, f"dialogue speaker {line['speaker']!r} missing"
            assert line["text"] in prompt, f"dialogue text {line['text']!r} missing"
    else:
        assert "No diarized dialogue is available" in prompt, (
            "scenes-only note missing for empty dialogue set"
        )


# ---------------------------------------------------------------------------
# Unit tests for the static prompt instructions: `_build_prompt` (Task 5.3)
#
# These are EXAMPLE-BASED tests (NO hypothesis). Unlike Property 3 (which
# verifies the dynamic grounding signals are carried into the prompt), these
# tests pin the STATIC accuracy constraints baked into `_build_prompt`:
#   - Req 3.5: candidate boundaries must be selected ONLY from the provided
#     scene/dialogue times (no invented/rounded/interpolated timestamps),
#   - Req 3.6: every candidate runtime must fall within the inclusive 30-100s
#     Duration_Window, and
#   - Req 3.7: candidates must be ordered chronologically by start time.
#
# We build one prompt from a minimal-but-valid input (a short plot, two valid
# ValidScene dicts, a tiny dialogue list, and bounds from
# `_compute_timeline_bounds`) and assert on STABLE keywords/numbers that are
# actually present in the implementation ("ONLY", "Do NOT invent", "30", "100",
# "inclusive"/"between", "chronological") rather than whole sentences, so the
# tests stay robust to minor wording changes. `_PARTITION_GENERATOR` is reused
# because `_build_prompt` is pure (no filesystem / no LLM access).
#
# Requirements: 3.5, 3.6, 3.7
# ---------------------------------------------------------------------------


def _build_minimal_prompt():
    """Build a prompt from a minimal but valid `_build_prompt` input.

    Returns the generated prompt string. Two valid scenes (normalized
    ValidScene dicts with float `start_sec`/`end_sec`) and a single dialogue
    line provide just enough grounding data for the static instructions to be
    emitted; bounds are computed via `_compute_timeline_bounds` so they reflect
    the real min-start / max-end of the scenes.
    """
    valid_scenes = [
        {
            "scene_number": 1,
            "start_sec": 2.2,
            "end_sec": 40.0,
            "characters_present": ["HERO"],
            "setting": "Street",
            "description": "Opening confrontation.",
        },
        {
            "scene_number": 2,
            "start_sec": 40.0,
            "end_sec": 173.233,
            "characters_present": ["HERO", "VILLAIN"],
            "setting": "Rooftop",
            "description": "The chase escalates.",
        },
    ]
    dialogues = [
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 9.5, "text": "Telugu line one"},
    ]
    bounds = _PARTITION_GENERATOR._compute_timeline_bounds(valid_scenes)
    return _PARTITION_GENERATOR._build_prompt(
        "A tense thriller about a chase across the city.",
        valid_scenes,
        dialogues,
        bounds,
    )


def test_prompt_states_boundary_only_selection():
    """Req 3.5: prompt instructs boundary-only selection from provided times.

    The static rules must (a) constrain selection to ONLY the provided scene
    and dialogue times, and (b) forbid inventing/rounding/interpolating
    timestamps. We assert on the stable "ONLY" token, the words tying the
    selection to the scene and dialogue times, and the "Do NOT invent" phrasing.

    Validates: Requirements 3.5
    """
    prompt = _build_minimal_prompt()

    assert "ONLY" in prompt
    # Selection is tied to the provided scene/dialogue time fields.
    assert "scene" in prompt
    assert "dialogue" in prompt
    # No hallucinated timestamps: the "Do NOT invent" prohibition is present.
    assert "Do NOT invent" in prompt


def test_prompt_states_duration_window_30_to_100_inclusive():
    """Req 3.6: prompt states the inclusive 30-100 second Duration_Window.

    Both bound numbers (30 and 100) must appear, and the constraint must read
    as a range (a "between" phrasing) that is "inclusive".

    Validates: Requirements 3.6
    """
    prompt = _build_minimal_prompt()

    assert "30" in prompt
    assert "100" in prompt
    assert "between" in prompt
    assert "inclusive" in prompt


def test_prompt_states_chronological_ordering():
    """Req 3.7: prompt instructs chronological ordering by start time.

    The "chronological" keyword anchors the ordering instruction and is robust
    to minor wording changes around it.

    Validates: Requirements 3.7
    """
    prompt = _build_minimal_prompt()

    assert "chronological" in prompt


# ---------------------------------------------------------------------------
# Property test for the duration window invariant: `_enforce_candidate`
# (Task 7.2)
#
# Property 4 concerns the strict Duration_Window enforcement applied to every
# model-proposed RAW candidate. For each candidate `_enforce_candidate` must:
#   - recompute duration as (end_sec - start_sec),
#   - ACCEPT it only when 30 <= duration <= 100 (inclusive), emitting a
#     `duration_seconds` equal to that recomputed value, and
#   - EXCLUDE it (returning (None, reason) with a non-empty reason) whenever the
#     duration is strictly below 30s or strictly above 100s.
#
# We generate RAW candidates whose durations span all three regimes (below 30s,
# within [30, 100]s, and above 100s) by drawing a `start_sec` and a `duration`
# in [0, 200]. To ISOLATE the duration invariant from the timeline-bounds and
# ordering checks, the bounds are made deliberately generous (0 .. 1e9) and all
# candidate times are non-negative and far below the upper bound, so bounds can
# never cause an exclusion.
#
# Crispness (no flaky boundary flips): `_time_to_seconds` returns a float input
# UNCHANGED, so passing the raw `start_sec`/`end_sec` floats directly means the
# service computes `duration_seconds = end_sec - start_sec` with the EXACT same
# float arithmetic this test uses to classify each candidate. Classifying on the
# service-observed difference (rather than on the intended `duration`) means the
# 30.0/100.0 boundaries are compared identically on both sides, so rounding can
# never flip an accept/reject decision. A small tolerance still guards the
# emitted `duration_seconds` against its 3-decimal rounding.
# ---------------------------------------------------------------------------

# Tolerance for the emitted `duration_seconds`, which is `round(dur, 3)`. The
# worst-case rounding error is 0.0005s; 0.0015s also absorbs float noise.
_DURATION_WINDOW_TOLERANCE = 0.0015

# A timeline upper bound far above any generated candidate end time, so the
# bounds checks never trigger and only the duration invariant is exercised.
_GENEROUS_LATEST_BOUND = 1e9


@st.composite
def _raw_duration_candidate_strategy(draw):
    """Generate a single model-proposed RAW candidate (untrusted, pre-enforce).

    `start_sec` is a non-negative finite float and `duration` is drawn from
    [0, 200] so the resulting `end_sec = start_sec + duration` spans durations
    below 30s, within [30, 100]s, and above 100s. Times are passed as raw floats
    (which `_time_to_seconds` accepts unchanged), keeping the duration invariant
    crisp by avoiding any HH:MM:SS.mmm rounding. Creative fields are arbitrary;
    only the timestamps drive the duration enforcement under test.
    """
    start_sec = draw(
        st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
    )
    duration = draw(
        st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)
    )
    end_sec = start_sec + duration
    return {
        "title": "Candidate",
        "start_time": start_sec,
        "end_time": end_sec,
        "opening_hook": "hook",
        "central_conflict": "conflict",
        "cliffhanger_ending": "cliffhanger",
        "retention_score": 50,
    }


# Feature: microdrama-generator, Property 4: duration window invariant
@settings(max_examples=200, deadline=None)
@given(raw_candidates=st.lists(_raw_duration_candidate_strategy(), min_size=1, max_size=12))
def test_duration_window_invariant(raw_candidates):
    """Property 4: Duration window invariant.

    For any set of model-proposed candidates with arbitrary start/end times,
    `_enforce_candidate` must (with the timeline bounds made generous so they
    never cause exclusion):
      - EXCLUDE every candidate whose recomputed duration is strictly below 30s
        or strictly above 100s, returning (None, reason) with a non-empty
        reason, and
      - ACCEPT every candidate whose duration is within [30, 100]s, emitting a
        `duration_seconds` that equals (end_sec - start_sec) and lies within the
        inclusive [30, 100] window.

    Validates: Requirements 5.1, 5.2, 5.3, 5.4
    """
    # Generous bounds (all candidate times are >= 0 and far below 1e9) so the
    # timeline/ordering checks never exclude a candidate; coverage is irrelevant
    # to this property, so an empty valid-scene set is used.
    bounds = (0.0, _GENEROUS_LATEST_BOUND)
    valid_scenes = []

    min_dur = MicrodramaGenerator.MIN_DURATION
    max_dur = MicrodramaGenerator.MAX_DURATION

    for raw in raw_candidates:
        candidate, reason = _PARTITION_GENERATOR._enforce_candidate(raw, bounds, valid_scenes)

        start_sec = raw["start_time"]
        end_sec = raw["end_time"]
        # The service parses float times unchanged, so this is EXACTLY the
        # difference `_enforce_candidate` computes and compares against.
        service_duration = end_sec - start_sec

        if service_duration < min_dur or service_duration > max_dur:
            # Out-of-window candidates MUST be excluded with a non-empty reason
            # (Req 5.2, 5.3).
            assert candidate is None, (
                f"duration {service_duration} outside [{min_dur}, {max_dur}] was not excluded"
            )
            assert isinstance(reason, str) and reason, (
                "excluded candidate must carry a non-empty reason"
            )
        else:
            # In-window candidates MUST be accepted (Req 5.2, 5.3) ...
            assert reason is None, (
                f"in-window duration {service_duration} was excluded: {reason!r}"
            )
            assert candidate is not None

            emitted = candidate["duration_seconds"]

            # ... with the recomputed duration inside the inclusive window
            # (Req 5.4) ...
            assert min_dur - _DURATION_WINDOW_TOLERANCE <= emitted <= max_dur + _DURATION_WINDOW_TOLERANCE, (
                f"emitted duration_seconds {emitted} outside [{min_dur}, {max_dur}]"
            )

            # ... and equal to (end_time - start_time) within rounding tolerance
            # (Req 5.1).
            assert abs(emitted - (end_sec - start_sec)) <= _DURATION_WINDOW_TOLERANCE, (
                f"emitted duration_seconds {emitted} != recomputed {end_sec - start_sec}"
            )


# ---------------------------------------------------------------------------
# Property test for timestamp validity and exclusion with reason:
# `_enforce_candidate` (Task 7.3)
#
# Property 5 concerns the timestamp accuracy guarantee applied to every
# model-proposed RAW candidate. Against a Timeline_Bounds (earliest, latest)
# `_enforce_candidate` must:
#   - EXCLUDE (return (None, reason) with a non-empty reason) any candidate that
#     violates ordering (end <= start) OR starts before the earliest bound OR
#     ends after the latest bound, and
#   - for any candidate it ACCEPTS, emit start_time/end_time (in HH:MM:SS.mmm)
#     that satisfy start >= earliest, end <= latest, and end > start.
#
# Faithfulness to Property 5 (approach (a)): `_enforce_candidate` ALSO enforces
# the 30-100s duration window, so a candidate that passes ALL three timestamp
# checks may STILL be excluded for duration. To avoid entangling the duration
# rule we assert *conditionally*:
#   - timestamp-INVALID candidates MUST be excluded with a reason (the heart of
#     Req 6.4), and
#   - IF a candidate is accepted, its emitted bounds/ordering MUST hold
#     (Req 6.1, 6.2, 6.3),
# while making NO acceptance claim about timestamp-valid candidates (duration
# may legitimately drop them). This keeps every assertion a true consequence of
# Property 5.
#
# Crispness (no flaky boundary flips): times are passed as raw floats, which
# `_time_to_seconds` returns UNCHANGED, so the service compares the EXACT same
# float values this test uses to classify each candidate. The earliest/latest
# bounds are likewise the raw floats handed to the service, so the `start <
# earliest`, `end > latest`, and `end <= start` comparisons are identical on
# both sides and can never disagree at the boundary. A small tolerance guards
# only the accepted-candidate emitted times against the 3-decimal
# HH:MM:SS.mmm rounding.
# ---------------------------------------------------------------------------

# Tolerance for the ACCEPTED candidate's emitted timestamps, which are formatted
# to HH:MM:SS.mmm (worst-case rounding 0.0005s) then parsed back; 0.0015s also
# absorbs float noise. Only the accept-branch bound checks need it — the
# exclusion classification uses exact float comparisons.
_TS_BOUND_TOLERANCE = 0.0015


@st.composite
def _timestamp_candidate_strategy(draw, earliest, latest):
    """Generate a single model-proposed RAW candidate against given bounds.

    A `regime` is drawn so the candidate set deterministically spans every
    branch Property 5 cares about:
      - "valid":        ordered, strictly within [earliest, latest], duration in
                        [30, 100] -> should be ACCEPTED (exercises the emitted
                        bounds/ordering assertions),
      - "start_before": start < earliest -> excluded for the earliest-bound check,
      - "end_after":    end > latest    -> excluded for the latest-bound check,
      - "inverted":     end <= start    -> excluded for the ordering check,
      - "free":         start/end drawn independently across (earliest-500,
                        latest+500) -> an arbitrary mix of the above plus
                        duration-only exclusions.

    Times are emitted as raw floats so `_time_to_seconds` returns them unchanged,
    keeping the bounds/ordering comparisons crisp. Creative fields are arbitrary;
    only the timestamps drive the verification under test.
    """
    regime = draw(
        st.sampled_from(["valid", "start_before", "end_after", "inverted", "free"])
    )

    if regime == "valid":
        # span >= 50 (see `_bounds_and_timestamp_candidates_strategy`), so there
        # is always room for a 30-100s window strictly inside the bounds.
        max_room = latest - earliest
        max_dur = min(100.0, max_room - 0.05)
        duration = draw(
            st.floats(min_value=30.0, max_value=max_dur, allow_nan=False, allow_infinity=False)
        )
        start = draw(
            st.floats(
                min_value=earliest,
                max_value=latest - duration - 0.01,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        end = start + duration

    elif regime == "start_before":
        start = draw(
            st.floats(
                min_value=earliest - 1000.0,
                max_value=earliest - 0.001,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        duration = draw(
            st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False)
        )
        end = start + duration

    elif regime == "end_after":
        start = draw(
            st.floats(min_value=earliest, max_value=latest, allow_nan=False, allow_infinity=False)
        )
        over = draw(
            st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False)
        )
        end = latest + over

    elif regime == "inverted":
        start = draw(
            st.floats(min_value=earliest, max_value=latest, allow_nan=False, allow_infinity=False)
        )
        back = draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        )
        end = start - back  # end <= start (== when back is 0.0)

    else:  # "free"
        start = draw(
            st.floats(
                min_value=earliest - 500.0,
                max_value=latest + 500.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        end = draw(
            st.floats(
                min_value=earliest - 500.0,
                max_value=latest + 500.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )

    return {
        "title": "Candidate",
        "start_time": start,
        "end_time": end,
        "opening_hook": "hook",
        "central_conflict": "conflict",
        "cliffhanger_ending": "cliffhanger",
        "retention_score": 50,
    }


@st.composite
def _bounds_and_timestamp_candidates_strategy(draw):
    """Draw a Timeline_Bounds and a non-empty RAW candidate list against it.

    `earliest` in [0, 1000] and `latest = earliest + span` with span in
    [50, 5000] guarantee `earliest < latest` and leave room for in-bounds
    30-100s candidates. The candidates are drawn with the SAME bounds so the
    regime-based generation can target in-/out-of-bounds and inverted cases.
    """
    earliest = draw(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    span = draw(
        st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
    )
    latest = earliest + span
    candidates = draw(
        st.lists(_timestamp_candidate_strategy(earliest, latest), min_size=1, max_size=12)
    )
    return earliest, latest, candidates


# Feature: microdrama-generator, Property 5: timestamp validity and exclusion with reason
@settings(max_examples=200, deadline=None)
@given(data=_bounds_and_timestamp_candidates_strategy())
def test_timestamp_validity_and_exclusion_with_reason(data):
    """Property 5: Timestamp validity and exclusion with reason.

    For any set of model-proposed candidates and any Timeline_Bounds
    (earliest, latest) with earliest < latest, `_enforce_candidate` must:
      - EXCLUDE every candidate that violates ordering (end <= start) OR starts
        before `earliest` OR ends after `latest`, returning (None, reason) with
        a non-empty reason string (Req 6.4), and
      - for every ACCEPTED candidate, emit start_time/end_time that satisfy
        start >= earliest, end <= latest, and end > start (Req 6.1, 6.2, 6.3).

    To stay faithful to Property 5 without entangling the separately-enforced
    30-100s duration window, no acceptance claim is made about candidates that
    merely pass the timestamp checks (duration may still exclude them); the
    accept-branch assertions run only when a candidate is actually emitted.

    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    """
    earliest, latest, raw_candidates = data
    bounds = (earliest, latest)
    # Coverage derivation is irrelevant to this property, so no valid scenes.
    valid_scenes = []

    for raw in raw_candidates:
        candidate, reason = _PARTITION_GENERATOR._enforce_candidate(raw, bounds, valid_scenes)

        start_sec = raw["start_time"]
        end_sec = raw["end_time"]

        # The service parses these float times unchanged and compares them
        # against the same `earliest`/`latest`, so this classification matches
        # the service's own ordering/bounds checks exactly.
        timestamp_invalid = (
            end_sec <= start_sec  # ordering violation (Req 6.3)
            or start_sec < earliest  # before earliest bound (Req 6.1)
            or end_sec > latest  # after latest bound (Req 6.2)
        )

        if timestamp_invalid:
            # Every timestamp-invalid candidate MUST be excluded with a
            # non-empty reason (Req 6.4).
            assert candidate is None, (
                f"timestamp-invalid candidate (start={start_sec}, end={end_sec}, "
                f"bounds=({earliest}, {latest})) was not excluded"
            )
            assert isinstance(reason, str) and reason, (
                "excluded candidate must carry a non-empty reason"
            )

        if candidate is not None:
            # Every EMITTED candidate must satisfy bounds/ordering on its
            # formatted timestamps (Req 6.1, 6.2, 6.3). A small tolerance
            # absorbs the HH:MM:SS.mmm rounding on the emitted strings.
            emitted_start = MicrodramaGenerator._time_to_seconds(candidate["start_time"])
            emitted_end = MicrodramaGenerator._time_to_seconds(candidate["end_time"])

            assert emitted_start >= earliest - _TS_BOUND_TOLERANCE, (
                f"emitted start {emitted_start} before earliest bound {earliest}"
            )
            assert emitted_end <= latest + _TS_BOUND_TOLERANCE, (
                f"emitted end {emitted_end} after latest bound {latest}"
            )
            assert emitted_end > emitted_start, (
                f"emitted end {emitted_end} not after emitted start {emitted_start}"
            )


# ---------------------------------------------------------------------------
# Property test for coverage derivation: `_enforce_candidate` (Task 7.4)
#
# Property 7 concerns how an ACCEPTED candidate derives its coverage from the
# valid scene set. A valid scene (dict with float `start_sec`/`end_sec`,
# `scene_number`, `characters_present`) OVERLAPS the candidate range
# [cand_start, cand_end] iff `scene.start_sec < cand_end AND scene.end_sec >
# cand_start` (strict on both sides, so a scene touching the candidate only at
# an endpoint does NOT overlap). For an accepted candidate the service emits:
#   - `included_scene_numbers` = sorted, de-duplicated `scene_number`s of the
#     overlapping scenes (Req 7.5), and
#   - `characters_present`     = sorted, de-duplicated union of
#     `characters_present` across the overlapping scenes (Req 7.4).
#
# To make coverage observable we must force ACCEPTANCE, so the generated
# candidate is built to clear every other enforcement gate:
#   - duration drawn from [31, 99] (strictly inside the inclusive 30-100s window
#     with margin, so float noise on `cand_end - cand_start` can never flip the
#     window decision), giving `cand_end > cand_start` (ordering holds), and
#   - bounds set to (min(cand_start, all scene starts), max(cand_end, all scene
#     ends)) so `earliest <= cand_start` and `cand_end <= latest` always hold and
#     the timeline checks never exclude the candidate.
#
# Crispness (no flaky boundary flips): the raw candidate carries `start_time`/
# `end_time` as RAW FLOATS, which `_time_to_seconds` returns UNCHANGED, so the
# service computes overlap on exactly `cand_start`/`cand_end`. The test computes
# the expected overlap with the SAME strict `<`/`>` comparisons against the same
# floats, so endpoint-touching scenes are classified identically on both sides.
#
# Scene times are positioned around the candidate (starts drawn from
# [cand_start - 150, cand_end + 150]) so examples naturally span: empty coverage
# (no overlap), single and multiple overlaps, shared vs distinct characters, and
# empty character lists. `scene_number` is assigned by index for uniqueness.
# ---------------------------------------------------------------------------

# Character tokens: uppercase letters only, mirroring labels like "HERO" and
# keeping union/sort comparisons unambiguous. Lists may be empty (max_size=4,
# no min) so the empty-characters edge is exercised.
_coverage_char_strategy = st.text(
    alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=8
)


@st.composite
def _coverage_case_strategy(draw):
    """Draw an ACCEPTED candidate plus a valid-scene set positioned around it.

    Returns (raw_candidate, bounds, valid_scenes, cand_start, cand_end) where:
      - cand_start in [0, 1000] and duration in [31, 99] -> cand_end =
        cand_start + duration is strictly after cand_start and the duration sits
        well inside the inclusive 30-100s window (so the candidate is accepted),
      - each valid scene has float `start_sec` (drawn from [cand_start - 150,
        cand_end + 150]) and `end_sec = start_sec + scene_dur` (scene_dur >= 0),
        a unique `scene_number` (its index), and a possibly-empty character list,
      - bounds = (min(cand_start, scene starts), max(cand_end, scene ends)) so
        the timeline checks never exclude the candidate.
    """
    cand_start = draw(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    duration = draw(
        st.floats(min_value=31.0, max_value=99.0, allow_nan=False, allow_infinity=False)
    )
    cand_end = cand_start + duration

    n_scenes = draw(st.integers(min_value=0, max_value=8))
    valid_scenes = []
    for idx in range(n_scenes):
        scene_start = draw(
            st.floats(
                min_value=cand_start - 150.0,
                max_value=cand_end + 150.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        scene_dur = draw(
            st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)
        )
        valid_scenes.append({
            "scene_number": idx,  # unique by construction
            "start_sec": scene_start,
            "end_sec": scene_start + scene_dur,  # end_sec >= start_sec
            "characters_present": draw(st.lists(_coverage_char_strategy, max_size=4)),
        })

    # Bounds that always contain the candidate: earliest <= cand_start and
    # latest >= cand_end (cand_start/cand_end are included in the min/max).
    earliest = min([cand_start] + [s["start_sec"] for s in valid_scenes])
    latest = max([cand_end] + [s["end_sec"] for s in valid_scenes])
    bounds = (earliest, latest)

    raw_candidate = {
        "title": "Candidate",
        "start_time": cand_start,  # raw float -> parsed unchanged
        "end_time": cand_end,      # raw float -> parsed unchanged
        "opening_hook": "hook",
        "central_conflict": "conflict",
        "cliffhanger_ending": "cliffhanger",
        "retention_score": 50,
    }
    return raw_candidate, bounds, valid_scenes, cand_start, cand_end


# Feature: microdrama-generator, Property 7: coverage derivation for characters and scene numbers
@settings(max_examples=200, deadline=None)
@given(case=_coverage_case_strategy())
def test_coverage_derivation_for_characters_and_scene_numbers(case):
    """Property 7: Coverage derivation for characters and scene numbers.

    For any valid scene set and any ACCEPTED candidate time range
    [cand_start, cand_end], `_enforce_candidate` must derive coverage from
    exactly the valid scenes whose range OVERLAPS the candidate range under the
    strict rule `scene.start_sec < cand_end AND scene.end_sec > cand_start`:
      - `included_scene_numbers` equals the sorted, de-duplicated `scene_number`
        values of those overlapping scenes (Req 7.5), and
      - `characters_present` equals the sorted, de-duplicated union of
        `characters_present` across those overlapping scenes (Req 7.4).

    Endpoint-touching scenes (`scene.end_sec == cand_start` or
    `scene.start_sec == cand_end`) do NOT overlap under the strict comparison;
    the expected computation uses the same strict comparison so the boundary is
    handled identically.

    Validates: Requirements 7.4, 7.5
    """
    raw_candidate, bounds, valid_scenes, cand_start, cand_end = case

    # Compute the expected coverage independently, using the SAME strict overlap
    # comparison the service applies to the parsed (here: raw float) times.
    overlapping = [
        scene
        for scene in valid_scenes
        if scene["start_sec"] < cand_end and scene["end_sec"] > cand_start
    ]
    expected_scene_numbers = sorted({scene["scene_number"] for scene in overlapping})
    expected_characters = sorted(
        {c for scene in overlapping for c in scene["characters_present"]}
    )

    candidate, reason = _PARTITION_GENERATOR._enforce_candidate(
        raw_candidate, bounds, valid_scenes
    )

    # The candidate is constructed to clear duration/ordering/bounds, so it must
    # be ACCEPTED for the derived coverage to be observable.
    assert reason is None, f"candidate unexpectedly excluded: {reason!r}"
    assert candidate is not None, "candidate was excluded but coverage was expected"

    # Coverage must match the independently-computed overlap exactly.
    assert candidate["included_scene_numbers"] == expected_scene_numbers, (
        f"included_scene_numbers {candidate['included_scene_numbers']} != "
        f"expected {expected_scene_numbers}"
    )
    assert candidate["characters_present"] == expected_characters, (
        f"characters_present {candidate['characters_present']} != "
        f"expected {expected_characters}"
    )


# ---------------------------------------------------------------------------
# Property test for output structure completeness: `generate` (Task 8.3)
#
# Property 8 concerns the SHAPE of a successful generation rather than its
# numeric accuracy. With the Gemini_LLM MOCKED (so no network call happens),
# `generate` must, for ANY successful run, return an envelope that carries
# `video_id`, a `status` field, and a list `microdrama_candidates`; and EVERY
# emitted candidate must carry `title`, `start_time`, `end_time`,
# `duration_seconds`, an opening hook, a central conflict, a cliffhanger ending,
# and a `retention_score` within the inclusive 0..100 range.
#
# Mocking strategy (per Task 8.3): `@given` runs many iterations, so the LLM is
# mocked INSIDE the test body by assigning an instance-level `_invoke_llm` that
# returns `(raw_candidates, None)` — a hypothesis-generated list of RAW candidate
# dicts. As an instance attribute it shadows the bound method and, being a plain
# lambda, is invoked as `self._invoke_llm(prompt)` -> `lambda(prompt)` (no `self`).
#
# Performance note: constructing a `MicrodramaGenerator` initializes the Vertex
# AI `genai.Client`, which performs an expensive credential lookup (~12s each
# when ADC is absent). Building one PER example would make 100 iterations take
# many minutes, so the module-level `_PARTITION_GENERATOR` (already constructed
# once at import) is REUSED. Its `output_base_dir` and `_invoke_llm` are set for
# the example and restored in a `finally` block so no shared state leaks to
# other tests. `_invoke_llm` is the only seam the LLM hides behind, so reusing
# the instance is equivalent to a fresh one for this property.
#
# Setup per example: a FRESH temp dir (created with `tempfile.mkdtemp`, removed
# in `finally`) holds a small valid `scenes_and_plot.json` whose two scenes span
# the timeline [0, 100000]s and a minimal `dialogue_diarization.json`
# (`{"dialogues": []}`, scenes-only mode). RAW candidates are generated with
# `start_time`/`end_time` as raw floats inside that timeline and durations in
# [0, 150]s, so some clear the 30-100s window (accepted) and some are excluded —
# the property asserts STRUCTURE, not that every candidate is accepted.
# Generating `retention_score` outside [0, 100] also exercises the score clamp.
# ---------------------------------------------------------------------------
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

# Two scenes spanning the timeline [0, 100000]s. Times are written as raw floats
# (which `_time_to_seconds` returns unchanged), so the computed Timeline_Bounds
# are exactly (0.0, 100000.0) and generated candidates can sit well inside them.
_STRUCTURE_SCENES_INPUT = {
    "total_scenes": 2,
    "Scenes": [
        {
            "scene_number": 1,
            "start_time": 0.0,
            "end_time": 50000.0,
            "characters_present": ["HERO"],
            "setting": "Street",
            "description": "Opening confrontation.",
        },
        {
            "scene_number": 2,
            "start_time": 50000.0,
            "end_time": 100000.0,
            "characters_present": ["HERO", "VILLAIN"],
            "setting": "Rooftop",
            "description": "The climactic chase.",
        },
    ],
    "plot_of_the_movie": "A tense thriller used as fixed grounding data.",
}

# Minimal dialogue input: a present `dialogues` key with an empty list exercises
# the tolerated scenes-only mode (Req 2.3) without affecting output structure.
_STRUCTURE_DIALOGUE_INPUT = {"dialogues": []}


@st.composite
def _structure_raw_candidate_strategy(draw):
    """Generate one model-proposed RAW candidate (untrusted, pre-enforcement).

    `start_time`/`end_time` are raw floats inside the [0, 100000]s timeline:
    `start` in [0, 99000] and `duration` in [0, 150] give `end = start +
    duration <= 99150`, so every candidate is comfortably in-bounds and only the
    30-100s duration window decides acceptance (a healthy mix of accepted and
    excluded across a list). `retention_score` is drawn across [-50, 200] —
    including values OUTSIDE [0, 100] — to exercise the service's score clamp.
    Creative fields are arbitrary text (carried through untouched).
    """
    start_sec = draw(
        st.floats(min_value=0.0, max_value=99000.0, allow_nan=False, allow_infinity=False)
    )
    duration = draw(
        st.floats(min_value=0.0, max_value=150.0, allow_nan=False, allow_infinity=False)
    )
    end_sec = start_sec + duration
    return {
        "title": draw(st.text(max_size=20)),
        "start_time": start_sec,
        "end_time": end_sec,
        "opening_hook": draw(st.text(max_size=20)),
        "central_conflict": draw(st.text(max_size=20)),
        "cliffhanger_ending": draw(st.text(max_size=20)),
        "retention_score": draw(st.integers(min_value=-50, max_value=200)),
    }


# Keys every EMITTED Microdrama_Candidate must carry (Req 7.1, 7.2; the opening
# hook / central conflict / cliffhanger ending fields, plus retention_score).
_REQUIRED_CANDIDATE_KEYS = (
    "title",
    "start_time",
    "end_time",
    "duration_seconds",
    "opening_hook",
    "central_conflict",
    "cliffhanger_ending",
    "retention_score",
)


# Feature: microdrama-generator, Property 8: output structure completeness
@settings(max_examples=100, deadline=None)
@given(raw_candidates=st.lists(_structure_raw_candidate_strategy(), min_size=0, max_size=8))
def test_output_structure_completeness(raw_candidates):
    """Property 8: Output structure completeness.

    For any successful generation (LLM mocked to return a list of RAW
    candidates), the Candidates_Output envelope must contain `video_id`, a
    `status` field, and a list `microdrama_candidates`; and EVERY emitted
    candidate must contain `title`, `start_time`, `end_time`,
    `duration_seconds`, an opening hook, a central conflict, a cliffhanger
    ending, and a `retention_score` within the inclusive range 0..100. The
    same structure must hold for the persisted `microdrama_candidates.json`.

    Validates: Requirements 7.1, 7.2, 7.3, 8.2
    """
    # Reuse the already-constructed generator (avoids the ~12s client init per
    # example); save the bits we mutate so we can restore them in `finally`.
    gen = _PARTITION_GENERATOR
    tmp_dir = tempfile.mkdtemp(prefix="pbt_structure_")
    original_base = gen.output_base_dir
    had_instance_invoke = "_invoke_llm" in gen.__dict__
    try:
        # Write the two valid input files into the fresh temp dir.
        (Path(tmp_dir) / "scenes_and_plot.json").write_text(
            json.dumps(_STRUCTURE_SCENES_INPUT), encoding="utf-8"
        )
        (Path(tmp_dir) / "dialogue_diarization.json").write_text(
            json.dumps(_STRUCTURE_DIALOGUE_INPUT), encoding="utf-8"
        )

        # Point the generator at the temp dir and MOCK the LLM so no network
        # call happens: `_invoke_llm` returns the generated RAW candidates.
        gen.output_base_dir = Path(tmp_dir)
        gen._invoke_llm = lambda prompt: (raw_candidates, None)

        envelope = gen.generate("VID")

        # --- Envelope-level structure (Req 8.2) ---
        assert "video_id" in envelope, "envelope missing video_id"
        assert "status" in envelope, "envelope missing status"
        assert "microdrama_candidates" in envelope, "envelope missing microdrama_candidates"
        assert isinstance(envelope["microdrama_candidates"], list), (
            "microdrama_candidates is not a list"
        )
        # A successful generation reports a completed status (Req 8.2 / 8.4).
        assert envelope["status"] == "completed", (
            f"status was {envelope['status']!r}, expected 'completed'"
        )

        # --- Persisted file mirrors the returned envelope (Req 8.1, 8.2) ---
        output_file = Path(tmp_dir) / "microdrama_candidates.json"
        assert output_file.exists(), "microdrama_candidates.json was not written"
        written = json.loads(output_file.read_text(encoding="utf-8"))
        assert "video_id" in written and "status" in written, (
            "written file missing video_id/status"
        )
        assert isinstance(written.get("microdrama_candidates"), list), (
            "written microdrama_candidates is not a list"
        )
        assert len(written["microdrama_candidates"]) == len(
            envelope["microdrama_candidates"]
        ), "written candidate count differs from returned envelope"

        # --- Per-candidate completeness (Req 7.1, 7.2, 7.3) ---
        for candidate in envelope["microdrama_candidates"]:
            for key in _REQUIRED_CANDIDATE_KEYS:
                assert key in candidate, f"emitted candidate missing {key!r}"

            score = candidate["retention_score"]
            # bool is an int subclass; a retention score must be a real number.
            assert isinstance(score, (int, float)) and not isinstance(score, bool), (
                f"retention_score {score!r} is not numeric"
            )
            assert 0 <= score <= 100, (
                f"retention_score {score} outside the inclusive 0..100 range"
            )
    finally:
        # Restore the shared generator's state so no leak reaches other tests.
        gen.output_base_dir = original_base
        if not had_instance_invoke:
            gen.__dict__.pop("_invoke_llm", None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Property test for full-exclusion terminal state: `generate` (Task 8.4)
#
# Property 9 concerns the TERMINAL STATE of a generation in which EVERY
# model-proposed candidate is dropped by enforcement. With the Gemini_LLM
# MOCKED to return a non-empty list of RAW candidates that ALL violate the
# duration or timestamp checks, `generate` must still complete: the
# Candidates_Output envelope has `status == "completed"` and an EMPTY
# `microdrama_candidates` list (Req 8.4) — full exclusion is a successful,
# not a failed, outcome.
#
# Guaranteed full exclusion BY CONSTRUCTION: the fixed inputs
# (`_STRUCTURE_SCENES_INPUT`) make the Timeline_Bounds exactly (0.0, 100000.0),
# so each generated candidate is drawn from one of five regimes that each
# violate a specific enforcement gate against those bounds:
#   - "too_short":    duration in [0, 29.9] (strictly < 30s), in-bounds and
#                     ordered -> excluded by the duration MINIMUM (or, when
#                     duration is 0.0, by the ordering check),
#   - "too_long":     duration in [100.1, 50000]s with end <= 99000 (in-bounds,
#                     ordered) -> excluded by the duration MAXIMUM,
#   - "inverted":     end <= start -> excluded by the ordering check,
#   - "start_before": start < 0.0 (before the earliest bound) -> excluded by the
#                     earliest-bound check,
#   - "end_after":    end = 100000 + over (after the latest bound) -> excluded by
#                     the latest-bound check.
#
# Crispness (no flaky boundary flips): times are passed as RAW FLOATS, which
# `_time_to_seconds` returns UNCHANGED, so the service compares exactly the
# values generated here against the same (0.0, 100000.0) bounds. Every regime
# keeps a comfortable margin from the 30.0/100.0/0.0/100000.0 boundaries (>= 0.1s
# for durations, >= 0.001s for bounds), far larger than any float noise on
# `(start + duration) - start`, so no candidate can accidentally be accepted.
# A valid 30-100s in-bounds candidate is therefore NEVER generated.
#
# Mocking/performance: identical to Task 8.3 — the module-level
# `_PARTITION_GENERATOR` is REUSED (its construction pays the ~12s genai.Client
# credential lookup ONCE), with `output_base_dir`/`_invoke_llm` set for the
# example and restored in `finally`, and a fresh `tempfile.mkdtemp` dir holding
# the same fixed `scenes_and_plot.json` / `dialogue_diarization.json`.
# ---------------------------------------------------------------------------


@st.composite
def _fully_excluded_raw_candidate_strategy(draw):
    """Generate one model-proposed RAW candidate guaranteed to be EXCLUDED.

    A `regime` is sampled so the candidate list spans every exclusion gate
    against the fixed Timeline_Bounds (0.0, 100000.0): too-short duration,
    too-long duration, inverted ordering, start-before-earliest, and
    end-after-latest. Times are emitted as raw floats (returned unchanged by
    `_time_to_seconds`) with wide margins from the enforcement boundaries, so
    `_enforce_candidate` must reject every one. Creative fields are arbitrary.
    """
    regime = draw(
        st.sampled_from(
            ["too_short", "too_long", "inverted", "start_before", "end_after"]
        )
    )

    if regime == "too_short":
        # In-bounds, ordered, but duration strictly below the 30s minimum
        # (0.0 collapses to an ordering violation — still excluded).
        start = draw(
            st.floats(min_value=0.0, max_value=99000.0, allow_nan=False, allow_infinity=False)
        )
        duration = draw(
            st.floats(min_value=0.0, max_value=29.9, allow_nan=False, allow_infinity=False)
        )
        end = start + duration  # end <= 99029.9 < 100000 -> in-bounds

    elif regime == "too_long":
        # In-bounds, ordered, but duration strictly above the 100s maximum.
        start = draw(
            st.floats(min_value=0.0, max_value=49000.0, allow_nan=False, allow_infinity=False)
        )
        duration = draw(
            st.floats(min_value=100.1, max_value=50000.0, allow_nan=False, allow_infinity=False)
        )
        end = start + duration  # end <= 99000 < 100000 -> in-bounds

    elif regime == "inverted":
        # end <= start -> ordering violation (== when back is 0.0).
        start = draw(
            st.floats(min_value=100.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
        )
        back = draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        )
        end = start - back

    elif regime == "start_before":
        # start strictly below the earliest bound (0.0); duration keeps it
        # ordered so the earliest-bound check is the gate that excludes it.
        start = draw(
            st.floats(min_value=-1000.0, max_value=-0.001, allow_nan=False, allow_infinity=False)
        )
        duration = draw(
            st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False)
        )
        end = start + duration

    else:  # "end_after"
        # end strictly above the latest bound (100000.0); start stays in-bounds
        # and ordered so the latest-bound check is the gate that excludes it.
        start = draw(
            st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
        )
        over = draw(
            st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False)
        )
        end = 100000.0 + over

    return {
        "title": draw(st.text(max_size=20)),
        "start_time": start,
        "end_time": end,
        "opening_hook": "hook",
        "central_conflict": "conflict",
        "cliffhanger_ending": "cliffhanger",
        "retention_score": 50,
    }


# Feature: microdrama-generator, Property 9: full-exclusion terminal state
@settings(max_examples=100, deadline=None)
@given(
    raw_candidates=st.lists(
        _fully_excluded_raw_candidate_strategy(), min_size=1, max_size=10
    )
)
def test_full_exclusion_terminal_state(raw_candidates):
    """Property 9: Full-exclusion terminal state.

    For any NON-EMPTY set of model-proposed candidates in which every candidate
    violates the duration or timestamp checks (LLM mocked to return them), the
    Candidates_Output must report `status == "completed"` with an EMPTY
    `microdrama_candidates` list — full exclusion is a successful terminal
    state, not a failure. Every excluded candidate is recorded with a reason in
    `excluded_candidates`.

    Validates: Requirements 8.4
    """
    # Reuse the already-constructed generator (avoids the ~12s client init per
    # example); save the bits we mutate so we can restore them in `finally`.
    gen = _PARTITION_GENERATOR
    tmp_dir = tempfile.mkdtemp(prefix="pbt_full_exclusion_")
    original_base = gen.output_base_dir
    had_instance_invoke = "_invoke_llm" in gen.__dict__
    try:
        # Write the two valid input files into the fresh temp dir. These fix the
        # Timeline_Bounds at (0.0, 100000.0) that the regimes are built against.
        (Path(tmp_dir) / "scenes_and_plot.json").write_text(
            json.dumps(_STRUCTURE_SCENES_INPUT), encoding="utf-8"
        )
        (Path(tmp_dir) / "dialogue_diarization.json").write_text(
            json.dumps(_STRUCTURE_DIALOGUE_INPUT), encoding="utf-8"
        )

        # Point the generator at the temp dir and MOCK the LLM so no network
        # call happens: `_invoke_llm` returns the generated RAW candidates that
        # are all guaranteed to be excluded by enforcement.
        gen.output_base_dir = Path(tmp_dir)
        gen._invoke_llm = lambda prompt: (raw_candidates, None)

        envelope = gen.generate("VID")

        # Full exclusion is a SUCCESSFUL terminal state (Req 8.4).
        assert envelope["status"] == "completed", (
            f"status was {envelope['status']!r}, expected 'completed'"
        )
        # No candidate survived enforcement -> the list is empty.
        assert envelope["microdrama_candidates"] == [], (
            f"expected no surviving candidates, got {envelope['microdrama_candidates']!r}"
        )
        # Every raw candidate was excluded and recorded with a reason.
        assert len(envelope["excluded_candidates"]) == len(raw_candidates), (
            f"expected {len(raw_candidates)} excluded candidates, "
            f"got {len(envelope['excluded_candidates'])}"
        )
    finally:
        # Restore the shared generator's state so no leak reaches other tests.
        gen.output_base_dir = original_base
        if not had_instance_invoke:
            gen.__dict__.pop("_invoke_llm", None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Unit tests for LLM invocation and persistence: `_invoke_llm` / `generate`
# (Task 8.5)
#
# These are EXAMPLE-BASED tests (NO hypothesis). They pin the invocation
# contract of `_invoke_llm` and the no-LLM-call / persistence behavior of
# `generate`. The Gemini client is replaced with a `unittest.mock.MagicMock`
# (or the `_invoke_llm` seam is monkeypatched with a plain lambda) so NO network
# call ever happens. pytest's `tmp_path` fixture isolates every filesystem
# touch; `_make_generator(tmp_path)` builds the service rooted there and we then
# OVERRIDE `gen.llm_client` (so absent GCP creds are irrelevant).
#
# Coverage:
#   - Req 4.2: `_invoke_llm` calls `generate_content` with
#     `response_mime_type="application/json"` and returns the parsed list.
#   - Req 4.3: client unavailable (`llm_client is None`) -> error, no call.
#   - Req 4.4: `generate_content` raises -> error carrying the failure detail.
#   - Req 4.5: non-JSON response text -> parse error.
#   - Req 1.3 / 1.4 / 1.5 / 2.3: `generate` no-LLM-call paths (missing scenes
#     file, missing dialogue FILE, invalid-JSON scenes) each fail with a named
#     error and NEVER invoke the mocked client. The missing dialogue FILE case
#     (failure, Req 1.4) is deliberately distinct from a present-but-keyless
#     dialogue input (tolerated scenes-only mode, Req 2.3) used by the
#     persistence test below.
#   - Req 8.1: a successful run writes a parseable `microdrama_candidates.json`
#     at the expected path whose content equals the returned envelope.
#
# Requirements: 1.3, 1.4, 1.5, 2.3, 4.2, 4.3, 4.4, 4.5, 8.1
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock  # noqa: E402


def _valid_scenes_input_8_5():
    """A small but valid Scenes_Input whose two scenes span a wide timeline.

    The two scenes cover [0, 100000]s, so `_compute_timeline_bounds` yields
    exactly (0.0, 100000.0) and a 60s candidate sits comfortably in-bounds.
    """
    return {
        "total_scenes": 2,
        "Scenes": [
            {
                "scene_number": 1,
                "start_time": 0.0,
                "end_time": 50000.0,
                "characters_present": ["HERO"],
                "setting": "Street",
                "description": "Opening confrontation.",
            },
            {
                "scene_number": 2,
                "start_time": 50000.0,
                "end_time": 100000.0,
                "characters_present": ["HERO", "VILLAIN"],
                "setting": "Rooftop",
                "description": "The climactic chase.",
            },
        ],
        "plot_of_the_movie": "A tense thriller used as fixed grounding data.",
    }


def _valid_dialogue_input_8_5():
    """A minimal but valid Dialogue_Input with one diarized Telugu line."""
    return {
        "dialogues": [
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 9.5, "text": "Telugu line"}
        ]
    }


def test_invoke_llm_requests_json_mime_type_and_returns_parsed_list(tmp_path):
    """Req 4.2: `_invoke_llm` requests application/json and returns the list.

    With the client mocked to return a `.text` holding a JSON array, the method
    must (a) return the parsed list with no error, (b) call
    `models.generate_content` exactly once, and (c) pass a `config` whose
    `response_mime_type` is `"application/json"`.

    Validates: Requirements 4.2
    """
    generator = _make_generator(tmp_path)

    raw_list = [
        {
            "title": "Candidate A",
            "start_time": "00:00:00.000",
            "end_time": "00:01:00.000",
            "opening_hook": "hook",
            "central_conflict": "conflict",
            "cliffhanger_ending": "cliffhanger",
            "retention_score": 80,
        }
    ]
    fake_response = MagicMock()
    fake_response.text = json.dumps(raw_list)

    generator.llm_client = MagicMock()
    generator.llm_client.models.generate_content.return_value = fake_response

    result, error = generator._invoke_llm("prompt text")

    # The parsed array is returned verbatim with no error.
    assert error is None
    assert result == raw_list

    # generate_content was called exactly once.
    generator.llm_client.models.generate_content.assert_called_once()

    # The JSON response mode was requested (config passed as a keyword).
    call_args = generator.llm_client.models.generate_content.call_args
    config = call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"


def test_invoke_llm_client_unavailable_returns_error_and_no_call(tmp_path):
    """Req 4.3: a None client yields a client-unavailable error without raising.

    Validates: Requirements 4.3
    """
    generator = _make_generator(tmp_path)
    generator.llm_client = None

    # Must not raise.
    result, error = generator._invoke_llm("prompt text")

    assert result is None
    assert isinstance(error, str) and error
    assert "unavailable" in error.lower()


def test_invoke_llm_call_raises_returns_error_with_detail(tmp_path):
    """Req 4.4: when generate_content raises, the failure detail is surfaced.

    Validates: Requirements 4.4
    """
    generator = _make_generator(tmp_path)
    generator.llm_client = MagicMock()
    generator.llm_client.models.generate_content.side_effect = Exception("boom")

    result, error = generator._invoke_llm("prompt text")

    assert result is None
    assert isinstance(error, str) and error
    # The original failure detail is included in the message.
    assert "boom" in error


def test_invoke_llm_non_json_response_returns_parse_error(tmp_path):
    """Req 4.5: a response whose text is not valid JSON yields a parse error.

    Validates: Requirements 4.5
    """
    generator = _make_generator(tmp_path)
    fake_response = MagicMock()
    fake_response.text = "not json {"

    generator.llm_client = MagicMock()
    generator.llm_client.models.generate_content.return_value = fake_response

    result, error = generator._invoke_llm("prompt text")

    assert result is None
    assert isinstance(error, str) and error
    # Message indicates the response could not be parsed.
    assert "pars" in error.lower()  # matches "parsed" / "parse"


def test_generate_missing_scenes_file_fails_without_llm_call(tmp_path):
    """Req 1.3: a missing scenes file fails naming the file and never calls LLM.

    Validates: Requirements 1.3
    """
    generator = _make_generator(tmp_path)  # empty dir: no scenes file
    generator.llm_client = MagicMock()

    result = generator.generate("VID")

    assert result["status"] == "failed"
    assert SCENES_FILENAME in result["error"]
    generator.llm_client.models.generate_content.assert_not_called()


def test_generate_missing_dialogue_file_fails_without_llm_call(tmp_path):
    """Req 1.4: scenes present but dialogue FILE absent fails naming the dialogue
    file and never calls the LLM.

    This is the failure case (missing FILE) — distinct from a present dialogue
    input that merely lacks a `dialogues` key, which is tolerated as scenes-only
    mode (Req 2.3) and exercised by the persistence test.

    Validates: Requirements 1.4
    """
    generator = _make_generator(tmp_path)
    (tmp_path / SCENES_FILENAME).write_text(
        json.dumps(_valid_scenes_input_8_5()), encoding="utf-8"
    )
    # No dialogue file is written.
    generator.llm_client = MagicMock()

    result = generator.generate("VID")

    assert result["status"] == "failed"
    assert DIALOGUE_FILENAME in result["error"]
    generator.llm_client.models.generate_content.assert_not_called()


def test_generate_invalid_json_scenes_fails_without_llm_call(tmp_path):
    """Req 1.5: a malformed scenes file fails naming the file and never calls LLM.

    Validates: Requirements 1.5
    """
    generator = _make_generator(tmp_path)
    (tmp_path / SCENES_FILENAME).write_text("{ this is not valid json ", encoding="utf-8")
    generator.llm_client = MagicMock()

    result = generator.generate("VID")

    assert result["status"] == "failed"
    assert SCENES_FILENAME in result["error"]
    generator.llm_client.models.generate_content.assert_not_called()


def test_generate_persists_candidates_output(tmp_path):
    """Req 8.1: a successful run writes a parseable microdrama_candidates.json at
    the expected path whose content equals the returned envelope.

    The `_invoke_llm` seam is monkeypatched with a plain lambda returning one
    valid 60s in-bounds raw candidate, so no network call happens and at least
    one candidate survives enforcement.

    Validates: Requirements 8.1
    """
    generator = _make_generator(tmp_path)
    (tmp_path / SCENES_FILENAME).write_text(
        json.dumps(_valid_scenes_input_8_5()), encoding="utf-8"
    )
    (tmp_path / DIALOGUE_FILENAME).write_text(
        json.dumps(_valid_dialogue_input_8_5()), encoding="utf-8"
    )

    # One valid candidate: 0..60s -> 60s duration (within 30-100s) and strictly
    # inside the [0, 100000]s Timeline_Bounds, so it is accepted. Times are raw
    # floats, which `_time_to_seconds` returns unchanged.
    raw_candidate = {
        "title": "Opening Standoff",
        "start_time": 0.0,
        "end_time": 60.0,
        "opening_hook": "A gun is drawn in a crowded street.",
        "central_conflict": "The hero is cornered by the villain.",
        "cliffhanger_ending": "A shot rings out as the screen cuts to black.",
        "retention_score": 90,
    }
    # Monkeypatch the LLM seam (instance attribute shadows the bound method; the
    # plain lambda is called as `self._invoke_llm(prompt)` -> lambda(prompt)).
    generator._invoke_llm = lambda prompt: ([raw_candidate], None)

    result = generator.generate("VID")

    # Successful terminal state with at least one emitted candidate.
    assert result["status"] == "completed"
    assert len(result["microdrama_candidates"]) >= 1

    # The output file exists at the expected path.
    output_path = tmp_path / "microdrama_candidates.json"
    assert output_path.exists()

    # Its parsed content equals the returned envelope exactly.
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == result
