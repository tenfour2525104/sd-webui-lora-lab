# -*- coding: utf-8 -*-
"""
Lora Lab — multi-Lora verification helper for AUTOMATIC1111 / Forge (Classic & Neo).

All UI text uses the spelling "Lora".

What it does
------------
Adds an always-visible accordion ("Lora Lab") to the txt2img and img2img tabs
with 10 Lora slots.  Each slot has:

  * an on/off checkbox
  * a trigger-word checkbox (default ON): when on, the Lora's trigger words
    (from its .civitai.info "trainedWords", a JSON sidecar, or the
    safetensors "activation text" metadata) are added to the front of the
    prompt at generation time — words already in the prompt are not doubled
  * a searchable dropdown of the Lora files the WebUI can see
    (you can also type a name by hand if a Lora is not listed)
  * a strength slider (-2.00 .. +2.00) with -0.1 / +0.1 step buttons

Slots are separated by thin divider lines (no per-slot thumbnails).

At the top of the accordion, a "reflect to prompt" button writes the current
slots into the prompt box directly: each Lora is placed at the end of the
prompt as a ``<lora:name:weight>`` tag followed by its trigger words (a word
or tag already present in the prompt is moved there, never silently skipped
and never duplicated), so they are visible and editable there.  The preview
line reports the outcome (how many words are new vs. moved, how many tags
were placed).  (Generation, separately, adds anything that is not already
in the prompt, and never adds the same thing twice.)

The thumbnail picker at the bottom is a plain HTML card grid driven by a
small javascript file: clicking a card selects it *instantly* — the card
gets a frame and a sequence number in the browser right away, and the
selection is mirrored into the slots above through a hidden bridge field.
Cards can be filtered by folder and by name.  Clicking a selected card
again deselects it.

On Generate, the enabled slots are applied as ordinary
``<lora:name:weight>`` tags appended to the prompt, exactly as if you had
typed them by hand, plus the trigger words of slots whose trigger checkbox
is on, prepended to the prompt.  Both are written into the image infotext
(``Lora Lab:`` / ``Lora Lab Trigger:``) so verification runs stay traceable.

Author: slop (Muse) for tenho — v1.8.1, 2026-11-12
"""

import html as html_lib
import json
import os
import re
import time
from urllib.parse import quote

try:
    import gradio as gr
except Exception:  # pragma: no cover - gradio is always present in the WebUI
    gr = None

try:
    from modules import script_callbacks, scripts
except Exception:  # allows importing this file outside the WebUI (unit tests)

    class _DummyScript:  # minimal stand-in, only used outside the WebUI
        pass

    class _DummyScripts:
        Script = _DummyScript
        AlwaysVisible = True

    scripts = _DummyScripts()
    script_callbacks = None


# The tab prompt boxes, captured so the "reflect to prompt" button can write
# into them.  Keys are the component elem_ids.
_captured_prompts = {}


def _capture_prompt_component(component, **_kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id in ("txt2img_prompt", "img2img_prompt"):
        _captured_prompts[elem_id] = component


if script_callbacks is not None:
    try:
        script_callbacks.on_after_component(_capture_prompt_component)
    except Exception:
        pass


SLOT_COUNT = 10
SLOT_STRIDE = 4  # per slot: enabled, use_trigger, name, weight
OFF_ENABLED = 0
OFF_TRIGGER = 1
OFF_NAME = 2
OFF_WEIGHT = 3
WEIGHT_MIN = -2.0
WEIGHT_MAX = 2.0
WEIGHT_STEP_BUTTON = 0.1
GALLERY_LIMIT = None  # no cap: cards reference images by URL, so a full listing stays cheap
LORA_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
ALL_FOLDERS_LABEL = "すべて"
ROOT_FOLDER_LABEL = "（直下）"


# ---------------------------------------------------------------------------
# Pure helpers (no WebUI / gradio dependency — unit-testable)
# ---------------------------------------------------------------------------

def format_weight(weight):
    """1.0 -> '1', 0.85 -> '0.85', -0.5 -> '-0.5'."""
    try:
        value = float(weight)
    except (TypeError, ValueError):
        value = 0.0
    value = max(WEIGHT_MIN, min(WEIGHT_MAX, value))
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def clean_lora_name(name):
    """Normalise a user-typed / dropdown Lora name for use inside <lora:...>."""
    if not name:
        return ""
    text = str(name).strip().strip("<>").strip()
    lower = text.lower()
    for ext in LORA_EXTENSIONS:
        if lower.endswith(ext):
            text = text[: -len(ext)]
            break
    return text.replace("\\", "/").strip()


def build_lora_tags(slots):
    """
    slots: iterable of (enabled, name, weight) triples.
    Returns a list of '<lora:name:weight>' strings.  A Lora chosen in two
    slots collapses to one tag (the last slot's weight, first slot's place).
    """
    order = []
    by_name = {}
    for enabled, name, weight in slots:
        if not enabled:
            continue
        cleaned = clean_lora_name(name)
        if not cleaned:
            continue
        tag = f"<lora:{cleaned}:{format_weight(weight)}>"
        if cleaned not in by_name:
            order.append(cleaned)
        by_name[cleaned] = tag
    return [by_name[name] for name in order]


def inject_tags_into_prompt(prompt, tags):
    """Append tags that are not already present in the prompt text."""
    base = (prompt or "").rstrip()
    new_tags = [tag for tag in tags if tag not in base]
    if not new_tags:
        return prompt or ""
    return (base + " " + " ".join(new_tags)).strip()


def split_trigger_words(value):
    """
    Normalise a trigger-word source into a clean list.
    Accepts a comma-separated string or a list of strings (items may
    themselves contain commas).  Order preserved, duplicates removed.
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            items.extend(str(item).split(","))
    else:
        items = [str(value)]
    words = []
    for item in items:
        word = item.strip()
        if word and word not in words:
            words.append(word)
    return words


def parse_civitai_info(data):
    """Trigger words from a .civitai.info document."""
    if not isinstance(data, dict):
        return []
    words = split_trigger_words(data.get("trainedWords"))
    if words:
        return words
    versions = data.get("modelVersions")
    if isinstance(versions, list):
        for version in versions:
            if isinstance(version, dict):
                words = split_trigger_words(version.get("trainedWords"))
                if words:
                    return words
    return []


def parse_sidecar_json(data):
    """Trigger words from a <name>.json sidecar document."""
    if not isinstance(data, dict):
        return []
    for key in ("activation text", "activationText", "trainedWords",
                "triggerWords", "triggers", "trigger", "trigger_word"):
        words = split_trigger_words(data.get(key))
        if words:
            return words
    return []


def parse_safetensors_metadata(metadata):
    """Trigger words from safetensors __metadata__ ('activation text')."""
    if not isinstance(metadata, dict):
        return []
    return split_trigger_words(metadata.get("activation text"))


def prompt_tokens(prompt):
    """Comma-separated prompt tokens, lower-cased, for duplicate checks."""
    tokens = set()
    for part in (prompt or "").split(","):
        token = part.strip().lower()
        if token:
            tokens.add(token)
    return tokens


def filter_new_words(words, prompt, already=()):
    """Words that are neither in the prompt nor already collected."""
    tokens = prompt_tokens(prompt)
    seen = {w.lower() for w in already}
    fresh = []
    for word in words or []:
        lowered = word.lower()
        if lowered in tokens or lowered in seen:
            continue
        seen.add(lowered)
        fresh.append(word)
    return fresh


def collect_trigger_words(slots, resolver):
    """
    slots: iterable of (enabled, use_trigger, name, weight).
    resolver: callable(name) -> list of trigger words.
    Returns the ordered, de-duplicated trigger words of the active slots.
    """
    collected = []
    for enabled, use_trigger, name, _weight in slots:
        if not enabled or not use_trigger:
            continue
        cleaned = clean_lora_name(name)
        if not cleaned:
            continue
        for word in resolver(cleaned) or []:
            if word.lower() not in {w.lower() for w in collected}:
                collected.append(word)
    return collected


def build_prompt_with_extras(prompt, words, tags):
    """
    The full generation-time prompt assembly: fresh trigger words go to the
    front (comma-joined), fresh <lora:...> tags go to the end.
    Returns (new_prompt, added_words, added_tags).
    """
    base = prompt or ""
    added_words = filter_new_words(words, base)
    new_prompt = base
    if added_words:
        head = ", ".join(added_words)
        new_prompt = (head + ", " + base.strip()).strip() if base.strip() else head
    added_tags = [tag for tag in tags if tag not in new_prompt]
    if added_tags:
        new_prompt = (new_prompt.rstrip() + " " + " ".join(added_tags)).strip()
    return new_prompt, added_words, added_tags


def place_triggers_behind_tags(prompt, units):
    """
    Reflect semantics (tenho's call): each active Lora is written at the
    END of the prompt as ``<lora:name:weight>`` followed by its trigger
    words — the words belong with their Lora, behind its tag.

    units: [(tag, [word, ...]), ...] in slot order.
    Existing copies are relocated, never duplicated and never silently
    skipped: a slotted tag is removed wherever it currently sits (even
    inside a mixed comma token), and a trigger word already present as
    a whole comma token is removed from the body before the tail is
    appended.  Matching is case-insensitive for words; parenthesised or
    weighted tokens are left untouched.  Returns
    (new_prompt, fresh_words, moved_words, placed_tags, moved_tags).
    With no units the prompt is returned byte-identical.
    """
    units = [(tag, list(words)) for tag, words in (units or []) if tag]
    if not units:
        return prompt or "", [], [], [], []
    all_words = []
    for _tag, words in units:
        for word in words:
            if word.lower() not in {w.lower() for w in all_words}:
                all_words.append(word)
    lowered = {w.lower() for w in all_words}
    tags = [tag for tag, _words in units]

    tokens = []
    for part in (prompt or "").split(","):
        token = part.strip()
        if token:
            tokens.append(token)
    token_set = {t.lower() for t in tokens}
    fresh = [w for w in all_words if w.lower() not in token_set]
    moved = [w for w in all_words if w.lower() in token_set]
    placed_tags = [t for t in tags if t not in (prompt or "")]
    moved_tags = [t for t in tags if t in (prompt or "")]

    kept = []
    for token in tokens:
        cleaned = token
        for tag in tags:
            if tag in cleaned:
                cleaned = cleaned.replace(tag, "").strip()
        if not cleaned:
            continue
        if cleaned.lower() in lowered:
            continue
        kept.append(cleaned)

    tail_units = []
    for tag, words in units:
        unit = tag
        if words:
            unit += ", " + ", ".join(words)
        tail_units.append(unit)
    tail = " ".join(tail_units)
    body = ", ".join(kept)
    new_prompt = (body + " " + tail) if body else tail
    return new_prompt, fresh, moved, placed_tags, moved_tags


def slot_trigger_units(slots, resolver):
    """
    [(tag, [trigger words])] for the active slots, in slot order.
    A Lora chosen in two slots collapses to one unit (the last slot's
    weight and words, at the first slot's place), mirroring
    build_lora_tags.
    """
    order = []
    by_name = {}
    for enabled, use_trigger, name, weight in slots:
        if not enabled:
            continue
        cleaned = clean_lora_name(name)
        if not cleaned:
            continue
        tag = f"<lora:{cleaned}:{format_weight(weight)}>"
        words = list(resolver(cleaned) or []) if use_trigger else []
        if cleaned not in by_name:
            order.append(cleaned)
        by_name[cleaned] = (tag, words)
    return [by_name[name] for name in order]


def sweep_step_values(min_v, max_v, step):
    """
    Inclusive sweep values from min to max by step.  The endpoint is
    included when the range divides evenly by the step.  Float-safe:
    values are rounded to 4 decimals.  Empty list for invalid input.
    """
    try:
        lo, hi, st = float(min_v), float(max_v), float(step)
    except (TypeError, ValueError):
        return []
    if st <= 0 or hi < lo:
        return []
    n = int(round((hi - lo) / st))
    if n > 100000:
        n = 100000
    return [round(lo + i * st, 4) for i in range(n + 1)]


def sweep_targets(slots, resolver):
    """
    [(name, slot_tag, words)] for the batch comparison: enabled slots
    with a name, deduplicated by Lora (last slot's weight and words at
    the first slot's place), mirroring slot_trigger_units but keeping
    the name next to its tag.
    """
    order = []
    by_name = {}
    for enabled, use_trigger, name, weight in slots:
        if not enabled:
            continue
        cleaned = clean_lora_name(name)
        if not cleaned:
            continue
        tag = f"<lora:{cleaned}:{format_weight(weight)}>"
        words = list(resolver(cleaned) or []) if use_trigger else []
        if cleaned not in by_name:
            order.append(cleaned)
        by_name[cleaned] = (cleaned, tag, words)
    return [by_name[name] for name in order]


def build_sweep_schedule(targets, values):
    """[(target_index, weight), ...]: each target in turn, each weight."""
    plan = []
    for ti in range(len(targets or [])):
        for w in values or []:
            plan.append((ti, w))
    return plan


def apply_sweep_to_prompt(prompt, targets, current, weight):
    """
    Rewrite an injected prompt for one sweep step: keep ONLY the
    current target's tag (at the sweep weight) and its trigger words.
    Other targets' tags are removed and their trigger words dropped
    (whole comma tokens), so each image isolates exactly one Lora.
    """
    text = prompt or ""
    if not targets or current is None or current >= len(targets):
        return text
    cur_name, cur_tag, _cw = targets[current]
    drop_words = set()
    for j, (_nm, tag, words) in enumerate(targets):
        if j == current:
            continue
        if tag:
            text = text.replace(tag, "")
        for w in words or []:
            drop_words.add(str(w).lower())
    if drop_words:
        toks = [t.strip() for t in text.split(",")]
        toks = [t for t in toks if t and t.lower() not in drop_words]
        text = ", ".join(toks)
    new_tag = f"<lora:{cur_name}:{format_weight(weight)}>"
    if cur_tag:
        text = text.replace(cur_tag, new_tag)
    text = re.sub(r" {2,}", " ", text)
    text = text.replace(" ,", ",").strip().strip(",").strip()
    return text


def sweep_summary_text(target_count, step_count):
    total = (target_count or 0) * (step_count or 0)
    if target_count and step_count:
        return (f"一括比較：対象 {target_count} 枠 × {step_count} 段階 ＝ 合計 {total} 枚"
                "（シード固定・バッチ1で1枚ずつ生成されます）")
    return "一括比較：対象の枠（チェックON＋Lora選択あり）と範囲・刻みを設定すると、合計枚数がここに出ます"


def reflect_status_text(total_words, total_tags, fresh_words, moved_words):
    """
    Human-readable summary of a reflect action, so the outcome is never
    silent: how many trigger words were placed behind their Lora tags
    (and how many of them are new vs. moved from inside the prompt),
    and how many Lora tags were placed at the end.
    """
    parts = []
    if total_words:
        parts.append(
            f"トリガー {total_words} 語を Lora タグの後ろに反映"
            f"（新規 {len(fresh_words or [])} 語／移動 {len(moved_words or [])} 語）"
        )
    if total_tags:
        parts.append(f"タグ {total_tags} 件を末尾に配置")
    if not parts:
        return ""
    return "反映しました：" + "／".join(parts)


def step_weight(weight, delta):
    try:
        value = float(weight)
    except (TypeError, ValueError):
        value = 0.0
    value = max(WEIGHT_MIN, min(WEIGHT_MAX, value + float(delta)))
    return round(value, 2)


def find_preview_image(model_path):
    """Preview image sitting next to a Lora file, or None."""
    if not model_path:
        return None
    stem_path = os.path.splitext(model_path)[0]
    for ext in PREVIEW_EXTENSIONS:
        candidate = stem_path + ".preview" + ext
        if os.path.isfile(candidate):
            return candidate
    for ext in PREVIEW_EXTENSIONS:
        candidate = stem_path + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def selection_status_text(selected, start_slot, assigned):
    selected = list(selected or [])
    if not selected:
        return "選択中 0 件"
    listing = " ／ ".join(f"{i + 1}: {name}" for i, name in enumerate(selected))
    text = f"選択中 {len(selected)} 件（枠{start_slot}〜枠{start_slot + assigned - 1}に反映中）： {listing}"
    overflow = len(selected) - assigned
    if overflow > 0:
        text += f"（枠が足りず {overflow} 件は未反映）"
    return text


def _slot_start_ok(start_slot):
    try:
        start = int(start_slot)
    except (TypeError, ValueError):
        start = 1
    return max(1, min(SLOT_COUNT, start))


def assign_names_to_slots(values, start_slot, names):
    """Fill consecutive slots from start_slot with names. (values, assigned)"""
    result = list(values)
    start = _slot_start_ok(start_slot)
    assigned = 0
    for offset, name in enumerate(names):
        slot_index = start - 1 + offset
        if slot_index >= SLOT_COUNT:
            break
        base = slot_index * SLOT_STRIDE
        result[base + OFF_ENABLED] = True
        result[base + OFF_NAME] = name
        assigned += 1
    return result, assigned


def rewrite_selection_slots(values, start_slot, prev_count, selected):
    """Clear the range the previous selection used, then write the new one."""
    result = list(values)
    start = _slot_start_ok(start_slot)
    for j in range(max(0, prev_count)):
        idx = start - 1 + j
        if idx < SLOT_COUNT:
            result[idx * SLOT_STRIDE + OFF_NAME] = None
    return assign_names_to_slots(result, start, selected)


def parse_bridge_value(text):
    """The hidden bridge field carries a JSON list of selected Lora names."""
    if not text:
        return []
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    names = []
    for item in data:
        cleaned = clean_lora_name(item)
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return names


def filter_gallery(preview_map, query, folder_map=None, folder=None, limit=GALLERY_LIMIT):
    """
    Returns (names, [(preview_path, name), ...]) for Lora that have a
    thumbnail, optionally restricted to one folder and/or a name substring.
    folder: None or ALL_FOLDERS_LABEL = all folders; ROOT_FOLDER_LABEL = the
    root (folder ""); anything else is an exact folder path match.
    """
    needle = (query or "").strip().lower()
    names = []
    for name in sorted(preview_map.keys(), key=lambda s: s.lower()):
        preview = preview_map.get(name)
        if not preview:
            continue
        if folder is not None and folder != ALL_FOLDERS_LABEL:
            wanted = "" if folder == ROOT_FOLDER_LABEL else folder
            if (folder_map or {}).get(name, "") != wanted:
                continue
        if needle and needle not in name.lower():
            continue
        names.append(name)
        if limit is not None and len(names) >= limit:
            break
    items = [(preview_map[name], name) for name in names]
    return names, items


def count_with_preview(preview_map, query, folder_map=None, folder=None):
    needle = (query or "").strip().lower()
    total = 0
    for name, preview in preview_map.items():
        if not preview:
            continue
        if folder is not None and folder != ALL_FOLDERS_LABEL:
            wanted = "" if folder == ROOT_FOLDER_LABEL else folder
            if (folder_map or {}).get(name, "") != wanted:
                continue
        if needle and needle not in name.lower():
            continue
        total += 1
    return total


def folder_choices(folder_map, preview_map):
    folders = set()
    has_root = False
    for name, folder in (folder_map or {}).items():
        if not preview_map.get(name):
            continue
        if folder:
            folders.add(folder)
        else:
            has_root = True
    choices = [ALL_FOLDERS_LABEL] + sorted(folders, key=lambda s: s.lower())
    if has_root:
        choices.append(ROOT_FOLDER_LABEL)
    return choices


_CARDS_STYLE = """
<style>
.lora-lab-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:4px 0}
.lora-lab-card{position:relative;cursor:pointer;border-radius:8px;overflow:hidden;background:rgba(127,127,127,.12)}
.lora-lab-card img{width:100%;height:110px;object-fit:cover;display:block}
.lora-lab-card .lora-lab-cap{font-size:10px;line-height:1.25;padding:3px 4px;word-break:break-all;opacity:.85}
.lora-lab-card.lora-lab-selected{outline:3px solid #f472b6;outline-offset:-3px}
.lora-lab-badge{position:absolute;top:4px;left:4px;min-width:22px;height:22px;border-radius:11px;
 background:#f472b6;color:#fff;font-weight:700;font-size:13px;display:none;align-items:center;justify-content:center;padding:0 5px;z-index:2}
.lora-lab-empty{opacity:.6;padding:8px 2px}
</style>
"""


def build_cards_html(displayed, preview_map, selected, bridge_id, token):
    """The clickable card grid (selection state rendered into the markup)."""
    selected = list(selected or [])
    names_json = html_lib.escape(json.dumps(list(displayed or []), ensure_ascii=False), quote=True)
    selected_json = html_lib.escape(json.dumps(selected, ensure_ascii=False), quote=True)
    parts = [_CARDS_STYLE]
    parts.append(
        f'<div class="lora-lab-grid" data-names="{names_json}" '
        f'data-selected="{selected_json}" data-bridge="{html_lib.escape(bridge_id, quote=True)}" '
        f'data-token="{html_lib.escape(str(token), quote=True)}">'
    )
    if not displayed:
        parts.append('<div class="lora-lab-empty">（表示するサムネがありません）</div>')
    for idx, name in enumerate(displayed or []):
        preview = preview_map.get(name)
        if name in selected:
            number = selected.index(name) + 1
            cls = "lora-lab-card lora-lab-selected"
            badge = f'<span class="lora-lab-badge" style="display:flex">{number}</span>'
        else:
            cls = "lora-lab-card"
            badge = '<span class="lora-lab-badge"></span>'
        src = ""
        if preview:
            src = "/file=" + quote(str(preview).replace(os.sep, "/"), safe="/:")
        parts.append(
            f'<div class="{cls}" data-idx="{idx}" onclick="loraLabToggle(this)">'
            f"{badge}"
            f'<img src="{html_lib.escape(src, quote=True)}" loading="lazy">'
            f'<div class="lora-lab-cap">{html_lib.escape(str(name))}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Trigger-word loading (file IO)
# ---------------------------------------------------------------------------

def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def _read_safetensors_metadata(model_path):
    """The __metadata__ dict from a safetensors header, or {}."""
    try:
        with open(model_path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {}
            size = int.from_bytes(raw, "little")
            if size <= 0 or size > 64 * 1024 * 1024:
                return {}
            header = json.loads(f.read(size).decode("utf-8", errors="replace"))
        metadata = header.get("__metadata__")
        return metadata if isinstance(metadata, dict) else {}
    except Exception:
        return {}


def load_trigger_words_for_path(model_path):
    """
    Trigger words for one Lora file, in priority order:
    <name>.civitai.info (trainedWords) -> <name>.json sidecar ->
    safetensors 'activation text' metadata.
    """
    if not model_path:
        return []
    stem_path = os.path.splitext(model_path)[0]
    civitai_path = stem_path + ".civitai.info"
    if os.path.isfile(civitai_path):
        words = parse_civitai_info(_read_json_file(civitai_path))
        if words:
            return words
    sidecar_path = stem_path + ".json"
    if os.path.isfile(sidecar_path):
        words = parse_sidecar_json(_read_json_file(sidecar_path))
        if words:
            return words
    if str(model_path).lower().endswith(".safetensors"):
        return parse_safetensors_metadata(_read_safetensors_metadata(model_path))
    return []


_PATH_MAP_CACHE = {}
_TRIGGER_CACHE = {}


def get_trigger_words(name):
    """Cached trigger-word lookup by Lora name (builds the catalog once)."""
    cleaned = clean_lora_name(name)
    if not cleaned:
        return []
    if cleaned in _TRIGGER_CACHE:
        return _TRIGGER_CACHE[cleaned]
    if not _PATH_MAP_CACHE:
        try:
            list_lora_catalog()
        except Exception:
            pass
    path = _PATH_MAP_CACHE.get(cleaned)
    if path is None:
        path = _PATH_MAP_CACHE.get(cleaned.rsplit("/", 1)[-1])
    words = load_trigger_words_for_path(path) if path else []
    if not words:
        # Same-stem fallback: another copy of the same Lora file may carry
        # the trigger data even when the resolved copy does not.
        stem = cleaned.rsplit("/", 1)[-1].lower()
        for other in _PATH_MAP_CACHE.values():
            if not other or other == path:
                continue
            other_stem = os.path.splitext(os.path.basename(str(other)))[0].lower()
            if other_stem == stem:
                words = load_trigger_words_for_path(other)
                if words:
                    break
    _TRIGGER_CACHE[cleaned] = words
    return words


# ---------------------------------------------------------------------------
# Lora discovery
# ---------------------------------------------------------------------------

def _candidate_lora_dirs():
    dirs = []

    def _add(path):
        if path and isinstance(path, str) and path not in dirs:
            dirs.append(path)

    try:
        from modules import paths, shared

        cmd = getattr(shared, "cmd_opts", None)
        if cmd is not None:
            for attr in ("lora_dir", "lyco_dir_backcompat", "lora_dirs"):
                value = getattr(cmd, attr, None)
                if isinstance(value, (list, tuple)):
                    for item in value:
                        _add(item)
                else:
                    _add(value)
        models_path = getattr(paths, "models_path", None)
        if models_path:
            _add(os.path.join(models_path, "Lora"))
            _add(os.path.join(models_path, "LyCORIS"))
        data_dir = getattr(paths, "data_dir", None)
        if data_dir:
            _add(os.path.join(data_dir, "models", "Lora"))
    except Exception:
        pass

    try:
        webui_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        )
        _add(os.path.join(webui_root, "models", "Lora"))
        _add(os.path.join(webui_root, "models", "LyCORIS"))
    except Exception:
        pass

    return dirs


def _folder_for_path(filename, dirs):
    """Folder of a Lora file, relative to a known Lora root ('' = root)."""
    if not filename:
        return ""
    try:
        norm = os.path.normpath(str(filename))
    except Exception:
        return ""
    for base in dirs or []:
        try:
            base_norm = os.path.normpath(base)
        except Exception:
            continue
        if norm.lower().startswith(base_norm.lower() + os.sep):
            rel = norm[len(base_norm) + 1:]
            return os.path.dirname(rel).replace(os.sep, "/")
    return os.path.basename(os.path.dirname(norm))


def _catalog_from_builtin_network_registry(dirs):
    """{name: (preview, folder, path)} from the built-in Lora registry."""
    catalog = {}
    for module_name in ("networks",):
        try:
            module = __import__(module_name)
            available = getattr(module, "available_networks", None)
            if available is None:
                continue
            if len(available) == 0 and hasattr(module, "list_available_networks"):
                try:
                    module.list_available_networks()
                    available = getattr(module, "available_networks", None) or {}
                except Exception:
                    pass
            for key, entry in available.items():
                name = clean_lora_name(str(key))
                if not name:
                    continue
                filename = getattr(entry, "filename", None)
                preview = find_preview_image(filename) if filename else None
                catalog[name] = (preview, _folder_for_path(filename, dirs), filename)
        except Exception:
            continue
    return catalog


def _walk_model_files(base):
    """Yield (rel_no_ext, model_path) under base, descending into junctions
    by hand (os.walk does not reliably follow Windows junctions)."""
    results = []
    visited = set()

    def _recurse(dir_path, rel_prefix, depth):
        if depth > 12:
            return
        try:
            real = os.path.realpath(dir_path)
        except Exception:
            real = dir_path
        if real in visited:
            return
        visited.add(real)
        try:
            entries = list(os.scandir(dir_path))
        except OSError:
            return
        for entry in entries:
            try:
                if entry.is_dir():
                    _recurse(entry.path, rel_prefix + entry.name + "/", depth + 1)
                elif entry.is_file() and entry.name.lower().endswith(LORA_EXTENSIONS):
                    rel_no_ext = rel_prefix + os.path.splitext(entry.name)[0]
                    results.append((rel_no_ext, entry.path))
            except OSError:
                continue

    _recurse(base, "", 0)
    return results


def _scan_catalog(dirs):
    """{name: (preview, folder, path)} from scanning the Lora folders."""
    found = {}  # stem -> list of (rel_no_ext, model_path)
    for base in dirs:
        if not base or not os.path.isdir(base):
            continue
        for rel_no_ext, model_path in _walk_model_files(base):
            stem = os.path.splitext(os.path.basename(model_path))[0]
            found.setdefault(stem, []).append((rel_no_ext, model_path))
    catalog = {}
    for stem, entries in found.items():
        if len(entries) == 1:
            rel_no_ext, model_path = entries[0]
            folder = rel_no_ext.rsplit("/", 1)[0] if "/" in rel_no_ext else ""
            catalog[stem] = (find_preview_image(model_path), folder, model_path)
        else:
            for rel_no_ext, model_path in entries:
                folder = rel_no_ext.rsplit("/", 1)[0] if "/" in rel_no_ext else ""
                catalog[rel_no_ext] = (find_preview_image(model_path), folder, model_path)
    return catalog


def list_lora_catalog():
    """Returns (sorted_names, {name: preview}, {name: folder}, {name: path})."""
    dirs = _candidate_lora_dirs()
    merged = _scan_catalog(dirs)
    for name, value in _catalog_from_builtin_network_registry(dirs).items():
        if name not in merged or (merged[name][0] is None and value[0] is not None):
            merged[name] = value
    names = sorted(merged.keys(), key=lambda s: s.lower())
    preview_map = {name: merged[name][0] for name in names}
    folder_map = {name: merged[name][1] for name in names}
    path_map = {name: merged[name][2] for name in names}
    _PATH_MAP_CACHE.clear()
    _PATH_MAP_CACHE.update(path_map)
    return names, preview_map, folder_map, path_map


def list_lora_names():
    return list_lora_catalog()[0]


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

class LoraLabScript(scripts.Script):
    _sweep = None  # per-job sweep plan, set in before_process

    def title(self):
        return "Lora Lab"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # -- UI -----------------------------------------------------------------
    def ui(self, is_img2img):
        catalog_names, preview_map, folder_map, _path_map = list_lora_catalog()
        bridge_id = "lora_lab_bridge_img2img" if is_img2img else "lora_lab_bridge_txt2img"

        def _dropdown(label, choices):
            kwargs = dict(choices=choices, value=None, label=label,
                          show_label=False, scale=3)
            try:
                return gr.Dropdown(allow_custom_value=True, **kwargs)
            except TypeError:
                kwargs.pop("scale", None)
                try:
                    return gr.Dropdown(allow_custom_value=True, **kwargs)
                except TypeError:
                    return gr.Dropdown(**kwargs)

        def _checkbox(label, value=True, min_width=None):
            kwargs = dict(label=label, value=value, scale=0)
            if min_width:
                kwargs["min_width"] = min_width
            try:
                return gr.Checkbox(**kwargs)
            except TypeError:
                kwargs.pop("scale", None)
                kwargs.pop("min_width", None)
                return gr.Checkbox(**kwargs)

        def _wide_button(text):
            try:
                return gr.Button(text, scale=1)
            except TypeError:
                return gr.Button(text)

        def _small_button(text):
            try:
                return gr.Button(text, scale=0, min_width=72)
            except TypeError:
                return gr.Button(text)

        slot_components = []

        with gr.Accordion("Lora Lab", open=False,
                          elem_id="lora_lab_accordion"):
            # Compact spacing for this accordion only.  (A style.css file
            # also ships with the extension, but this Forge build does not
            # serve extension stylesheets, so the rules are inlined here.)
            gr.HTML(
                "<style>"
                "#lora_lab_accordion{--layout-gap:3px;--block-padding:3px 6px}"
                "#lora_lab_accordion .row{gap:4px;margin:0}"
                "#lora_lab_accordion .block{padding:2px 4px}"
                "#lora_lab_accordion .form{gap:3px}"
                "#lora_lab_accordion button{white-space:nowrap}"
                "</style>"
            )
            master_enabled = gr.Checkbox(
                label="Lora Lab を有効にする（OFF の間は生成時にタグを付けません）",
                value=True,
            )
            # No help paragraph here (tenho's call): how it works lives in
            # the README, the panel stays clean.

            with gr.Row():
                reflect_btn = _wide_button("⬆ プロンプトに反映（トリガーワード＋タグをプロンプト欄へ入れる）")

            with gr.Row():
                refresh_btn = _wide_button("🔄 Lora一覧を再読み込み")
                clear_btn = _wide_button("🧹 全枠クリア")
                reset_weight_btn = _wide_button("強度を全部 1.0 に戻す")

            for i in range(SLOT_COUNT):
                # One row per slot (tenho's layout call): the number
                # checkbox, the name dropdown, the trigger checkbox, and
                # the strength slider with its step buttons.  No per-slot
                # thumbnail (it cost more width than it was worth); a
                # separator line under each slot keeps the ten slots
                # visually apart instead.  The slider is kept narrow
                # (scale=1) so the strength block does not stretch across
                # the whole accordion and its value box stays close to
                # the trigger checkbox cluster.  Labels are hidden (the
                # number checkbox identifies the slot) and the scoped
                # CSS trims the gaps.
                with gr.Row():
                    enabled = _checkbox(f"{i + 1}", True, min_width=42)
                    name = _dropdown(f"Lora {i + 1}", catalog_names)
                    use_trigger = _checkbox("トリガー", True, min_width=80)
                    try:
                        weight = gr.Slider(
                            minimum=WEIGHT_MIN,
                            maximum=WEIGHT_MAX,
                            step=0.05,
                            value=1.0,
                            label="強度",
                            show_label=False,
                            scale=1,
                        )
                    except TypeError:
                        weight = gr.Slider(
                            minimum=WEIGHT_MIN,
                            maximum=WEIGHT_MAX,
                            step=0.05,
                            value=1.0,
                            label="強度",
                            show_label=False,
                        )
                    minus_btn = _small_button("−0.1")
                    plus_btn = _small_button("＋0.1")

                minus_btn.click(
                    fn=lambda v: step_weight(v, -WEIGHT_STEP_BUTTON),
                    inputs=[weight],
                    outputs=[weight],
                    show_progress=False,
                )
                plus_btn.click(
                    fn=lambda v: step_weight(v, WEIGHT_STEP_BUTTON),
                    inputs=[weight],
                    outputs=[weight],
                    show_progress=False,
                )

                slot_components.extend([enabled, use_trigger, name, weight])
                gr.HTML(
                    "<hr style='margin:1px 0;border:none;"
                    "border-top:1px solid rgba(127,127,127,.28)'>"
                )

            preview = gr.Textbox(
                label="生成時に追加される内容（プレビュー）",
                value="",
                interactive=False,
                placeholder="（枠で Lora を選ぶとここに出ます）",
            )

            # Batch comparison (strength sweep): when ON, generation
            # produces one image per (checked Lora x step value), each
            # image isolating one Lora at one weight, seed fixed.
            def _number(label, value, mn, mx):
                try:
                    return gr.Number(label=label, value=value, minimum=mn,
                                     maximum=mx, step=0.01, show_label=True)
                except TypeError:
                    return gr.Number(label=label, value=value)

            with gr.Row():
                sweep_enabled = _checkbox("一括比較", False, min_width=100)
                sweep_min = _number("最小", -1.0, -4.0, 4.0)
                sweep_max = _number("最大", 2.0, -4.0, 4.0)
                sweep_step = _number("刻み", 0.5, 0.01, 4.0)
            sweep_status = gr.Textbox(
                label="",
                value=sweep_summary_text(0, 0),
                interactive=False,
                show_label=False,
            )

            gr.Markdown(
                "**サムネ一覧から選ぶ**（サムネ画像がある Lora だけが出ます）"
                " ― フォルダと名前で絞り込んで表示し、サムネをクリックした瞬間に選択され、"
                "開始枠から選んだ順に枠へ入ります。選んだサムネには枠と番号が付き、"
                "もう一度クリックで解除です。"
            )
            with gr.Row():
                target_slot = gr.Dropdown(
                    choices=[f"枠{i + 1}" for i in range(SLOT_COUNT)],
                    value="枠1",
                    label="入れる枠（開始枠）",
                    show_label=True,
                )
                folder_dropdown = gr.Dropdown(
                    choices=folder_choices(folder_map, preview_map),
                    value=ALL_FOLDERS_LABEL,
                    label="フォルダ",
                    show_label=True,
                )
                gallery_search = gr.Textbox(
                    label="名前で絞り込み",
                    value="",
                    placeholder="（空欄なら先頭から表示）",
                )
                gallery_btn = _wide_button("🖼 サムネ一覧を表示")
            gallery_status = gr.Textbox(
                label="",
                value="",
                interactive=False,
                placeholder="（「サムネ一覧を表示」を押すとここに件数が出ます）",
            )
            cards_html = gr.HTML(value="")
            gallery_state = gr.State([])
            selected_state = gr.State([])
            bridge = gr.Textbox(value="", visible=False, elem_id=bridge_id)

        def _slots_from(values):
            return [
                (values[i + OFF_ENABLED], values[i + OFF_TRIGGER],
                 values[i + OFF_NAME], values[i + OFF_WEIGHT])
                for i in range(0, SLOT_COUNT * SLOT_STRIDE, SLOT_STRIDE)
            ]

        def _preview_text(slots):
            words = collect_trigger_words(slots, get_trigger_words)
            tags = build_lora_tags([(en, nm, w) for en, _t, nm, w in slots])
            parts = []
            if words:
                parts.append("トリガー: " + ", ".join(words))
            elif any(en and trg and clean_lora_name(nm)
                     for en, trg, nm, _w in slots):
                # A named slot wants triggers but the Lora carries no
                # trigger data in any source file.  Say so explicitly:
                # silence here reads as "the feature is broken".
                parts.append("トリガー: （データなし）")
            if tags:
                parts.append("タグ: " + " ".join(tags))
            return " ｜ ".join(parts)

        def _sync_outputs(values):
            slots = _slots_from(values)
            return [_preview_text(slots)]

        def _on_slot_change(*args):
            return _sync_outputs(list(args))

        for component in slot_components:
            component.change(
                fn=_on_slot_change,
                inputs=slot_components,
                outputs=[preview],
                show_progress=False,
            )

        def _sweep_count(*args):
            vals = list(args)
            slots = _slots_from(vals[:SLOT_COUNT * SLOT_STRIDE])
            _on, mn, mx, st = vals[-4], vals[-3], vals[-2], vals[-1]
            targets = sweep_targets(slots, lambda _n: [])
            steps = sweep_step_values(mn, mx, st)
            return sweep_summary_text(len(targets), len(steps))

        for component in slot_components + [sweep_enabled, sweep_min, sweep_max, sweep_step]:
            component.change(
                fn=_sweep_count,
                inputs=slot_components + [sweep_enabled, sweep_min, sweep_max, sweep_step],
                outputs=[sweep_status],
                show_progress=False,
            )

        # Reflect current slots into the tab's prompt box.
        prompt_key = "img2img_prompt" if is_img2img else "txt2img_prompt"
        prompt_component = _captured_prompts.get(prompt_key)
        if prompt_component is not None:
            def _reflect(prompt_text, *args):
                values = list(args)
                slots = _slots_from(values)
                units = slot_trigger_units(slots, get_trigger_words)
                total_words = sum(len(words) for _tag, words in units)
                new_prompt, fresh, moved, _pt, _mt = place_triggers_behind_tags(prompt_text or "", units)
                status = reflect_status_text(total_words, len(units), fresh, moved)
                if not status:
                    status = _preview_text(slots)
                return new_prompt, status

            reflect_btn.click(
                fn=_reflect,
                inputs=[prompt_component] + slot_components,
                outputs=[prompt_component, preview],
                show_progress=False,
            )

        def _refresh(*args):
            nonlocal catalog_names, preview_map, folder_map
            catalog_names, preview_map, folder_map, _pm = list_lora_catalog()
            _TRIGGER_CACHE.clear()
            values = list(args)
            result = []
            for idx, value in enumerate(values):
                if idx % SLOT_STRIDE == OFF_NAME:
                    try:
                        result.append(gr.Dropdown.update(choices=catalog_names, value=value))
                    except Exception:
                        result.append(gr.update(choices=catalog_names, value=value))
                else:
                    result.append(value)
            folder_update = gr.update(choices=folder_choices(folder_map, preview_map))
            return result + _sync_outputs(values) + [folder_update]

        refresh_btn.click(
            fn=_refresh,
            inputs=slot_components,
            outputs=slot_components + [preview] + [folder_dropdown],
            show_progress=False,
        )

        def _clear(displayed, _selected):
            values = []
            for _ in range(SLOT_COUNT):
                values.extend([True, True, None, 1.0])
            token = str(int(time.time() * 1000))
            cards = build_cards_html(displayed, preview_map, [], bridge_id, token)
            return values + [""] + [cards, [], "選択中 0 件"]

        clear_btn.click(
            fn=_clear,
            inputs=[gallery_state, selected_state],
            outputs=slot_components + [preview] + [cards_html, selected_state, gallery_status],
            show_progress=False,
        )

        def _reset_weights(*args):
            values = list(args)
            for idx in range(OFF_WEIGHT, len(values), SLOT_STRIDE):
                values[idx] = 1.0
            return values + _sync_outputs(values)

        reset_weight_btn.click(
            fn=_reset_weights,
            inputs=slot_components,
            outputs=slot_components + [preview],
            show_progress=False,
        )

        def _show_gallery(query, folder, selected):
            names, _items = filter_gallery(preview_map, query, folder_map, folder)
            total = count_with_preview(preview_map, query, folder_map, folder)
            token = str(int(time.time() * 1000))
            cards = build_cards_html(names, preview_map, selected, bridge_id, token)
            if total == 0:
                status = "サムネ画像がある Lora が見つかりませんでした。"
            elif len(names) < total:
                status = f"サムネ付き {total} 件中、先頭 {len(names)} 件を表示中。クリックで選べます。"
            else:
                status = f"サムネ付き {total} 件を表示中。クリックで選べます。"
            if selected:
                status += f"（現在 {len(selected)} 件選択中）"
            return cards, names, status

        gallery_btn.click(
            fn=_show_gallery,
            inputs=[gallery_search, folder_dropdown, selected_state],
            outputs=[cards_html, gallery_state, gallery_status],
            show_progress=False,
        )

        def _on_bridge(bridge_text, target, selected, *args):
            values = list(args)
            chosen = parse_bridge_value(bridge_text)
            start_slot = _slot_start_ok(str(target).replace("枠", "").strip()) if target else 1
            prev_count = len(selected or [])
            new_values, assigned = rewrite_selection_slots(values, start_slot, prev_count, chosen)
            status = selection_status_text(chosen, start_slot, assigned)
            return new_values + _sync_outputs(new_values) + [status, chosen]

        bridge.input(
            fn=_on_bridge,
            inputs=[bridge, target_slot, selected_state] + slot_components,
            outputs=slot_components + [preview] + [gallery_status, selected_state],
            show_progress=False,
        )

        return [master_enabled] + slot_components + [sweep_enabled, sweep_min, sweep_max, sweep_step]

    # -- Generation ----------------------------------------------------------
    # The prompt mutation MUST happen in before_process, not process:
    # Forge snapshots p.prompt into all_prompts (setup_prompts) BEFORE
    # scripts.process() runs, and generation reads only that snapshot —
    # so anything appended in process() is recorded in the infotext but
    # never reaches the sampler (the <lora:> tags never activate and the
    # trigger words never condition the image).  before_process is
    # called from process_images() before the snapshot, so edits made
    # here are part of the prompt that is actually parsed and sampled.
    def before_process(self, p, master_enabled=True, *slot_args):
        self._sweep = None
        if not master_enabled:
            return

        values = list(slot_args)
        slot_values = values[:SLOT_COUNT * SLOT_STRIDE]
        sweep_values = values[SLOT_COUNT * SLOT_STRIDE:SLOT_COUNT * SLOT_STRIDE + 4]
        slots = []
        for i in range(0, len(slot_values) - 3, SLOT_STRIDE):
            slots.append((slot_values[i + OFF_ENABLED], slot_values[i + OFF_TRIGGER],
                          slot_values[i + OFF_NAME], slot_values[i + OFF_WEIGHT]))
        if not slots:
            return

        words = collect_trigger_words(slots, get_trigger_words)
        tags = build_lora_tags([(en, nm, w) for en, _t, nm, w in slots])
        if not words and not tags:
            return

        new_prompt, added_words, added_tags = build_prompt_with_extras(p.prompt or "", words, tags)
        p.prompt = new_prompt

        try:
            if p.extra_generation_params is None:
                p.extra_generation_params = {}
            if tags:
                p.extra_generation_params["Lora Lab"] = " ".join(tags)
            if added_words:
                p.extra_generation_params["Lora Lab Trigger"] = ", ".join(added_words)
        except Exception:
            pass

        # Batch comparison setup: plan one image per (target x step).
        # The per-image rewrite happens in before_process_batch, which
        # Forge calls for every iteration BEFORE that iteration's
        # prompt is parsed and its Loras activated.
        if len(sweep_values) == 4 and sweep_values[0]:
            targets = sweep_targets(slots, get_trigger_words)
            steps = sweep_step_values(sweep_values[1], sweep_values[2], sweep_values[3])
            plan = build_sweep_schedule(targets, steps)
            if plan:
                self._sweep = {"targets": targets, "plan": plan}
                try:
                    p.n_iter = len(plan)
                    p.batch_size = 1
                except Exception:
                    pass
                try:
                    if p.extra_generation_params is None:
                        p.extra_generation_params = {}
                    p.extra_generation_params["Lora Lab Sweep"] = (
                        f"{len(targets)} Loras x {len(steps)} steps = {len(plan)} images")
                except Exception:
                    pass

    def before_process_batch(self, p, *args, **kwargs):
        state = getattr(self, "_sweep", None)
        if not state:
            return
        prompts = kwargs.get("prompts")
        if prompts is None:
            return
        batch_number = kwargs.get("batch_number", 0) or 0
        plan = state["plan"]
        if batch_number >= len(plan):
            return
        targets = state["targets"]
        ti, weight = plan[batch_number]
        for i in range(len(prompts)):
            prompts[i] = apply_sweep_to_prompt(prompts[i], targets, ti, weight)
        # Keep the records truthful per image: the infotext headline
        # and the hires prompts are read from these lists later.
        try:
            if batch_number < len(p.all_prompts):
                p.all_prompts[batch_number] = apply_sweep_to_prompt(
                    p.all_prompts[batch_number], targets, ti, weight)
        except Exception:
            pass
        try:
            hr = getattr(p, "all_hr_prompts", None)
            if hr is not None and batch_number < len(hr):
                hr[batch_number] = apply_sweep_to_prompt(hr[batch_number], targets, ti, weight)
        except Exception:
            pass
        try:
            if p.extra_generation_params is None:
                p.extra_generation_params = {}
            name = targets[ti][0]
            p.extra_generation_params["Lora Lab"] = f"<lora:{name}:{format_weight(weight)}>"
            p.extra_generation_params["Lora Lab Sweep"] = (
                f"{name} @ {format_weight(weight)} ({batch_number + 1}/{len(plan)})")
        except Exception:
            pass

    def postprocess(self, p, processed, *args):
        self._sweep = None

    def process(self, p, master_enabled=True, *slot_args):
        # Prompt changes live in before_process (see above); process
        # runs after the prompt snapshot, too late to affect generation.
        # It does run after all_seeds is built and before the iteration
        # loop, which makes it the one place that can pin the sweep's
        # seeds: the per-iteration RNG (p.rng) is constructed from
        # all_seeds BEFORE before_process_batch is called, so fixing
        # seeds in the batch hook would be too late.
        if getattr(self, "_sweep", None):
            try:
                if p.all_seeds:
                    base = p.all_seeds[0]
                    p.all_seeds[:] = [base] * len(p.all_seeds)
            except Exception:
                pass
        return
