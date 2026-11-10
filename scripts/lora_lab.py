# -*- coding: utf-8 -*-
"""
Lora Lab — multi-Lora verification helper for AUTOMATIC1111 / Forge (Classic & Neo).

All UI text uses the spelling "Lora".

What it does
------------
Adds an always-visible accordion ("Lora Lab") to the txt2img and img2img tabs
with 10 Lora slots.  Each slot has:

  * an on/off checkbox
  * a searchable dropdown of the Lora files the WebUI can see
    (you can also type a name by hand if a Lora is not listed)
  * a preview thumbnail of the selected Lora (its .preview image, if any)
  * a strength slider (-2.00 .. +2.00) with -0.1 / +0.1 step buttons

At the top of the accordion, a "reflect to prompt" button appends the current
slots' ``<lora:name:weight>`` tags directly into the prompt box, so they are
visible and editable there.  (Generation also adds any tags that are not
already in the prompt, and never adds one twice.)

The thumbnail picker at the bottom is a plain HTML card grid driven by a
small javascript file: clicking a card selects it *instantly* — the card
gets a frame and a sequence number in the browser right away, and the
selection is mirrored into the slots above through a hidden bridge field.
Cards can be filtered by folder and by name.  Clicking a selected card
again deselects it.

On Generate, the enabled slots are appended to the prompt as ordinary
``<lora:name:weight>`` tags, exactly as if you had typed them by hand, and
the combined tag string is written into the image infotext as
``Lora Lab: <lora:...> ...`` so verification runs stay traceable.

Author: slop (Muse) for tenho — v1.4, 2026-11-10
"""

import html as html_lib
import json
import os
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
WEIGHT_MIN = -2.0
WEIGHT_MAX = 2.0
WEIGHT_STEP_BUTTON = 0.1
GALLERY_LIMIT = 60
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


def toggle_selection(selected, name):
    result = list(selected or [])
    if name in result:
        result.remove(name)
    else:
        result.append(name)
    return result


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


def assign_names_to_slots(values, start_slot, names):
    """Fill consecutive slots from start_slot with names. (values, assigned)"""
    result = list(values)
    try:
        start = int(start_slot)
    except (TypeError, ValueError):
        start = 1
    start = max(1, min(SLOT_COUNT, start))
    assigned = 0
    for offset, name in enumerate(names):
        slot_index = start - 1 + offset
        if slot_index >= SLOT_COUNT:
            break
        base = slot_index * 3
        result[base] = True
        result[base + 1] = name
        assigned += 1
    return result, assigned


def rewrite_selection_slots(values, start_slot, prev_count, selected):
    """Clear the range the previous selection used, then write the new one."""
    result = list(values)
    try:
        start = int(start_slot)
    except (TypeError, ValueError):
        start = 1
    start = max(1, min(SLOT_COUNT, start))
    for j in range(max(0, prev_count)):
        idx = start - 1 + j
        if idx < SLOT_COUNT:
            result[idx * 3 + 1] = None
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
        if len(names) >= limit:
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
    """{name: (preview, folder)} from the built-in Lora registry."""
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
                catalog[name] = (preview, _folder_for_path(filename, dirs))
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
    """{name: (preview, folder)} from scanning the Lora folders."""
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
            catalog[stem] = (find_preview_image(model_path), folder)
        else:
            for rel_no_ext, model_path in entries:
                folder = rel_no_ext.rsplit("/", 1)[0] if "/" in rel_no_ext else ""
                catalog[rel_no_ext] = (find_preview_image(model_path), folder)
    return catalog


def list_lora_catalog():
    """Returns (sorted_names, {name: preview}, {name: folder})."""
    dirs = _candidate_lora_dirs()
    merged = _scan_catalog(dirs)
    for name, value in _catalog_from_builtin_network_registry(dirs).items():
        if name not in merged or (merged[name][0] is None and value[0] is not None):
            merged[name] = value
    names = sorted(merged.keys(), key=lambda s: s.lower())
    preview_map = {name: merged[name][0] for name in names}
    folder_map = {name: merged[name][1] for name in names}
    return names, preview_map, folder_map


def list_lora_names():
    return list_lora_catalog()[0]


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

class LoraLabScript(scripts.Script):
    def title(self):
        return "Lora Lab"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # -- UI -----------------------------------------------------------------
    def ui(self, is_img2img):
        catalog_names, preview_map, folder_map = list_lora_catalog()
        bridge_id = "lora_lab_bridge_img2img" if is_img2img else "lora_lab_bridge_txt2img"

        def _dropdown(label, choices):
            kwargs = dict(choices=choices, value=None, label=label, show_label=True)
            try:
                return gr.Dropdown(allow_custom_value=True, **kwargs)
            except TypeError:
                return gr.Dropdown(**kwargs)

        def _wide_button(text):
            try:
                return gr.Button(text, scale=1)
            except TypeError:
                return gr.Button(text)

        def _small_button(text):
            try:
                return gr.Button(text, scale=0, min_width=56)
            except TypeError:
                return gr.Button(text)

        def _image():
            kwargs = dict(value=None, type="filepath", interactive=False,
                          show_label=False, height=76, width=76)
            try:
                return gr.Image(**kwargs)
            except TypeError:
                kwargs.pop("width", None)
                return gr.Image(**kwargs)

        def _preview_for(name):
            cleaned = clean_lora_name(name)
            if not cleaned:
                return None
            if cleaned in preview_map:
                return preview_map[cleaned]
            stem = cleaned.rsplit("/", 1)[-1]
            return preview_map.get(stem)

        slot_enabled = []
        slot_names = []
        slot_weights = []
        slot_images = []

        with gr.Accordion("Lora Lab — 複数Lora検証（10枠）", open=False):
            master_enabled = gr.Checkbox(
                label="Lora Lab を有効にする（OFF の間は生成時にタグを付けません）",
                value=True,
            )
            gr.Markdown(
                "枠ごとに Lora を選んで強度を決めると、生成時に `<lora:名前:強度>` が"
                "プロンプト末尾へ自動で付きます。一時的に外したい枠は左端のチェックを外してね。"
                "下のサムネ一覧はクリックした瞬間に選択されて枠へ入ります。"
            )

            with gr.Row():
                reflect_btn = _wide_button("⬆ プロンプトに反映（今の枠のタグをプロンプト欄へ入れる）")

            with gr.Row():
                refresh_btn = _wide_button("🔄 Lora一覧を再読み込み")
                clear_btn = _wide_button("🧹 全枠クリア")
                reset_weight_btn = _wide_button("強度を全部 1.0 に戻す")

            for i in range(SLOT_COUNT):
                with gr.Row():
                    enabled = gr.Checkbox(label=f"{i + 1}", value=True)
                    name = _dropdown(f"Lora {i + 1}", catalog_names)
                    image = _image()
                    weight = gr.Slider(
                        minimum=WEIGHT_MIN,
                        maximum=WEIGHT_MAX,
                        step=0.05,
                        value=1.0,
                        label="強度",
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

                slot_enabled.append(enabled)
                slot_names.append(name)
                slot_weights.append(weight)
                slot_images.append(image)

            preview = gr.Textbox(
                label="生成時に追加されるタグ（プレビュー）",
                value="",
                interactive=False,
                placeholder="（枠で Lora を選ぶとここに出ます）",
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

        slot_components = []
        for enabled, name, weight in zip(slot_enabled, slot_names, slot_weights):
            slot_components.extend([enabled, name, weight])

        def _sync_outputs(values):
            slots = list(zip(values[0::3], values[1::3], values[2::3]))
            text = " ".join(build_lora_tags(slots))
            images = [_preview_for(name) for _en, name, _w in slots]
            return [text] + images

        def _on_slot_change(*args):
            return _sync_outputs(list(args))

        for component in slot_components:
            component.change(
                fn=_on_slot_change,
                inputs=slot_components,
                outputs=[preview] + slot_images,
                show_progress=False,
            )

        # Reflect current slots into the tab's prompt box.
        prompt_key = "img2img_prompt" if is_img2img else "txt2img_prompt"
        prompt_component = _captured_prompts.get(prompt_key)
        if prompt_component is not None:
            def _reflect(prompt_text, *args):
                values = list(args)
                slots = list(zip(values[0::3], values[1::3], values[2::3]))
                return inject_tags_into_prompt(prompt_text or "", build_lora_tags(slots))

            reflect_btn.click(
                fn=_reflect,
                inputs=[prompt_component] + slot_components,
                outputs=[prompt_component],
                show_progress=False,
            )

        def _refresh(*args):
            nonlocal catalog_names, preview_map, folder_map
            catalog_names, preview_map, folder_map = list_lora_catalog()
            values = list(args)
            result = []
            for idx, value in enumerate(values):
                if idx % 3 == 1:
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
            outputs=slot_components + [preview] + slot_images + [folder_dropdown],
            show_progress=False,
        )

        def _clear(displayed, _selected):
            values = []
            for _ in range(SLOT_COUNT):
                values.extend([True, None, 1.0])
            token = str(int(time.time() * 1000))
            cards = build_cards_html(displayed, preview_map, [], bridge_id, token)
            return values + [""] + [None] * SLOT_COUNT + [cards, [], "選択中 0 件"]

        clear_btn.click(
            fn=_clear,
            inputs=[gallery_state, selected_state],
            outputs=slot_components + [preview] + slot_images + [cards_html, selected_state, gallery_status],
            show_progress=False,
        )

        def _reset_weights(*args):
            values = list(args)
            for idx in range(2, len(values), 3):
                values[idx] = 1.0
            return values + _sync_outputs(values)

        reset_weight_btn.click(
            fn=_reset_weights,
            inputs=slot_components,
            outputs=slot_components + [preview] + slot_images,
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
            try:
                start_slot = int(str(target).replace("枠", "").strip())
            except (TypeError, ValueError):
                start_slot = 1
            start_slot = max(1, min(SLOT_COUNT, start_slot))
            prev_count = len(selected or [])
            new_values, assigned = rewrite_selection_slots(values, start_slot, prev_count, chosen)
            status = selection_status_text(chosen, start_slot, assigned)
            return new_values + _sync_outputs(new_values) + [status, chosen]

        bridge.input(
            fn=_on_bridge,
            inputs=[bridge, target_slot, selected_state] + slot_components,
            outputs=slot_components + [preview] + slot_images + [gallery_status, selected_state],
            show_progress=False,
        )

        return [master_enabled] + slot_components

    # -- Generation ----------------------------------------------------------
    def process(self, p, master_enabled=True, *slot_args):
        if not master_enabled:
            return

        slots = []
        values = list(slot_args)
        for i in range(0, len(values) - 2, 3):
            slots.append((values[i], values[i + 1], values[i + 2]))

        tags = build_lora_tags(slots)
        if not tags:
            return

        new_tags = [tag for tag in tags if tag not in (p.prompt or "")]
        if new_tags:
            p.prompt = ((p.prompt or "").rstrip() + " " + " ".join(new_tags)).strip()

        try:
            if p.extra_generation_params is None:
                p.extra_generation_params = {}
            p.extra_generation_params["Lora Lab"] = " ".join(tags)
        except Exception:
            pass
