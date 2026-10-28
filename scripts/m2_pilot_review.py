#!/usr/bin/env python3
"""Render pilot variants for model-independent answerability review.

Each (image, corruption) panel is rendered twice and compared byte for byte.
The output directory receives PNG panels plus an index.html page where a human
reviewer unticks questions that are no longer visually answerable and downloads
a decisions JSONL for `mpvs m2-validate-answerability`. No model is used.
Outputs contain dataset images: keep them under the ignored artifacts/ tree.
"""
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from mpvs.corruptions import VariantRecord, apply_corruption

KIND_ORDER = {"clean": 0, "darkness": 1, "gaussian_blur": 2}

SAVE_SCRIPT = """
function save(){const who=document.getElementById('who').value.trim();
if(!who){alert('Enter a review method first');return;}const lines=[];
document.querySelectorAll('input[data-variant]').forEach(b=>{lines.push(JSON.stringify({
schema_version:1,variant_id:b.dataset.variant,answerable:b.checked,review_method:who,
reason:b.checked?null:'not visually answerable after corruption'}));});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],
{type:'application/jsonl'}));a.download='pilot-answerability.jsonl';a.click();
document.getElementById('n').textContent=lines.length+' decisions saved';}
"""

STYLE = """
body{font:14px system-ui;margin:16px;background:#fafafa;color:#222}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
figure{margin:0;background:#fff;border:1px solid #ddd;padding:8px;border-radius:6px}
img{width:100%}label,figcaption div{display:block;margin:3px 0}.lab{color:#888}
header{position:sticky;top:0;background:#fff;padding:10px;border-bottom:1px solid #ccc;z-index:1}
"""


def _question_html(variant: VariantRecord, reviewable: bool) -> str:
    record = variant.assigned.record
    text = f'{html.escape(record.question)} <span class="lab">(label: {record.label})</span>'
    if not reviewable:
        return f"<div>{text}</div>"
    return (
        f'<label><input type="checkbox" data-variant="{html.escape(variant.variant_id)}" '
        f"checked> {text}</label>"
    )


def render(variants_path: Path, image_root: Path, output_dir: Path) -> dict:
    variants = [
        VariantRecord.from_dict(json.loads(line))
        for line in variants_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_pilot = sorted({v.variant_id for v in variants if v.assigned.split != "pilot"})
    if non_pilot:
        raise ValueError(f"answerability review is pilot-only; found {len(non_pilot)} others")
    panels: dict[tuple[str, str, str], list[VariantRecord]] = defaultdict(list)
    for variant in variants:
        spec = variant.corruption
        panels[(variant.assigned.record.original_image_id, spec.kind, spec.severity_id)].append(
            variant
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    mismatches = 0
    by_image: dict[str, list[str]] = defaultdict(list)
    for (image_id, kind, severity_id), group in sorted(
        panels.items(), key=lambda item: (item[0][0], KIND_ORDER[item[0][1]], item[0][2])
    ):
        source = image_root / group[0].assigned.record.image_relpath
        first = apply_corruption(Image.open(source), group[0].corruption)
        second = apply_corruption(Image.open(source), group[0].corruption)
        mismatches += first.tobytes() != second.tobytes()
        name = f"{image_id}__{kind}__{severity_id}.png".replace("/", "_")
        first.save(output_dir / name)
        value = group[0].corruption.value
        caption = f"<b>{kind} · {severity_id}</b>" + ("" if value is None else f" (value {value})")
        questions = "".join(
            _question_html(variant, kind != "clean")
            for variant in sorted(group, key=lambda v: v.assigned.record.question_id)
        )
        by_image[image_id].append(
            f'<figure><img src="{html.escape(name)}" loading="lazy">'
            f"<figcaption>{caption}{questions}</figcaption></figure>"
        )
    sections = "".join(
        f'<section><h2>{html.escape(image_id)}</h2><div class="grid">{"".join(figures)}</div></section>'
        for image_id, figures in sorted(by_image.items())
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Pilot answerability review</title>'
        f"<style>{STYLE}</style></head><body><header><b>Pilot answerability review.</b> "
        "Untick a question if a person can no longer tell the answer from the corrupted image. "
        "Judge by eye only, never by model output. Review method: "
        '<input id="who" size="34" placeholder="e.g. NAME visual review"> '
        '<button onclick="save()">Download decisions .jsonl</button> <span id="n"></span>'
        f"</header>{sections}<script>{SAVE_SCRIPT}</script></body></html>"
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return {
        "variant_count": len(variants),
        "rendered_panels": len(panels),
        "pixel_mismatches_between_two_renders": mismatches,
        "review_page": str(output_dir / "index.html"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.variants, args.image_root, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pixel_mismatches_between_two_renders"] == 0 else 6


if __name__ == "__main__":
    raise SystemExit(main())
