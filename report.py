"""Downloadable report builders (HTML / JSON / CSV)."""
import json
from datetime import datetime
import pandas as pd


def build_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)


def build_csv(rows: list) -> str:
    return pd.DataFrame(rows).to_csv(index=False)


def build_html(p: dict) -> str:
    def li(items):
        return "".join(f"<li>{i}</li>" for i in items) or "<li>None</li>"
    kv = lambda d: "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items())  # noqa: E731
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>AgriSpectra report</title>
<style>body{{font-family:Segoe UI,Arial;max-width:820px;margin:30px auto;color:#222}}
h1{{color:#2e7d32}}table{{border-collapse:collapse;width:100%;margin:10px 0}}
td{{border:1px solid #ddd;padding:6px 10px}}td:first-child{{background:#f3f8f3;width:40%}}</style></head><body>
<h1>AgriSpectra.AI - Field Report</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M} | File: {p['filename']} | Crop: {p['crop']}</p>
<h2>Summary</h2><table>{kv(p['summary'])}</table>
<h2>Vegetation indices (field mean)</h2><table>{kv(p['indices'])}</table>
<h2>Health zones (% area)</h2><table>{kv(p['zones'])}</table>
<h2>Sensors</h2><table>{kv(p['sensors'])}</table>
<h2>Alerts</h2><ul>{li(p['alerts'])}</ul>
<h2>Recommendations</h2><ul>{li(p['recommendations'])}</ul>
<p style="color:#777;font-size:12px">Decision support only. Verify findings in the field with a qualified agronomist.</p>
</body></html>"""
