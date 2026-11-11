"""
build_dashboard.py — Assemble the self-contained dashboard.html.

Inlines the Plotly library and the precomputed data into the template so the result
needs NO network and NO server: it runs offline from a double-click or any static host.

Pipeline:
    python export_data.py     # (re)generate dashboard_data.json from the real pipeline
    python build_dashboard.py # inline plotly + data -> dashboard.html

Inputs : dashboard_template.html, plotly.min.js, dashboard_data.json
Output : dashboard.html
"""
from __future__ import annotations

import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name: str) -> str:
    return io.open(os.path.join(HERE, name), encoding="utf-8").read()


def main() -> None:
    tpl = read("dashboard_template.html")
    plotly = read("plotly.min.js")
    data = read("dashboard_data.json")

    json.loads(data)  # sanity: valid JSON
    for marker in ("/*__PLOTLY__*/", "/*__DATA__*/"):
        if marker not in tpl:
            raise SystemExit(f"template is missing marker {marker}")

    # guard: a literal </script> inside inlined JS would close the tag early
    plotly_safe = plotly.replace("</script>", "<\\/script>")
    data_safe = data.replace("</script>", "<\\/script>")

    # Plotly's string-decoder polyfill embeds 4 literal U+FFFD chars inside string
    # literals; rewrite them as the equivalent JS escape so the file carries no raw
    # replacement character (the artifact publisher rejects U+FFFD).
    plotly_safe = plotly_safe.replace("�", "\\ufffd")

    out = tpl.replace("/*__PLOTLY__*/", plotly_safe).replace("/*__DATA__*/", data_safe)
    # Write both names: dashboard.html for local double-click, index.html so the file
    # also loads at a web host's root (GitHub Pages, Netlify, etc. serve "/" as index.html).
    for name in ("dashboard.html", "index.html"):
        io.open(os.path.join(HERE, name), "w", encoding="utf-8").write(out)

    size_mb = os.path.getsize(os.path.join(HERE, "dashboard.html")) / 1024 / 1024
    print(f"Wrote dashboard.html and index.html  ({size_mb:.2f} MB each, fully self-contained)")


if __name__ == "__main__":
    main()
