"""Render the deck sources (report/slides/project) to per-slide HTML, PNG, PDF and PPTX."""
import json, re, subprocess, sys
from pathlib import Path
ROOT = Path("/Users/karthikeyansundaravaradhan/Desktop/Masters Project")
SRC = ROOT / "report/slides/project"
OUT = Path(sys.argv[1])
FIG = ROOT / "results/figures"
BLOB = {"6e7a0c611e96c80e7ab5b579efebda14": "f0_architecture", "97289c9156b36e3f5c54a218ed8e8590": "f9_progression",
        "b5ba95111b3de991b6216bfe3f120dee": "f1_model_comparison", "4649d53743782de47d13f5d9f552c4fa": "f2_confusion_y3",
        "74d729dc5be6c61ac21fb5f82ed91859": "f4_robustness", "ec8c37452c264774c267ab28b9bffd76": "f5_session_confound",
        "262e8dff09a13019a709091b1b5064b5": "f7_timeline_K", "c668d14a83281ad6e667097519add6d7": "f6_subject_dependent"}
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HEAD = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&family=Inter:wght@400;600&display=swap">
<style>
@page { size: 1920px 1080px; margin: 0 }
* { margin: 0; box-sizing: border-box }
html, body { background: #fff }
section { position: relative; width: 1920px; height: 1080px; overflow: hidden; page-break-after: always; break-after: page }
aside { display: none }
table { border-collapse: collapse }
th { font-weight: 600; padding: 14px 16px; border-bottom: 2px solid #1A2233 }
td { padding: 12px 16px; border-bottom: 1px solid #D5D9DF }
b { font-weight: 600 }
</style></head><body>"""
deck = json.loads((SRC / "deck.json").read_text())
slides, notes = [], []
for sid in deck["order"]:
    h = (SRC / "slides" / f"{sid}.html").read_text()
    for b, f in BLOB.items():
        h = h.replace(f"/_blob/{b}", (FIG / f"{f}.png").as_uri())
    assert "/_blob/" not in h, sid
    m = re.search(r"<aside>(.*?)</aside>", h, re.S)
    notes.append(m.group(1).strip() if m else "")
    slides.append(h)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "deck.html").write_text(HEAD + "\n".join(slides) + "</body></html>")
pngs = []
for i, h in enumerate(slides, 1):
    p = OUT / f"s{i:02d}.html"
    p.write_text(HEAD + h + "</body></html>")
    png = OUT / f"slide_{i:02d}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                    "--force-device-scale-factor=1", "--window-size=1920,1080", "--virtual-time-budget=4000",
                    f"--screenshot={png}", p.as_uri()], check=True, capture_output=True)
    pngs.append(png)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--no-pdf-header-footer",
                "--virtual-time-budget=6000", f"--print-to-pdf={OUT / 'deck.pdf'}", (OUT / "deck.html").as_uri()],
               check=True, capture_output=True)
from pptx import Presentation
from pptx.util import Emu
prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
for png, note in zip(pngs, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_picture(str(png), 0, 0, prs.slide_width, prs.slide_height)
    s.notes_slide.notes_text_frame.text = re.sub(r"<[^>]+>", "", note)
prs.save(OUT / "deck.pptx")
print("slides", len(pngs))
