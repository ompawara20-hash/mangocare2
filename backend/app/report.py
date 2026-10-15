"""PDF report from STORED analysis metadata + knowledge base. No image is embedded (images are not retained)."""
import io
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

GREEN = colors.HexColor("#1b4d2e")
STATUS_TEXT = {
    "healthy_appearance": "Healthy appearance detected (not a guarantee)",
    "possible_disease": "Possible disease detected",
    "uncertain": "Uncertain result - another photograph is recommended",
}
LIMITS = ["Preliminary, image-based screening for educational use only; not a laboratory diagnosis.",
          "The model is trained for mango LEAF disease classes and may be unreliable on other plant parts or real-world conditions.",
          "Model scores are not guaranteed probabilities of correctness; no accuracy metrics have been independently verified.",
          "Consult a local agricultural expert for confirmation and treatment decisions."]


def _footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"MangoCare AI - Page {doc.page}")


def build_pdf(rec: dict, disease: dict | None) -> bytes:
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Heading2"], textColor=GREEN)
    body = ss["BodyText"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                            title="MangoCare AI Assessment Report")
    s = [Paragraph("MangoCare AI - Mango Leaf Health Assessment", ParagraphStyle("t", parent=ss["Title"], textColor=GREEN)),
         Paragraph("Preliminary screening report", body), Spacer(1, 6)]
    conf = f"{rec['confidence'] * 100:.1f}%" if rec["confidence"] is not None else "n/a"
    rows = [["Analysis ID", rec["id"]], ["Date (UTC)", rec["created_at"]],
            ["Result", STATUS_TEXT.get(rec["status"], rec["status"])],
            ["Top class", rec["class_name"] or "n/a"],
            ["Model score", f"{conf} (model score, not a guaranteed probability)"],
            ["Mango check score", f"{rec['validation_score']:.2f} (CLIP zero-shot, preliminary)"],
            ["Model", f"{rec['model_name']} {rec['model_version']}"]]
    t = Table([[Paragraph(escape(str(a)), body), Paragraph(escape(str(b)), body)] for a, b in rows], colWidths=[40 * mm, 130 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f3ea"))]))
    s += [t, Spacer(1, 8)]
    if rec["alternatives"]:
        s.append(Paragraph("Other model scores", h))
        s.append(Paragraph("; ".join(f"{escape(a['class_name'])}: {a['score'] * 100:.1f}%" for a in rec["alternatives"]), body))
    if disease and rec["status"] != "uncertain":
        s.append(Paragraph(escape(disease["name"]), h))
        s.append(Paragraph(escape(disease["description"]), body))
        for title, key in [("Visible symptoms", "symptoms"), ("Precautions", "precautions"), ("Recommended next steps", "next_steps")]:
            if disease.get(key):
                s.append(Paragraph(title, h))
                s += [Paragraph("- " + escape(x), body) for x in disease[key]]
    else:
        s.append(Paragraph("The result was uncertain. Please retake a clear, well-lit photograph of the affected leaf.", body))
    s.append(Paragraph("Limitations", h))
    s += [Paragraph("- " + x, body) for x in LIMITS]
    doc.build(s, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
