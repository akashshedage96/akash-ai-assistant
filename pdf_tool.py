
import os
import re
import uuid
from xml.sax.saxutils import escape

from langchain_core.tools import tool
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

REPORTS_DIR = "reports"


def _format_inline(text: str) -> str:
    escaped = escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def build_pdf(
    title: str,
    content: str,
    sources: list[str] | None = None,
) -> str:

    if not content or not content.strip():
        raise ValueError("PDF content cannot be empty.")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    filename = f"report_{uuid.uuid4().hex[:8]}.pdf"
    path = os.path.abspath(os.path.join(REPORTS_DIR, filename))

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = [
        Paragraph(_format_inline(title), styles["Title"]),
        Spacer(1, 12),
    ]

    for line in content.splitlines():
        line = line.strip()

        if not line:
            story.append(Spacer(1, 6))

        elif line.startswith("#"):
            heading = line.lstrip("# ").strip()
            story.append(
                Paragraph(
                    _format_inline(heading),
                    styles["Heading2"],
                )
            )

        elif line.startswith(("- ", "* ")):
            story.append(
                Paragraph(
                    "&bull; " + _format_inline(line[2:]),
                    styles["BodyText"],
                )
            )

        else:
            story.append(
                Paragraph(
                    _format_inline(line),
                    styles["BodyText"],
                )
            )

    if sources:
        story.extend([
            Spacer(1, 12),
            Paragraph("Sources", styles["Heading2"]),
        ])

        for source in sources:
            story.append(
                Paragraph(
                    "&bull; " + _format_inline(str(source)),
                    styles["BodyText"],
                )
            )

    doc.build(story)
    print("PDF saved at:", path)

    return path


@tool
def create_pdf_report(title: str) -> str:
    return "PDF requested. The graph will generate the report."
