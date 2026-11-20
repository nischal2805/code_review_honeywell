from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'samples'


def write_pdf(path: Path, title: str, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 54
    pdf.setTitle(title)
    pdf.setFont('Helvetica-Bold', 14)
    pdf.drawString(42, y, title)
    y -= 28
    pdf.setFont('Helvetica', 9)
    for line in lines:
        if y < 48:
            pdf.showPage()
            pdf.setFont('Helvetica', 9)
            y = height - 48
        pdf.drawString(42, y, line)
        y -= 15
    pdf.save()


HEADER = [
    'Rule ID | Rule | DAL A | DAL B | DAL C',
    'SCS-L-01 Dynamic Memory new delete malloc free | FORBIDDEN | FORBIDDEN | RESTRICTED',
    'SCS-L-02 goto Statement | FORBIDDEN | FORBIDDEN | FORBIDDEN',
    'SCS-L-03 Exceptions try catch throw | FORBIDDEN | FORBIDDEN | RESTRICTED',
    'SCS-L-04 Recursion | FORBIDDEN | RESTRICTED | ALLOWED',
    'SCS-L-05 RTTI dynamic_cast typeid | FORBIDDEN | FORBIDDEN | RESTRICTED',
    'SCS-M-01 Cyclomatic Complexity (McCabe) | <= 5 | <= 8 | <= 12',
    'SCS-M-02 Function Length (executable lines of code) | <= 20 | <= 35 | <= 60',
    'SCS-M-03 Control Flow Nesting Depth | <= 3 | <= 4 | <= 5',
    'SCS-M-04 Number of Function Parameters | <= 4 | <= 6 | <= 8',
]

write_pdf(OUTPUT / 'coding_standards_strict.pdf', 'Project Phoenix Strict Coding Standards', HEADER + [
    'All functions require a documentation comment.',
])
write_pdf(OUTPUT / 'coding_standards_relaxed.pdf', 'Project Phoenix Relaxed Coding Standards', [
    'Rule ID | Rule | DAL A | DAL B | DAL C',
    'SCS-L-01 Dynamic Memory new delete malloc free | FORBIDDEN | RESTRICTED | ALLOWED',
    'SCS-L-02 goto Statement | FORBIDDEN | FORBIDDEN | RESTRICTED',
    'SCS-L-03 Exceptions try catch throw | FORBIDDEN | RESTRICTED | ALLOWED',
    'SCS-L-04 Recursion | FORBIDDEN | RESTRICTED | ALLOWED',
    'SCS-L-05 RTTI dynamic_cast typeid | FORBIDDEN | RESTRICTED | ALLOWED',
    'SCS-M-01 Cyclomatic Complexity (McCabe) | <= 10 | <= 15 | <= 20',
    'SCS-M-02 Function Length (executable lines of code) | <= 50 | <= 75 | <= 100',
    'SCS-M-03 Control Flow Nesting Depth | <= 5 | <= 6 | <= 7',
    'SCS-M-04 Number of Function Parameters | <= 7 | <= 8 | <= 9',
    'Documentation comments are recommended for every function.',
])
