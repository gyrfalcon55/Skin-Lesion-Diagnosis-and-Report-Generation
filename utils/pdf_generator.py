from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime


def generate_pdf(file_path, title, content):
    """
    Generate a simple PDF report.

    Parameters:
    file_path : Path where the PDF will be saved
    title     : Title of the report
    content   : Text content of the report
    """

    # Create PDF canvas
    pdf = canvas.Canvas(file_path, pagesize=A4)
    page_width, page_height = A4

    # -------------------------------
    # Title
    # -------------------------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, page_height - 50, title)

    # -------------------------------
    # Timestamp
    # -------------------------------
    pdf.setFont("Helvetica", 10)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.drawString(50, page_height - 80, f"Generated on: {timestamp}")

    # -------------------------------
    # Body text
    # -------------------------------
    y_position = page_height - 120
    pdf.setFont("Helvetica", 11)

    for line in content.split("\n"):
        # Create a new page if space is insufficient
        if y_position < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y_position = page_height - 50

        pdf.drawString(50, y_position, line)
        y_position -= 15

    # Save the PDF
    pdf.save()
