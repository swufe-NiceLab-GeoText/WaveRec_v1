import img2pdf


# Convert single image to PDF
def png_to_pdf(input_path, output_path):
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(input_path))

# Usage example
png_to_pdf("ChengDu.png", "chengdu.pdf")
