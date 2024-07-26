import argparse

import markdown
import pdfkit
import os


def convert_md_to_html(md_file_path, html_file_path):
    # Read the markdown file
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    # Add CSS for image centering
    html_content = f"""
    <html>
    <head>
        <style>
            img {{
                display: block;
                margin-left: auto;
                margin-right: auto;
                max-width: 100%;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Write HTML content to a file
    with open(html_file_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)


def convert_html_to_pdf(html_file_path, pdf_file_path):
    # Path to wkhtmltopdf
    path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    # Options to set margins and enable local file access
    options = {
        'no-outline': None,
        'enable-local-file-access': None,
        'margin-top': '22mm',  # Margin superior
        'margin-bottom': '22mm',  # Margin inferior
        'margin-left': '19mm',
        'margin-right': '19mm'
    }

    try:
        # Convert HTML to PDF
        pdfkit.from_file(html_file_path, pdf_file_path, configuration=config, options=options)
    except Exception as e:
        print(f'Error converting HTML to PDF: {e}')


if __name__ == '__main__':
    # Criação do parser de argumentos
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF.')
    parser.add_argument('--input', type=str, default='ReadMe.md', help='Path to the input Markdown file.')
    parser.add_argument('--output', type=str, default='output.pdf', help='Path to the output PDF file.')

    # Parse dos argumentos
    args = parser.parse_args()

    # Paths para arquivos de entrada e saída
    md_file_path = args.input
    pdf_file_path = args.output
    html_file_path = 'temp.html'

    # Convert Markdown to HTML
    convert_md_to_html(md_file_path, html_file_path)

    # Convert HTML to PDF
    convert_html_to_pdf(html_file_path, pdf_file_path)

    # Remove arquivo temporário HTML
    os.remove(html_file_path)

    print(f'Success: The PDF has been generated at {pdf_file_path}.')
