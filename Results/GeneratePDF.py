import markdown
import pdfkit
import os
import tempfile

def convert_md_to_pdf(md_file_path, pdf_file_path, path_wkhtmltopdf):
    # Read the markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    # Configure pdfkit to use wkhtmltopdf
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_html_file:
        temp_html_file.write(html_content.encode('utf-8'))
        temp_html_file_path = temp_html_file.name

    # Define options to ignore errors
    options = {
        'no-outline': None,
        'disable-smart-shrinking': None,
        'enable-local-file-access': None
    }

    # Convert the HTML file to PDF
    pdfkit.from_file(temp_html_file_path, pdf_file_path, configuration=config, options=options)

    # Remove the temporary HTML file
    os.remove(temp_html_file_path)

if __name__ == '__main__':
    md_file_path = '../ReadMe.md'
    pdf_file_path = '../ReadMe.pdf'

    # Use the path returned by the 'which wkhtmltopdf' command
    path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Atualize este caminho conforme necessário

    if not os.path.exists(md_file_path):
        print(f'Error: The file {md_file_path} does not exist.')
    else:
        convert_md_to_pdf(md_file_path, pdf_file_path, path_wkhtmltopdf)
        print(f'Success: The file has been converted to {pdf_file_path}.')