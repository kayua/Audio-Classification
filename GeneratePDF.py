import markdown
import pdfkit
import os


def convert_md_to_html(md_file_path, html_file_path):
    # Lê o arquivo Markdown
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Converte Markdown para HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    # Escreve o HTML em um arquivo
    with open(html_file_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)


def convert_html_to_pdf(html_file_path, pdf_file_path):
    # Configure pdfkit para usar wkhtmltopdf
    path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'  # Atualize este caminho conforme necessário
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    # Defina o diretório base para recursos locais
    options = {
        'no-outline': None,
        'enable-local-file-access': None,  # Habilita o acesso a arquivos locais
    }

    # Converte HTML para PDF
    try:
        pdfkit.from_file(html_file_path, pdf_file_path, configuration=config, options=options)
    except Exception as e:
        print(f'Error converting HTML to PDF: {e}')


if __name__ == '__main__':
    # Caminho para o arquivo Markdown
    md_file_path = 'ReadMe.md'

    # Caminho para o arquivo HTML temporário
    html_file_path = 'temp.html'

    # Caminho para o arquivo PDF
    pdf_file_path = 'output.pdf'

    # Converta Markdown para HTML
    convert_md_to_html(md_file_path, html_file_path)

    # Converta HTML para PDF
    convert_html_to_pdf(html_file_path, pdf_file_path)

    # Remove o arquivo HTML temporário
    os.remove(html_file_path)

    print(f'Success: The PDF has been generated at {pdf_file_path}.')
