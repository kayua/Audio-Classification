#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2025/04/1'
__last_update__ = '2025/04/1'
__credits__ = ['unknown']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import os
    import sys

    import pdfkit
    import logging
    import argparse
    import markdown

except ImportError as error:
    print(error)
    print("1. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt")
    sys.exit(-1)


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_markdown_to_html(markdown_file_path, html_file_path):
    """
    Converts a Markdown file to HTML format and saves it to a specified file.

    Args:
        markdown_file_path (str): The path to the input Markdown file.
        html_file_path (str): The path to the output HTML file.
    """
    logging.info("Starting conversion from Markdown to HTML.")
    logging.debug("Reading Markdown file from path: %s", markdown_file_path)

    # Read the Markdown file
    with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    logging.debug("Markdown content read successfully. Converting to HTML.")

    # Convert Markdown to HTML
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

    logging.debug("HTML content generated successfully. Writing to file: %s", html_file_path)

    # Write HTML content to a file
    with open(html_file_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)

    logging.info("Conversion from Markdown to HTML completed successfully.")

def convert_html_to_pdf(html_file_path, pdf_file_path):
    """
    Converts an HTML file to PDF format and saves it to a specified file.

    Args:
        html_file_path (str): The path to the input HTML file.
        pdf_file_path (str): The path to the output PDF file.
    """
    logging.info("Starting conversion from HTML to PDF.")
    path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    # Options to set margins and enable local file access
    options = {
        'no-outline': None,
        'enable-local-file-access': None,
        'margin-top': '22mm',
        'margin-bottom': '22mm',
        'margin-left': '19mm',
        'margin-right': '19mm'
    }

    logging.debug("Converting HTML file: %s to PDF file: %s", html_file_path, pdf_file_path)

    try:
        # Convert HTML to PDF
        pdfkit.from_file(html_file_path, pdf_file_path, configuration=config, options=options)
        logging.info("Conversion from HTML to PDF completed successfully.")

    except Exception as e:
        logging.error("Error converting HTML to PDF: %s", e)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF.')
    parser.add_argument('--input', type=str, default='ReadMe.md', help='Path to the input Markdown file.')
    parser.add_argument('--output', type=str, default='output.pdf', help='Path to the output PDF file.')

    # Parse the arguments
    args = parser.parse_args()

    # Paths for input and output files
    md_file_path = args.input
    pdf_file_path = args.output
    html_file_path = 'temp.html'

    logging.info("Input Markdown file path: %s", md_file_path)
    logging.info("Output PDF file path: %s", pdf_file_path)

    # Convert Markdown to HTML
    convert_markdown_to_html(md_file_path, html_file_path)

    # Convert HTML to PDF
    convert_html_to_pdf(html_file_path, pdf_file_path)

    # Remove temporary HTML file
    os.remove(html_file_path)
    logging.info("Temporary HTML file removed.")

    logging.info("Success: The PDF has been generated at %s.", pdf_file_path)
