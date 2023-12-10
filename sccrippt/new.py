import tabula
import pandas as pd

# Path to the PDF file
pdf_path = '"C:/Users/admin/Downloads/fl.pdf"'

# Extract tables from the PDF
# This returns a list of DataFrames
tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

# Loop through the tables and save them as CSV files
for i, table in enumerate(tables):
    csv_path = f'table_{i}.csv'
    table.to_csv(csv_path, index=False)
