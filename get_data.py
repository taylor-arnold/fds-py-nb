import io
import os

import openpyxl
import polars as pl
import requests

SPREADSHEET_ID = "1CVHofm5ukU3CRh2-jvJdu_t4jPXRPFryOTZIo-lWN6E"
EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=xlsx"

os.makedirs("data", exist_ok=True)

response = requests.get(EXPORT_URL)
response.raise_for_status()

xlsx_bytes = io.BytesIO(response.content)

wb = openpyxl.load_workbook(xlsx_bytes, read_only=True)
sheet_names = wb.sheetnames
wb.close()

for sheet_name in sheet_names:
    xlsx_bytes.seek(0)
    df = pl.read_excel(xlsx_bytes, sheet_name=sheet_name)
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))
    output_path = os.path.join("data", f"{sheet_name}.csv")
    df.write_csv(output_path)
    print(f"Saved {output_path}")
