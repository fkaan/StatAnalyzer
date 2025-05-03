from fpdf import FPDF
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO
import re
from bs4 import BeautifulSoup

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Statistical Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(title, data_preview, descriptive_stats, var_types, results, ai_summary):
    """
    Create a PDF report with analysis results
    Returns FPDF object
    """
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, title, 0, 1)
    pdf.ln(5)
    
    # Data Preview
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Data Preview', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Convert HTML table to formatted text
    def html_to_text(html):
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if not table:
            return html
        
        text = []
        for row in table.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            text.append(' | '.join(cells))
        return '\n'.join(text)
    
    # Add formatted tables
    pdf.multi_cell(0, 5, html_to_text(data_preview))
    pdf.ln(5)
    
    # Descriptive Statistics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Descriptive Statistics', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, html_to_text(descriptive_stats))
    pdf.ln(5)
    
    # Variable Types
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Variable Types', 0, 1)
    pdf.set_font('Arial', '', 10)
    for var, type in var_types.items():
        pdf.cell(0, 5, f"{var}: {type}", 0, 1)
    pdf.ln(5)
    
    # Analysis Results
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Analysis Results', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if isinstance(results, dict):
        if 'type' in results:  # Hypothesis Test Results
            pdf.cell(0, 5, f"Test Type: {results['type']}", 0, 1)
            pdf.cell(0, 5, f"Variables: {results['variables']}", 0, 1)
            pdf.cell(0, 5, f"Statistic: {results['statistic']}", 0, 1)
            pdf.cell(0, 5, f"p-value: {results['p_value']}", 0, 1)
            pdf.cell(0, 5, f"Interpretation: {results['interpretation']}", 0, 1)
        elif 'r2' in results:  # Regression Results
            pdf.cell(0, 5, f"R² Score: {results['r2']}", 0, 1)
            pdf.cell(0, 5, "Coefficients:", 0, 1)
            for coeff in results['coefficients']:
                pdf.cell(0, 5, f"{coeff['Variable']}: {coeff['Coefficient']}", 0, 1)
            pdf.cell(0, 5, f"Intercept: {results['intercept']}", 0, 1)
            pdf.cell(0, 5, "VIF Scores:", 0, 1)
            for vif in results['vif']:
                pdf.cell(0, 5, f"{vif['Variable']}: {vif['VIF']}", 0, 1)
        elif 'formula' in results:  # ANCOVA Results
            pdf.cell(0, 5, f"Model Formula: {results['formula']}", 0, 1)
            pdf.cell(0, 5, "ANCOVA Results:", 0, 1)
            for row in results['results']:
                pdf.cell(0, 5, f"{row['Source']}: F={row['F']}, p={row['p_value']}", 0, 1)
    elif isinstance(results, str):  # Visualization Results
        pdf.cell(0, 5, "Visualization included in the report", 0, 1)
    
    pdf.ln(10)
    
    # AI Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'AI Analysis Summary', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, ai_summary)
    
    return pdf

def add_df_to_pdf(pdf, df):
    """Helper function to add pandas DataFrame to PDF"""
    pdf.set_font('Arial', 'B', 10)
    
    # Add headers
    col_widths = [40, 30, 30]  # Adjust based on your needs
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 7, str(col), border=1)
    pdf.ln()
    
    # Add data rows
    pdf.set_font('Arial', '', 10)
    for _, row in df.iterrows():
        for i, col in enumerate(df.columns):
            pdf.cell(col_widths[i], 6, str(row[col]), border=1)
        pdf.ln()