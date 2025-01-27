from django import forms

class ExcelUploadForm(forms.Form):
    excel_file = forms.FileField(
        label='Select Excel File',
        help_text='File must contain columns: material, date, and quantity'
    ) 