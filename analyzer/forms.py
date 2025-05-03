from django import forms
from django.core.validators import FileExtensionValidator

class DataUploadForm(forms.Form):
    """Form for uploading data files or pasting CSV content"""
    file = forms.FileField(
        required=False,
        label="Upload a file",
        help_text="Supported formats: CSV, Excel, JSON",
        validators=[
            FileExtensionValidator(allowed_extensions=['csv', 'xlsx', 'json'])
        ]
    )
    csv_text = forms.CharField(
        required=False,
        label="Or paste CSV data",
        widget=forms.Textarea(attrs={'rows': 5, 'placeholder': 'Paste CSV data here...'}),
        help_text="Paste comma-separated values with header row"
    )

    def clean(self):
        cleaned_data = super().clean()
        file = cleaned_data.get('file')
        csv_text = cleaned_data.get('csv_text')

        if not file and not csv_text:
            raise forms.ValidationError("You must either upload a file or paste CSV data")
        
        if file and csv_text:
            raise forms.ValidationError("Please provide either a file or CSV text, not both")
        
        return cleaned_data

class AnalysisForm(forms.Form):
    """Main analysis form with all possible fields"""
    # Hypothesis Testing Fields
    test_type = forms.ChoiceField(
        choices=[
            ('', '-- Select Test --'),
            ('t_test', 'T-Test'),
            ('anova', 'ANOVA'),
            ('chi_square', 'Chi-Square'),
            ('mann_whitney', 'Mann-Whitney U')
        ],
        required=False,
        label="Statistical Test"
    )
    column1 = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Variable 1"
    )
    column2 = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Variable 2"
    )
    group_column = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Group Variable"
    )

    # Visualization Fields
    plot_type = forms.ChoiceField(
        choices=[
            ('', '-- Select Plot --'),
            ('histogram', 'Histogram'),
            ('boxplot', 'Box Plot'),
            ('scatterplot', 'Scatter Plot'),
            ('correlation', 'Correlation Matrix')
        ],
        required=False,
        label="Visualization Type"
    )

    # Regression Fields
    target_column = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Target Variable (Y)"
    )
    predictor_columns = forms.MultipleChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Predictor Variables (X)",
        widget=forms.SelectMultiple(attrs={'class': 'select2'})
    )

    # Logistic Regression Fields
    target_logistic = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Binary Target (Y)"
    )
    predictors_logistic = forms.MultipleChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Predictors (X)",
        widget=forms.SelectMultiple(attrs={'class': 'select2'})
    )

    # ANCOVA Fields
    ancova_outcome = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Outcome Variable"
    )
    ancova_group = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Group Variable"
    )
    ancova_covariate = forms.ChoiceField(
        choices=[],  # Will be populated in view
        required=False,
        label="Covariate"
    )