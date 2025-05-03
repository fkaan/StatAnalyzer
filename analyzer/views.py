import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.urls import reverse
from .forms import DataUploadForm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from io import StringIO
from scipy import stats
import plotly.express as px
from .utils.hypo_func import *
from .utils.plot_func import *
from .utils.ai_summary import ai_summary
from .utils.pdf_download import create_pdf_report

def detect_variable_type(series):
    """Detect variable type (Metric, Ordinal, Nominal)"""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() < 10 and series.dtype == 'int64':
            return "Ordinal"
        return "Metric"
    return "Nominal"

def index(request):
    """Handle data upload and initial processing"""
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                if form.cleaned_data['file']:
                    file = form.cleaned_data['file']
                    if file.name.endswith('.csv'):
                        data = pd.read_csv(file)
                    elif file.name.endswith('.xlsx'):
                        data = pd.read_excel(file)
                    elif file.name.endswith('.json'):
                        data = pd.read_json(file)
                elif form.cleaned_data['csv_text']:
                    data = pd.read_csv(StringIO(form.cleaned_data['csv_text']))

                if data is not None and not data.empty:
                    data = data.dropna()
                    if not data.empty:
                        request.session['data'] = data.to_json(orient='split')
                        return redirect('analyzer:analysis')
                    else:
                        form.add_error(None, "Data contains no valid rows after cleaning")
                else:
                    form.add_error(None, "Uploaded file contains no data")
            except Exception as e:
                form.add_error(None, f"Error processing data: {str(e)}")
    else:
        form = DataUploadForm()

    return render(request, 'analyzer/index.html', {'form': form})

def analysis(request):
    """Main analysis view with all statistical functions"""
    if 'data' not in request.session:
        messages.error(request, "Please upload your data first")
        return redirect('analyzer:index')

    try:
        data = pd.read_json(StringIO(request.session['data']), orient='split')
        var_types = {col: detect_variable_type(data[col]) for col in data.columns}
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        categorical_cols = [col for col in data.columns if var_types.get(col) in ["Nominal", "Ordinal"]]
        binary_cols = [col for col in numeric_cols if data[col].nunique() == 2]

        # Initialize context with all possible variables
        context = {
            'data_preview': data.head().to_html(classes='table table-striped'),
            'descriptive_stats': data.describe().to_html(classes='table table-striped'),
            'var_types': var_types,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'binary_cols': binary_cols,
            'show_test': False,
            'show_plot': False,
            'show_regression': False,
            'show_logistic': False,
            'show_ancova': False
        }

        if request.method == 'POST':
            # AI Insights
            if 'ai_summary' in request.POST:
                try:
                    # Generate prompt for AI based on the current analysis
                    prompt = f"""
                    Analyze the following dataset and provide insights:
                    {data.describe().to_string()}
                    
                    Variable types:
                    {var_types}
                    """
                    
                    # Add analysis-specific context
                    if 'test_results' in request.session:
                        test_results = request.session['test_results']
                        prompt += f"""
                        
                        Hypothesis Test Results:
                        Type: {test_results['type']}
                        Variables: {test_results['variables']}
                        Statistic: {test_results['statistic']}
                        p-value: {test_results['p_value']}
                        Interpretation: {test_results['interpretation']}
                        
                        Please provide detailed insights about:
                        1. The statistical significance of these results
                        2. Practical implications of the findings
                        3. Potential limitations of the test
                        4. Recommendations for further analysis
                        """
                    
                    elif 'regression_results' in request.session:
                        reg_results = request.session['regression_results']
                        prompt += f"""
                        
                        Regression Analysis Results:
                        R² Score: {reg_results['r2']}
                        Coefficients:
                        {reg_results['coefficients']}
                        VIF Scores:
                        {reg_results['vif']}
                        
                        Please provide detailed insights about:
                        1. The model's overall fit and predictive power
                        2. The importance and significance of each predictor
                        3. Potential multicollinearity issues
                        4. Recommendations for model improvement
                        """
                    
                    elif 'plot_html' in request.session:
                        prompt += f"""
                        
                        Visualization Results:
                        Plot Type: {request.session.get('plot_type', 'Unknown')}
                        Variables: {request.session.get('plot_variables', 'Unknown')}
                        
                        Please provide detailed insights about:
                        1. Key patterns and trends observed
                        2. Distribution characteristics
                        3. Potential outliers or anomalies
                        4. Relationships between variables
                        5. Recommendations for further visualization
                        """
                    
                    elif 'ancova_results' in request.session:
                        ancova_results = request.session['ancova_results']
                        prompt += f"""
                        
                        ANCOVA Results:
                        Model Formula: {ancova_results['formula']}
                        Results:
                        {ancova_results['results']}
                        
                        Please provide detailed insights about:
                        1. Group differences while controlling for the covariate
                        2. The impact of the covariate
                        3. Practical significance of the findings
                        4. Potential limitations of the analysis
                        5. Recommendations for further investigation
                        """
                    
                    # Get AI summary
                    summary = ai_summary(prompt)
                    context.update({
                        'ai_summary': summary
                    })
                    request.session['ai_summary'] = summary
                except Exception as e:
                    messages.error(request, f"AI insights generation failed: {str(e)}")

            # Hypothesis Testing
            elif 'run_test' in request.POST:
                test_type = request.POST.get('test_type')
                col1 = request.POST.get('column1')
                col2 = request.POST.get('column2')
                group_col = request.POST.get('group_column')

                try:
                    if test_type == 't_test':
                        if col1 not in numeric_cols or col2 not in numeric_cols:
                            raise ValueError("T-Test requires numeric variables")
                        t_stat, p_val = t_test(data, col1, col2)
                        context.update({
                            'test_results': {
                                'type': 'T-Test',
                                'variables': f"{col1} vs {col2}",
                                'statistic': f"{t_stat:.4f}",
                                'p_value': f"{p_val:.4f}",
                                'interpretation': "Significant difference" if p_val < 0.05 else "No significant difference"
                            },
                            'show_test': True
                        })
                        request.session['test_results'] = context['test_results']
                    elif test_type == 'anova':
                        if col1 not in numeric_cols or group_col not in categorical_cols:
                            raise ValueError("ANOVA requires a numeric variable and a categorical group variable")
                        f_stat, p_val = anova_test(data, col1, group_col)
                        context.update({
                            'test_results': {
                                'type': 'ANOVA',
                                'variables': f"{col1} by {group_col}",
                                'statistic': f"{f_stat:.4f}",
                                'p_value': f"{p_val:.4f}",
                                'interpretation': "Significant group differences" if p_val < 0.05 else "No significant differences"
                            },
                            'show_test': True
                        })
                        request.session['test_results'] = context['test_results']
                    elif test_type == 'chi_square':
                        if col1 not in categorical_cols or col2 not in categorical_cols:
                            raise ValueError("Chi-Square test requires categorical variables")
                        chi2_stat, p_val = chi_square_test(data, col1, col2)
                        context.update({
                            'test_results': {
                                'type': 'Chi-Square Test',
                                'variables': f"{col1} vs {col2}",
                                'statistic': f"{chi2_stat:.4f}",
                                'p_value': f"{p_val:.4f}",
                                'interpretation': "Significant association" if p_val < 0.05 else "No significant association"
                            },
                            'show_test': True
                        })
                        request.session['test_results'] = context['test_results']
                    elif test_type == 'mann_whitney':
                        if col1 not in numeric_cols or col2 not in numeric_cols:
                            raise ValueError("Mann-Whitney U test requires numeric variables")
                        u_stat, p_val = mann_whitney_u(data, col1, col2)
                        context.update({
                            'test_results': {
                                'type': 'Mann-Whitney U Test',
                                'variables': f"{col1} vs {col2}",
                                'statistic': f"{u_stat:.4f}",
                                'p_value': f"{p_val:.4f}",
                                'interpretation': "Significant difference" if p_val < 0.05 else "No significant difference"
                            },
                            'show_test': True
                        })
                        request.session['test_results'] = context['test_results']

                except Exception as e:
                    messages.error(request, f"Test failed: {str(e)}")

            # Visualization
            elif 'generate_plot' in request.POST:
                plot_type = request.POST.get('plot_type')
                col1 = request.POST.get('column1')
                col2 = request.POST.get('column2')
                group_col = request.POST.get('group_column')

                try:
                    if plot_type == 'histogram':
                        if col1 not in numeric_cols:
                            raise ValueError("Histogram requires a numeric variable")
                        plot_html = plot_histogram(data, col1)
                        request.session['plot_type'] = 'Histogram'
                        request.session['plot_variables'] = col1
                    elif plot_type == 'boxplot':
                        if col1 not in numeric_cols:
                            raise ValueError("Boxplot requires a numeric variable")
                        if group_col and group_col not in categorical_cols:
                            raise ValueError("Group variable must be categorical")
                        plot_html = plot_boxplot(data, col1, group_col)
                        request.session['plot_type'] = 'Boxplot'
                        request.session['plot_variables'] = f"{col1} by {group_col}"
                    elif plot_type == 'scatterplot':
                        if col1 not in numeric_cols or col2 not in numeric_cols:
                            raise ValueError("Scatterplot requires two numeric variables")
                        plot_html = plot_scatter(data, col1, col2)
                        request.session['plot_type'] = 'Scatter Plot'
                        request.session['plot_variables'] = f"{col1} vs {col2}"
                    elif plot_type == 'correlation':
                        if len(numeric_cols) < 2:
                            raise ValueError("Correlation matrix requires at least two numeric variables")
                        plot_html = plot_correlation_matrix(data)
                        request.session['plot_type'] = 'Correlation Matrix'
                        request.session['plot_variables'] = 'All numeric variables'
                    
                    context.update({
                        'plot_html': plot_html,
                        'show_plot': True
                    })
                    request.session['plot_html'] = plot_html

                except Exception as e:
                    messages.error(request, f"Plot failed: {str(e)}")

            # Linear Regression
            elif 'run_regression' in request.POST:
                target = request.POST.get('target_column')
                predictors = request.POST.getlist('predictor_columns')

                try:
                    X = data[predictors]
                    y = data[target]
                    
                    # Add constant for statsmodels
                    X_with_const = sm.add_constant(X)
                    
                    # Fit model using statsmodels for detailed statistics
                    model = sm.OLS(y, X_with_const).fit()
                    
                    # Calculate VIF
                    vif = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
                    
                    # Format coefficients and VIF for display
                    coefficients = []
                    for i, var in enumerate(['Intercept'] + predictors):
                        coefficients.append({
                            'Variable': var,
                            'Coefficient': f"{model.params[i]:.4f}",
                            'Std_Error': f"{model.bse[i]:.4f}",
                            't_value': f"{model.tvalues[i]:.4f}",
                            'p_value': f"{model.pvalues[i]:.4f}"
                        })
                    
                    vif_results = [{'Variable': var, 'VIF': f"{v:.2f}"} 
                                  for var, v in zip(['Intercept'] + predictors, vif)]
                    
                    # Calculate residuals statistics
                    residuals = model.resid
                    residual_stats = {
                        'mean': f"{residuals.mean():.4f}",
                        'std': f"{residuals.std():.4f}",
                        'skew': f"{stats.skew(residuals):.4f}",
                        'kurtosis': f"{stats.kurtosis(residuals):.4f}"
                    }
                    
                    # Get ANOVA table
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    anova_results = []
                    for i, row in anova_table.iterrows():
                        anova_results.append({
                            'Source': i,
                            'Sum_Sq': f"{row['sum_sq']:.4f}",
                            'df': f"{row['df']:.0f}",
                            'F': f"{row['F']:.4f}",
                            'p_value': f"{row['PR(>F)']:.4f}"
                        })
                    
                    context.update({
                        'regression_results': {
                            'r2': f"{model.rsquared:.4f}",
                            'adjusted_r2': f"{model.rsquared_adj:.4f}",
                            'f_statistic': f"{model.fvalue:.4f}",
                            'p_value': f"{model.f_pvalue:.4f}",
                            'coefficients': coefficients,
                            'vif': vif_results,
                            'residuals': residual_stats,
                            'anova': anova_results,
                            'target': target,
                            'predictors': predictors
                        },
                        'show_regression': True
                    })
                    
                    # Generate AI interpretation
                    prompt = f"""
                    Analyze the following linear regression results:
                    R²: {model.rsquared:.4f}
                    Adjusted R²: {model.rsquared_adj:.4f}
                    F-statistic: {model.fvalue:.4f}
                    p-value: {model.f_pvalue:.4f}
                    
                    Coefficients:
                    {coefficients}
                    
                    ANOVA:
                    {anova_results}
                    
                    Residual Statistics:
                    {residual_stats}
                    
                    Please provide:
                    1. Overall model fit assessment
                    2. Significance of predictors
                    3. Interpretation of coefficients
                    4. Residual analysis
                    5. Recommendations for model improvement
                    """
                    
                    context['regression_results']['ai_interpretation'] = ai_summary(prompt)
                    
                    # Store results in session for PDF download
                    request.session['regression_results'] = context['regression_results']

                except Exception as e:
                    messages.error(request, f"Regression failed: {str(e)}")

            # Logistic Regression
            elif 'run_logistic' in request.POST:
                target = request.POST.get('target_logistic')
                predictors = request.POST.getlist('predictors_logistic')

                try:
                    X = data[predictors]
                    y = data[target]
                    
                    # Add constant for statsmodels
                    X_with_const = sm.add_constant(X)
                    
                    # Fit model using statsmodels for detailed statistics
                    model = sm.Logit(y, X_with_const).fit()
                    
                    # Calculate predictions and confusion matrix
                    y_pred = (model.predict(X_with_const) > 0.5).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                    
                    # Calculate performance metrics
                    accuracy = accuracy_score(y, y_pred)
                    precision = precision_score(y, y_pred)
                    recall = recall_score(y, y_pred)
                    f1 = f1_score(y, y_pred)
                    
                    # Format coefficients
                    coefficients = []
                    for i, var in enumerate(['Intercept'] + predictors):
                        coefficients.append({
                            'Variable': var,
                            'Coefficient': f"{model.params[i]:.4f}",
                            'Std_Error': f"{model.bse[i]:.4f}",
                            'z_value': f"{model.tvalues[i]:.4f}",
                            'p_value': f"{model.pvalues[i]:.4f}",
                            'Odds_Ratio': f"{np.exp(model.params[i]):.4f}"
                        })
                    
                    # Calculate chi-square test
                    chi2_stat = model.llr
                    chi2_pval = model.llr_pval
                    
                    context.update({
                        'logistic_results': {
                            'accuracy': f"{accuracy:.4f}",
                            'precision': f"{precision:.4f}",
                            'recall': f"{recall:.4f}",
                            'f1_score': f"{f1:.4f}",
                            'classification': {
                                'true_negatives': tn,
                                'false_positives': fp,
                                'false_negatives': fn,
                                'true_positives': tp
                            },
                            'log_likelihood': f"{model.llf:.4f}",
                            'aic': f"{model.aic:.4f}",
                            'bic': f"{model.bic:.4f}",
                            'chi_square': {
                                'statistic': f"{chi2_stat:.4f}",
                                'p_value': f"{chi2_pval:.4f}",
                                'df': len(predictors)
                            },
                            'coefficients': coefficients,
                            'target': target,
                            'predictors': predictors
                        },
                        'show_logistic': True
                    })
                    
                    # Generate AI interpretation
                    prompt = f"""
                    Analyze the following logistic regression results:
                    Accuracy: {accuracy:.4f}
                    Precision: {precision:.4f}
                    Recall: {recall:.4f}
                    F1 Score: {f1:.4f}
                    
                    Classification Table:
                    True Negatives: {tn}
                    False Positives: {fp}
                    False Negatives: {fn}
                    True Positives: {tp}
                    
                    Model Statistics:
                    Log-Likelihood: {model.llf:.4f}
                    AIC: {model.aic:.4f}
                    BIC: {model.bic:.4f}
                    
                    Chi-Square Test:
                    Statistic: {chi2_stat:.4f}
                    p-value: {chi2_pval:.4f}
                    
                    Coefficients:
                    {coefficients}
                    
                    Please provide:
                    1. Overall model performance assessment
                    2. Interpretation of classification results
                    3. Significance of predictors
                    4. Odds ratio interpretation
                    5. Recommendations for model improvement
                    """
                    
                    context['logistic_results']['ai_interpretation'] = ai_summary(prompt)
                    
                    # Store results in session for PDF download
                    request.session['logistic_results'] = context['logistic_results']

                except Exception as e:
                    messages.error(request, f"Logistic regression failed: {str(e)}")

            # ANCOVA
            elif 'run_ancova' in request.POST:
                outcome = request.POST.get('ancova_outcome')
                group = request.POST.get('ancova_group')
                covariate = request.POST.get('ancova_covariate')

                try:
                    formula = f"{outcome} ~ C({group}) + {covariate}"
                    model = sm.OLS.from_formula(formula, data).fit()
                    ancova_table = sm.stats.anova_lm(model, typ=2)
                    
                    # Format ANCOVA results for display
                    results = []
                    for i, row in ancova_table.iterrows():
                        results.append({
                            'Source': i,
                            'Sum_of_Squares': f"{row['sum_sq']:.4f}",
                            'df': f"{row['df']:.0f}",
                            'F': f"{row['F']:.4f}",
                            'p_value': f"{row['PR(>F)']:.4f}"
                        })
                    
                    context.update({
                        'ancova_results': {
                            'formula': formula,
                            'results': results,
                            'outcome': outcome,
                            'group': group,
                            'covariate': covariate
                        },
                        'show_ancova': True
                    })

                except Exception as e:
                    messages.error(request, f"ANCOVA failed: {str(e)}")

        return render(request, 'analyzer/analysis.html', context)

    except Exception as e:
        messages.error(request, f"Analysis error: {str(e)}")
        return redirect('analyzer:index')

def download_report(request):
    """Generate PDF report of analysis results"""
    if 'data' not in request.session:
        messages.warning(request, "No data available")
        return redirect('analyzer:analysis')

    try:
        data = pd.read_json(StringIO(request.session['data']), orient='split')
        var_types = {col: detect_variable_type(data[col]) for col in data.columns}
        
        # Get the current analysis results
        results = {}
        analysis_type = "Analysis"
        
        if 'regression_results' in request.session:
            results = request.session['regression_results']
            analysis_type = 'Regression'
        elif 'test_results' in request.session:
            results = request.session['test_results']
            analysis_type = results.get('type', 'Hypothesis Test')
        elif 'plot_html' in request.session:
            results = request.session['plot_html']
            analysis_type = 'Visualization'
        elif 'ancova_results' in request.session:
            results = request.session['ancova_results']
            analysis_type = 'ANCOVA'
        else:
            messages.warning(request, "No analysis results to download")
            return redirect('analyzer:analysis')
        
        # Create PDF report
        pdf = create_pdf_report(
            title=f"{analysis_type} Analysis Report",
            data_preview=data.head().to_html(),
            descriptive_stats=data.describe().to_html(),
            var_types=var_types,
            results=results,
            ai_summary=request.session.get('ai_summary', 'No AI summary available')
        )
        
        # Save PDF to a temporary file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf.output(tmp.name)
            tmp_path = tmp.name
        
        # Read the temporary file and create response
        with open(tmp_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="analysis_report.pdf"'
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return response

    except Exception as e:
        messages.error(request, f"Report generation failed: {str(e)}")
        return redirect('analyzer:analysis')