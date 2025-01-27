from django.shortcuts import render, redirect
from django.views.generic import ListView, CreateView
from django.contrib import messages
from .models import SalesData
from .forms import ExcelUploadForm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def log_info(message):
    """Helper function to log messages to both console and file"""
    print(message, flush=True)
    logger.info(message)
    sys.stdout.flush()

class SalesListView(ListView):
    model = SalesData
    template_name = 'forecasting/sales_list.html'
    context_object_name = 'sales'

class SalesCreateView(CreateView):
    model = SalesData
    template_name = 'forecasting/sales_form.html'
    fields = ['material', 'date', 'quantity']
    success_url = '/sales/'

def dashboard_view(request):
    return render(request, 'forecasting/dashboard.html')

def delete_all_sales(request):
    if request.method == 'POST':
        SalesData.objects.all().delete()
        messages.success(request, 'All sales data has been successfully deleted.')
    return redirect('forecasting:sales_list')

def import_excel(request):
    if request.method == 'POST':
        form = ExcelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                excel_file = request.FILES['excel_file']
                df = pd.read_excel(excel_file)
                
                required_columns = ['material', 'date', 'quantity']
                if not all(col in df.columns for col in required_columns):
                    messages.error(request, 'Excel file must contain columns: material, date, and quantity')
                    return redirect('forecasting:import_excel')
                
                df['date'] = pd.to_datetime(df['date'])
                
                for _, row in df.iterrows():
                    SalesData.objects.create(
                        material=row['material'],
                        date=row['date'].date(),
                        quantity=row['quantity']
                    )
                
                messages.success(request, f'Successfully imported {len(df)} records')
                return redirect('forecasting:sales_list')
            
            except Exception as e:
                messages.error(request, f'Error importing file: {str(e)}')
                return redirect('forecasting:import_excel')
    else:
        form = ExcelUploadForm()
    
    return render(request, 'forecasting/import_excel.html', {'form': form})

def forecast_view(request):
    log_info("Starting forecast view...")
    
    # Get all unique materials
    materials = SalesData.objects.values_list('material', flat=True).distinct().order_by('material')
    selected_material = request.GET.get('material', materials.first())
    
    log_info(f"Selected material: {selected_material}")
    
    # Filter sales data by material
    sales_data = SalesData.objects.filter(material=selected_material).order_by('date')
    if not sales_data:
        log_info("No data available for forecasting")
        return render(request, 'forecasting/forecast.html', {
            'error': 'No data available for forecasting',
            'materials': materials,
            'selected_material': selected_material
        })

    # Get selected models from request
    selected_models = request.GET.getlist('models')
    test_size = float(request.GET.get('test_size', 0.2))
    forecast_days = int(request.GET.get('forecast_days', 30))
    
    log_info(f"Selected models: {selected_models}")
    log_info(f"Test size: {test_size}")
    log_info(f"Forecast days: {forecast_days}")

    # Convert to DataFrame
    df = pd.DataFrame(list(sales_data.values()))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Create days_from_start feature for ML models
    df['days_from_start'] = (df.index - df.index.min()).days

    # Prepare data for modeling
    X = df[['days_from_start']]
    y = df['quantity']

    # Split data for traditional models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Generate future dates for forecasting
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]
    future_days = pd.DataFrame({
        'days_from_start': [(date - df.index.min()).days for date in future_dates]
    })

    # Available models
    all_models = {
        'linear': ('Linear Regression', LinearRegression()),
        'ridge': ('Ridge Regression', Ridge(alpha=1.0)),
        'lasso': ('Lasso Regression', Lasso(alpha=1.0)),
        'rf': ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
        'gbr': ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        'svr': ('Support Vector Regression', SVR(kernel='rbf')),
        'prophet': ('Meta Prophet', None),
        'arima': ('ARIMA', None)
    }

    # If no models selected, use default ones
    if not selected_models:
        selected_models = ['linear', 'rf', 'prophet']

    # Initialize results dictionary and predictions DataFrame
    results = {}
    daily_predictions = pd.DataFrame(index=df.index.union(future_dates))
    daily_predictions['Actual'] = df['quantity']

    # Split point for time series models
    split_idx = int(len(df) * (1 - test_size))
    
    for model_key in selected_models:
        if model_key not in all_models:
            continue
            
        name = all_models[model_key][0]
        
        try:
            if model_key == 'prophet':
                try:
                    log_info("Starting Prophet model predictions...")
                    
                    # Convert data to Prophet format
                    log_info("Converting data to Prophet format...")
                    prophet_data = pd.DataFrame({
                        'ds': df.index,
                        'y': df['quantity']
                    }).reset_index(drop=True)
                    
                    # Train/test split
                    log_info("Performing train/test split...")
                    cutoff_date = df.index[split_idx]
                    train_data = prophet_data.iloc[:split_idx].copy()
                    test_data = prophet_data.iloc[split_idx:].copy()
                    
                    log_info(f"Train data shape: {train_data.shape}")
                    log_info(f"Test data shape: {test_data.shape}")
                    
                    # Initialize and train Prophet model
                    log_info("Initializing Prophet model...")
                    model = Prophet(
                        growth='linear',
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        interval_width=0.95
                    )
                    
                    log_info("Fitting Prophet model...")
                    model.fit(train_data)
                    log_info("Model fitting completed")
                    
                    # Create future dates DataFrame
                    log_info("Creating future dates DataFrame...")
                    future_dates_df = model.make_future_dataframe(
                        periods=len(test_data) + forecast_days,
                        freq='D'
                    )
                    
                    # Make predictions
                    log_info("Making predictions...")
                    forecast = model.predict(future_dates_df)
                    log_info("Predictions completed")
                    
                    # Split predictions
                    test_end_idx = len(test_data)
                    y_pred = forecast['yhat'].values[len(train_data):len(train_data) + test_end_idx]
                    future_pred = forecast['yhat'].values[-forecast_days:]
                    
                    # Ensure non-negative predictions
                    y_pred = np.maximum(y_pred, 0)
                    future_pred = np.maximum(future_pred, 0)
                    
                    # Calculate metrics
                    log_info("Calculating metrics...")
                    test_actual = df['quantity'].iloc[split_idx:]
                    mae = mean_absolute_error(test_actual, y_pred)
                    rmse = np.sqrt(mean_squared_error(test_actual, y_pred))
                    r2 = r2_score(test_actual, y_pred)
                    
                    log_info(f"Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
                    
                    # Store results
                    results[name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2
                    }
                    
                    # Add predictions to DataFrame
                    log_info("Adding predictions to DataFrame...")
                    daily_predictions.loc[df.index[split_idx:], name] = y_pred
                    daily_predictions.loc[future_dates, name] = future_pred
                    
                    log_info("Prophet predictions completed successfully")
                    
                except Exception as e:
                    log_info(f"Prophet error details: {str(e)}")
                    log_info(f"Error occurred at line: {e.__traceback__.tb_lineno}")
                    import traceback
                    traceback.print_exc()
                    messages.warning(request, f"Prophet model error: {str(e)}")
                    continue
            elif model_key == 'arima':
                # Prepare data for ARIMA
                train_data = df['quantity'].iloc[:split_idx]
                test_data = df['quantity'].iloc[split_idx:]
                
                # Fit SARIMA model
                model = SARIMAX(train_data,
                              order=(1, 1, 1),
                              seasonal_order=(1, 1, 1, 7),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                
                model_fit = model.fit(disp=False)
                
                # Make predictions for test period
                y_pred = model_fit.forecast(steps=len(test_data))
                
                # Make future predictions
                future_pred = model_fit.forecast(steps=forecast_days)
                
                # Ensure predictions are non-negative
                y_pred = np.maximum(y_pred, 0)
                future_pred = np.maximum(future_pred, 0)
                
                # Calculate metrics
                mae = mean_absolute_error(test_data, y_pred)
                rmse = np.sqrt(mean_squared_error(test_data, y_pred))
                r2 = r2_score(test_data, y_pred)
                
                # Store results
                results[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
                
                # Add predictions to DataFrame
                daily_predictions.loc[df.index[split_idx:], name] = y_pred
                daily_predictions.loc[future_dates, name] = future_pred
                
            else:
                # Traditional ML models
                model = all_models[model_key][1]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                future_pred = model.predict(future_days)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Store results
                results[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
                
                # Add predictions to DataFrame
                daily_predictions.loc[df.index[-len(y_test):], name] = y_pred
                daily_predictions.loc[future_dates, name] = future_pred
                
        except Exception as e:
            print(f"Error with {name} model: {str(e)}")
            messages.warning(request, f"Error with {name} model: {str(e)}")
            continue

    # Create visualization
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Sales Forecast', 'Model Performance Metrics'),
                       row_heights=[0.7, 0.3])

    # Actual vs Predicted plot
    fig.add_trace(
        go.Scatter(x=df.index, y=df['quantity'], 
                  name='Actual', mode='lines+markers',
                  line=dict(color='blue')),
        row=1, col=1
    )

    # Add predictions for each model
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta']
    for (name, metrics), color in zip(results.items(), colors):
        # Historical predictions
        historical_pred = daily_predictions[name].dropna()
        historical_dates = historical_pred.index[historical_pred.index <= last_date]
        future_dates_pred = historical_pred.index[historical_pred.index > last_date]
        
        if len(historical_dates) > 0:
            fig.add_trace(
                go.Scatter(x=historical_dates, 
                          y=historical_pred[historical_dates],
                          name=f'{name} (Historical)',
                          mode='lines+markers',
                          line=dict(color=color, dash='dot')),
                row=1, col=1
            )
        
        if len(future_dates_pred) > 0:
            fig.add_trace(
                go.Scatter(x=future_dates_pred,
                          y=historical_pred[future_dates_pred],
                          name=f'{name} (Forecast)',
                          mode='lines+markers',
                          line=dict(color=color)),
                row=1, col=1
            )

    # Error metrics plot
    metrics = ['MAE', 'RMSE', 'R²']
    metric_keys = ['mae', 'rmse', 'r2']
    
    for i, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
        fig.add_trace(
            go.Bar(name=metric_name,
                  x=list(results.keys()),
                  y=[result[metric_key] for result in results.values()]),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Sales Forecasting Results",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Quantity", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Error Value", row=2, col=1)

    plot_div = fig.to_html(full_html=False)

    # Format predictions for template
    daily_predictions = daily_predictions.round(2)
    daily_predictions = daily_predictions.reset_index()
    daily_predictions.columns = ['Date'] + list(daily_predictions.columns[1:])
    daily_predictions['Date'] = daily_predictions['Date'].dt.strftime('%Y-%m-%d')
    
    # Convert NaN values to None for proper template handling
    predictions_dict = daily_predictions.where(pd.notnull(daily_predictions), None).to_dict('records')

    context = {
        'plot': plot_div,
        'results': results,
        'all_models': all_models,
        'selected_models': selected_models,
        'test_size': test_size,
        'forecast_days': forecast_days,
        'daily_predictions': predictions_dict,
        'materials': materials,
        'selected_material': selected_material
    }

    return render(request, 'forecasting/forecast.html', context) 