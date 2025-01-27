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
import traceback
warnings.filterwarnings('ignore')

class SalesListView(ListView):
    model = SalesData
    template_name = 'forecasting/sales_list.html'
    context_object_name = 'sales'

class SalesCreateView(CreateView):
    model = SalesData
    template_name = 'forecasting/sales_form.html'
    fields = ['material', 'date', 'quantity']
    success_url = '/sales/'

def forecast_view(request):
    # Get all unique materials
    materials = SalesData.objects.values_list('material', flat=True).distinct().order_by('material')
    
    # Get selected material from request, default to first material
    selected_material = request.GET.get('material', materials.first())
    
    # Filter sales data by material
    sales_data = SalesData.objects.filter(material=selected_material).order_by('date')
    if not sales_data:
        return render(request, 'forecasting/forecast.html', {
            'error': 'No data available for forecasting',
            'materials': materials,
            'selected_material': selected_material
        })

    # Get selected models from request
    selected_models = request.GET.getlist('models')
    test_size = float(request.GET.get('test_size', 0.2))
    forecast_days = int(request.GET.get('forecast_days', 30))

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
    predictions_df = pd.DataFrame(index=pd.date_range(start=df.index.min(), end=future_dates[-1], freq='D'))
    predictions_df['Actual'] = df['quantity']

    # Split point for time series models
    split_idx = int(len(df) * (1 - test_size))
    
    for model_key in selected_models:
        if model_key not in all_models:
            continue
            
        name = all_models[model_key][0]
        
        try:
            if model_key == 'prophet':
                # Debug için DataFrame yapısını kontrol edelim
                print("Original DataFrame columns:", df.columns)
                
                # Prepare data for Prophet
                prophet_df = df.reset_index()
                print("After reset_index columns:", prophet_df.columns)
                
                # Sadece gerekli sütunları seçelim ve yeniden adlandıralım
                prophet_df = prophet_df[['date', 'quantity']].copy()
                prophet_df.columns = ['ds', 'y']
                
                # Train Prophet model
                model = Prophet(yearly_seasonality=True, 
                              weekly_seasonality=True, 
                              daily_seasonality=False,
                              interval_width=0.95)
                
                # Model eğitimi
                model.fit(prophet_df.iloc[:split_idx])
                
                # Test dönemi tahminleri
                future_dates_test = prophet_df.iloc[split_idx:]['ds']
                forecast_test = model.predict(pd.DataFrame({'ds': future_dates_test}))
                y_pred = forecast_test['yhat'].values
                
                # Gelecek tahminleri
                future_dates_df = pd.DataFrame({'ds': future_dates})
                future_forecast = model.predict(future_dates_df)
                future_pred = future_forecast['yhat'].values
                
                # Negatif değerleri sıfırla
                y_pred = np.maximum(y_pred, 0)
                future_pred = np.maximum(future_pred, 0)
                
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
                
            else:
                # Traditional ML models
                model = all_models[model_key][1]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                future_pred = model.predict(future_days)
            
            # Calculate metrics
            if model_key in ['prophet', 'arima']:
                test_actual = df['quantity'].iloc[split_idx:]
                mae = mean_absolute_error(test_actual, y_pred)
                rmse = np.sqrt(mean_squared_error(test_actual, y_pred))
                r2 = r2_score(test_actual, y_pred)
            else:
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'future_predictions': future_pred
            }

            # Add predictions to DataFrame
            if model_key in ['prophet', 'arima']:
                predictions_df.loc[df.index[split_idx:], name] = y_pred
            else:
                predictions_df.loc[df.index[-len(y_test):], name] = y_pred
            
            # Add future predictions
            predictions_df.loc[future_dates, name] = future_pred
            
        except Exception as e:
            error_msg = f"Error with {name}: {str(e)}\n{traceback.format_exc()}"
            messages.warning(request, error_msg)
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
    for (name, result), color in zip(results.items(), colors):
        # Historical predictions
        if name in ['Meta Prophet', 'ARIMA']:
            pred_dates = df.index[split_idx:]
        else:
            pred_dates = df.index[-len(y_test):]
            
        fig.add_trace(
            go.Scatter(x=pred_dates, 
                      y=result['predictions'],
                      name=f'{name} (Historical)',
                      mode='lines+markers',
                      line=dict(color=color, dash='dot')),
            row=1, col=1
        )
        
        # Future predictions
        fig.add_trace(
            go.Scatter(x=future_dates,
                      y=result['future_predictions'],
                      name=f'{name} (Forecast)',
                      mode='lines+markers',
                      line=dict(color=color)),
            row=1, col=1
        )

    # Error metrics plot
    metrics = ['MAE', 'RMSE', 'R²']
    for i, metric in enumerate(['mae', 'rmse', 'r2']):
        fig.add_trace(
            go.Bar(name=metrics[i],
                  x=list(results.keys()),
                  y=[result[metric] for result in results.values()]),
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
    predictions_df = predictions_df.round(2)
    predictions_df = predictions_df.reset_index()
    predictions_df.columns = ['Date'] + list(predictions_df.columns[1:])
    predictions_df['Date'] = predictions_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Convert NaN values to None for proper template handling
    predictions_dict = predictions_df.where(pd.notnull(predictions_df), None).to_dict('records')

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

def dashboard_view(request):
    return render(request, 'forecasting/dashboard.html')

def delete_all_sales(request):
    if request.method == 'POST':
        # Delete all sales data
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
                
                # Validate required columns
                required_columns = ['material', 'date', 'quantity']
                if not all(col in df.columns for col in required_columns):
                    messages.error(request, 'Excel file must contain columns: material, date, and quantity')
                    return redirect('forecasting:import_excel')
                
                # Convert date column to datetime if it's not already
                df['date'] = pd.to_datetime(df['date'])
                
                # Create SalesData objects
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
