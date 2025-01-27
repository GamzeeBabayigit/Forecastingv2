from django.urls import path
from .views import SalesListView, SalesCreateView, forecast_view, dashboard_view, import_excel, delete_all_sales

app_name = 'forecasting'

urlpatterns = [
    path('', dashboard_view, name='dashboard'),
    path('sales/', SalesListView.as_view(), name='sales_list'),
    path('sales/add/', SalesCreateView.as_view(), name='sales_add'),
    path('sales/import/', import_excel, name='import_excel'),
    path('sales/delete-all/', delete_all_sales, name='delete_all_sales'),
    path('forecast/', forecast_view, name='forecast'),
] 