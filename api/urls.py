"""
URL configuration for API endpoints.
"""
from django.urls import path
from .views import (
    HealthCheckView,
    DatasetInfoView,
    DataSplitView,
    TransformDataView
)

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('dataset/info/', DatasetInfoView.as_view(), name='dataset-info'),
    path('dataset/split/', DataSplitView.as_view(), name='dataset-split'),
    path('dataset/transform/', TransformDataView.as_view(), name='dataset-transform'),
]
