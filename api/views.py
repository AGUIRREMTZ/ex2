"""
API views for machine learning data processing.
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.pipeline import Pipeline
import pandas as pd
import json

from .serializers import (
    DatasetInfoSerializer,
    DataSplitSerializer,
    TransformDataSerializer
)
from .utils import train_val_test_split, get_dataset_info
from .transformers import DeleteNanRows, CustomScaler, CustomOneHotEncoding


class HealthCheckView(APIView):
    """Health check endpoint."""
    
    def get(self, request):
        return Response({
            'status': 'healthy',
            'message': 'ML API is running'
        })


class DatasetInfoView(APIView):
    """Get information about uploaded dataset."""
    
    def post(self, request):
        try:
            # Expect JSON data with records
            data = request.data.get('data', [])
            if not data:
                return Response(
                    {'error': 'No data provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            df = pd.DataFrame(data)
            info = get_dataset_info(df)
            
            serializer = DatasetInfoSerializer(info)
            return Response(serializer.data)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DataSplitView(APIView):
    """Split dataset into train, validation, and test sets."""
    
    def post(self, request):
        try:
            data = request.data.get('data', [])
            if not data:
                return Response(
                    {'error': 'No data provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            serializer = DataSplitSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            df = pd.DataFrame(data)
            train_set, val_set, test_set = train_val_test_split(
                df,
                rstate=serializer.validated_data['random_state'],
                shuffle=serializer.validated_data['shuffle'],
                stratify=serializer.validated_data.get('stratify_column')
            )
            
            return Response({
                'train_size': len(train_set),
                'val_size': len(val_set),
                'test_size': len(test_set),
                'train_data': train_set.to_dict('records')[:100],  # Limit response size
                'val_data': val_set.to_dict('records')[:100],
                'test_data': test_set.to_dict('records')[:100],
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TransformDataView(APIView):
    """Apply transformations to dataset."""
    
    def post(self, request):
        try:
            data = request.data.get('data', [])
            if not data:
                return Response(
                    {'error': 'No data provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            serializer = TransformDataSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            df = pd.DataFrame(data)
            
            # Build pipeline based on requested transformations
            transformers = []
            
            if serializer.validated_data.get('remove_nan'):
                transformers.append(('delete_nan', DeleteNanRows()))
            
            if serializer.validated_data.get('scale_columns'):
                scale_cols = serializer.validated_data['scale_columns']
                transformers.append(('scaler', CustomScaler(scale_cols)))
            
            if serializer.validated_data.get('one_hot_encode'):
                transformers.append(('one_hot', CustomOneHotEncoding()))
            
            if transformers:
                pipeline = Pipeline(transformers)
                df_transformed = pipeline.fit_transform(df)
            else:
                df_transformed = df
            
            return Response({
                'original_shape': df.shape,
                'transformed_shape': df_transformed.shape,
                'transformed_data': df_transformed.to_dict('records')[:100],
                'columns': list(df_transformed.columns)
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
