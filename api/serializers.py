"""
Serializers for API endpoints.
"""
from rest_framework import serializers


class DatasetInfoSerializer(serializers.Serializer):
    """Serializer for dataset information."""
    shape = serializers.ListField(child=serializers.IntegerField())
    columns = serializers.ListField(child=serializers.CharField())
    dtypes = serializers.DictField()
    missing_values = serializers.DictField()
    memory_usage = serializers.IntegerField()


class DataSplitSerializer(serializers.Serializer):
    """Serializer for data split parameters."""
    random_state = serializers.IntegerField(default=42)
    shuffle = serializers.BooleanField(default=True)
    stratify_column = serializers.CharField(required=False, allow_null=True)


class TransformDataSerializer(serializers.Serializer):
    """Serializer for data transformation parameters."""
    remove_nan = serializers.BooleanField(default=False)
    scale_columns = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True
    )
    one_hot_encode = serializers.BooleanField(default=False)
