from typing import Dict, List, Optional

from aws_architect_agent.models.base import Component, ComponentType


class ETLComponentFactory:
    """Factory for creating ETL-specific components."""

    @staticmethod
    def create_data_source(
        name: str,
        source_type: str,
        configuration: Optional[Dict] = None,
    ) -> Component:
        """Create a data source component.

        Args:
            name: Name of the data source
            source_type: Type of data source (e.g., 's3', 'rds', 'dynamodb')
            configuration: Additional configuration for the data source

        Returns:
            Configured data source component
        """
        config = configuration or {}
        config.update({"source_type": source_type})

        return Component(
            id=f"source_{name.lower()}",
            type=ComponentType.DATA_SOURCE,
            name=name,
            description=f"{source_type.upper()} data source",
            configuration=config,
        )

    @staticmethod
    def create_processing(
        name: str,
        processing_type: str,
        dependencies: List[str],
        configuration: Optional[Dict] = None,
    ) -> Component:
        """Create a processing component.

        Args:
            name: Name of the processing component
            processing_type: Type of processing (e.g., 'glue', 'lambda', 'emr')
            dependencies: List of component IDs this component depends on
            configuration: Additional configuration for the processing

        Returns:
            Configured processing component
        """
        config = configuration or {}
        config.update({"processing_type": processing_type})

        return Component(
            id=f"process_{name.lower()}",
            type=ComponentType.PROCESSING,
            name=name,
            description=f"{processing_type.upper()} processing component",
            configuration=config,
            dependencies=dependencies,
        )

    @staticmethod
    def create_storage(
        name: str,
        storage_type: str,
        dependencies: List[str],
        configuration: Optional[Dict] = None,
    ) -> Component:
        """Create a storage component.

        Args:
            name: Name of the storage component
            storage_type: Type of storage (e.g., 's3', 'redshift', 'dynamodb')
            dependencies: List of component IDs this component depends on
            configuration: Additional configuration for the storage

        Returns:
            Configured storage component
        """
        config = configuration or {}
        config.update({"storage_type": storage_type})

        return Component(
            id=f"storage_{name.lower()}",
            type=ComponentType.STORAGE,
            name=name,
            description=f"{storage_type.upper()} storage component",
            configuration=config,
            dependencies=dependencies,
        )

    @staticmethod
    def create_monitoring(
        name: str,
        monitoring_type: str,
        dependencies: List[str],
        configuration: Optional[Dict] = None,
    ) -> Component:
        """Create a monitoring component.

        Args:
            name: Name of the monitoring component
            monitoring_type: Type of monitoring (e.g., 'cloudwatch', 'xray')
            dependencies: List of component IDs this component depends on
            configuration: Additional configuration for the monitoring

        Returns:
            Configured monitoring component
        """
        config = configuration or {}
        config.update({"monitoring_type": monitoring_type})

        return Component(
            id=f"monitor_{name.lower()}",
            type=ComponentType.MONITORING,
            name=name,
            description=f"{monitoring_type.upper()} monitoring component",
            configuration=config,
            dependencies=dependencies,
        ) 