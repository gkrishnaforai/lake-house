import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from etl_architect_agent_v2.backend.config import get_settings

settings = get_settings()

def setup_logging(
    service_name: str = "aws-architect-agent",
    log_level: Optional[str] = None,
) -> None:
    """Set up logging and tracing for the application.

    Args:
        service_name: Name of the service for tracing
        log_level: Logging level to use
    """
    # Configure logging
    level = log_level or settings.LOG_LEVEL
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configure tracing
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Add OTLP exporter if configured
    if settings.DEBUG:
        # Use console exporter in debug mode
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        span_processor = BatchSpanProcessor(ConsoleSpanExporter())
    else:
        # Use OTLP exporter in production
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="localhost:4317",
                insecure=True,
            )
        )

    tracer_provider.add_span_processor(span_processor)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance with the given name.

    Args:
        name: Name of the tracer

    Returns:
        Configured tracer instance
    """
    return trace.get_tracer(name) 