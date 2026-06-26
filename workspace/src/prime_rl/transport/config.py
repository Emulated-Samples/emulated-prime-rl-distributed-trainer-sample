from typing import Literal, TypeAlias

from pydantic import BaseModel


class BaseTransportConfig(BaseModel):
    """Base configuration for trainer/orchestrator transport."""

    pass


class FileSystemTransportConfig(BaseTransportConfig):
    """Configures filesystem-based transport for training examples."""

    type: Literal["filesystem"] = "filesystem"


class ZMQTransportConfig(BaseTransportConfig):
    """Configures ZMQ-based transport for training examples."""

    type: Literal["zmq"] = "zmq"
    # Public connection fields expected by the transport factory:
    # host, port, hwm.


TransportConfigType: TypeAlias = FileSystemTransportConfig | ZMQTransportConfig
