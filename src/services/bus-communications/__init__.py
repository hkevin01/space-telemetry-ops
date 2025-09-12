"""
Bus Communications Service

This module provides comprehensive spacecraft bus communication protocols
including MIL-STD-1553, CAN, SpaceWire, UART, and other common hardware/software buses
used in aerospace applications.
"""

from .mil_std_1553 import MilStd1553Bus, MilStd1553Controller
from .can_bus import CanBusController, CanMessage
from .spacewire import SpaceWireController, SpaceWirePacket
from .uart_bus import UartController, UartMessage
from .arinc_429 import Arinc429Controller, Arinc429Word
from .i2c_bus import I2cController, I2cTransaction
from .spi_bus import SpiController, SpiTransaction
from .ethernet import EthernetController, EthernetPacket
from .bus_manager import BusManager, BusType

__all__ = [
    # MIL-STD-1553
    'MilStd1553Bus',
    'MilStd1553Controller',

    # CAN Bus
    'CanBusController',
    'CanMessage',

    # SpaceWire
    'SpaceWireController',
    'SpaceWirePacket',

    # UART
    'UartController',
    'UartMessage',

    # ARINC-429
    'Arinc429Controller',
    'Arinc429Word',

    # I2C
    'I2cController',
    'I2cTransaction',

    # SPI
    'SpiController',
    'SpiTransaction',

    # Ethernet
    'EthernetController',
    'EthernetPacket',

    # Manager
    'BusManager',
    'BusType',
]

__version__ = "1.0.0"
__author__ = "Space Telemetry Operations Team"
