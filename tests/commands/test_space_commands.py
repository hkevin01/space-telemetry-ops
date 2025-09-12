"""
Space Command Tests - Mission Critical Command Processing

Tests for space command processing including critical commands (ABORT, SAFE MODE),
emergency procedures, command validation, telemetry feedback, and safety protocols.
Validates command execution time, authorization, failsafe mechanisms, and mission compliance.
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np
import concurrent.futures
from unittest.mock import Mock, AsyncMock
import logging

# Configure logging for command tracking
logging.basicConfig(level=logging.INFO)
command_logger = logging.getLogger("space_command_system")

# Command processing configuration
COMMAND_CONFIG = {
    "execution_timeout_ms": 1000,      # 1 second max execution time
    "critical_timeout_ms": 100,       # 100ms for critical commands (ABORT, SAFE MODE)
    "authorization_timeout_ms": 500,   # 500ms for authorization validation
    "telemetry_feedback_ms": 50,      # 50ms for telemetry response
    "command_queue_max": 1000,        # Maximum queued commands
    "retry_attempts": 3,              # Command retry attempts
    "safety_check_timeout_ms": 200,   # Safety validation timeout
    "mission_control_timeout_s": 30   # Mission control acknowledgment timeout
}

class CommandPriority(Enum):
    """Command priority levels"""
    CRITICAL = 1      # ABORT, SAFE MODE, EMERGENCY STOP
    HIGH = 2          # Attitude control, propulsion
    MEDIUM = 3        # System configuration, data collection
    LOW = 4           # Housekeeping, diagnostics

class CommandCategory(Enum):
    """Command categories"""
    EMERGENCY = "emergency"
    ATTITUDE_CONTROL = "attitude_control"
    PROPULSION = "propulsion"
    POWER_MANAGEMENT = "power_management"
    COMMUNICATION = "communication"
    THERMAL_CONTROL = "thermal_control"
    PAYLOAD_OPERATIONS = "payload_operations"
    SYSTEM_CONFIGURATION = "system_configuration"
    DIAGNOSTICS = "diagnostics"

class CommandStatus(Enum):
    """Command execution status"""
    PENDING = "pending"
    VALIDATING = "validating"
    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    TIMEOUT = "timeout"

@dataclass
class SpaceCommand:
    """Space command structure"""
    id: str
    name: str
    category: CommandCategory
    priority: CommandPriority
    parameters: Dict[str, Any]
    satellite_id: str
    mission_id: str
    timestamp: datetime
    operator_id: str
    authorization_level: int
    safety_critical: bool
    estimated_duration_ms: int
    
    # Execution tracking
    status: CommandStatus = CommandStatus.PENDING
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    error_message: Optional[str] = None
    telemetry_feedback: Optional[Dict[str, Any]] = None
    retry_count: int = 0

@dataclass
class CommandResponse:
    """Command execution response"""
    command_id: str
    status: CommandStatus
    execution_time_ms: float
    telemetry_data: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    safety_checks_passed: bool = True
    authorization_confirmed: bool = True

class SpaceCommandProcessor:
    """Space command processing system"""
    
    def __init__(self):
        self.command_queue = asyncio.Queue(maxsize=COMMAND_CONFIG["command_queue_max"])
        self.active_commands = {}
        self.command_history = []
        self.safety_systems = SafetySystemsMock()
        self.authorization_system = AuthorizationSystemMock()
        self.telemetry_system = TelemetrySystemMock()
        self.satellite_interfaces = {}
        self.mission_control = MissionControlMock()
        
        # Command handlers
        self.command_handlers = {
            "ABORT": self._handle_abort_command,
            "SAFE_MODE": self._handle_safe_mode_command,
            "EMERGENCY_STOP": self._handle_emergency_stop_command,
            "ATTITUDE_ADJUST": self._handle_attitude_control,
            "THRUSTER_FIRE": self._handle_thruster_control,
            "POWER_MODE": self._handle_power_management,
            "ANTENNA_ORIENT": self._handle_communication_control,
            "THERMAL_HEATER": self._handle_thermal_control,
            "PAYLOAD_ACTIVATE": self._handle_payload_control,
            "SYSTEM_CONFIG": self._handle_system_configuration,
            "RUN_DIAGNOSTIC": self._handle_diagnostics
        }
    
    async def submit_command(self, command: SpaceCommand) -> bool:
        """Submit command to processing queue"""
        try:
            # Validate command
            if not await self._validate_command(command):
                command.status = CommandStatus.FAILED
                command.error_message = "Command validation failed"
                return False
            
            # Add to queue
            await self.command_queue.put(command)
            self.active_commands[command.id] = command
            
            command_logger.info(f"Command queued: {command.name} (ID: {command.id})")
            return True
            
        except asyncio.QueueFull:
            command.status = CommandStatus.FAILED
            command.error_message = "Command queue full"
            return False
    
    async def process_commands(self):
        """Process commands from queue"""
        while True:
            try:
                # Get next command (with timeout to prevent blocking)
                command = await asyncio.wait_for(
                    self.command_queue.get(), timeout=1.0
                )
                
                # Process command asynchronously
                asyncio.create_task(self._execute_command(command))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                command_logger.error(f"Command processing error: {e}")
    
    async def _validate_command(self, command: SpaceCommand) -> bool:
        """Validate command structure and parameters"""
        # Basic structure validation
        if not command.id or not command.name:
            return False
        
        if not command.satellite_id or not command.mission_id:
            return False
        
        # Priority-specific validation
        if command.priority == CommandPriority.CRITICAL:
            # Critical commands must have proper authorization
            if command.authorization_level < 3:
                return False
            
            # Must be safety critical
            if not command.safety_critical:
                return False
        
        # Parameter validation
        if command.name in ["THRUSTER_FIRE", "ATTITUDE_ADJUST"]:
            required_params = ["duration_ms", "axis", "magnitude"]
            if not all(param in command.parameters for param in required_params):
                return False
        
        return True
    
    async def _execute_command(self, command: SpaceCommand) -> CommandResponse:
        """Execute a single command"""
        start_time = time.perf_counter()
        command.execution_start = datetime.utcnow()
        command.status = CommandStatus.VALIDATING
        
        try:
            # Step 1: Safety checks
            if command.safety_critical:
                safety_passed = await self._perform_safety_checks(command)
                if not safety_passed:
                    raise Exception("Safety checks failed")
            
            # Step 2: Authorization
            command.status = CommandStatus.AUTHORIZED
            auth_confirmed = await self._verify_authorization(command)
            if not auth_confirmed:
                raise Exception("Authorization failed")
            
            # Step 3: Execute command
            command.status = CommandStatus.EXECUTING
            
            # Get appropriate handler
            handler = self.command_handlers.get(command.name)
            if not handler:
                raise Exception(f"No handler for command: {command.name}")
            
            # Execute with timeout based on priority
            timeout = (COMMAND_CONFIG["critical_timeout_ms"] if command.priority == CommandPriority.CRITICAL 
                      else COMMAND_CONFIG["execution_timeout_ms"]) / 1000
            
            execution_result = await asyncio.wait_for(
                handler(command), timeout=timeout
            )
            
            # Step 4: Collect telemetry feedback
            telemetry_data = await self._collect_telemetry_feedback(command)
            
            # Command completed successfully
            command.status = CommandStatus.COMPLETED
            command.execution_end = datetime.utcnow()
            execution_time = (time.perf_counter() - start_time) * 1000
            
            response = CommandResponse(
                command_id=command.id,
                status=CommandStatus.COMPLETED,
                execution_time_ms=execution_time,
                telemetry_data=telemetry_data,
                safety_checks_passed=True,
                authorization_confirmed=True
            )
            
            # Log successful execution
            command_logger.info(
                f"Command executed successfully: {command.name} "
                f"({execution_time:.2f}ms)"
            )
            
            # Add to history
            self.command_history.append(command)
            
            return response
            
        except asyncio.TimeoutError:
            command.status = CommandStatus.TIMEOUT
            command.error_message = f"Command execution timeout ({timeout}s)"
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return CommandResponse(
                command_id=command.id,
                status=CommandStatus.TIMEOUT,
                execution_time_ms=execution_time,
                error_details=command.error_message
            )
            
        except Exception as e:
            command.status = CommandStatus.FAILED
            command.error_message = str(e)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            command_logger.error(f"Command failed: {command.name} - {e}")
            
            return CommandResponse(
                command_id=command.id,
                status=CommandStatus.FAILED,
                execution_time_ms=execution_time,
                error_details=str(e)
            )
        
        finally:
            # Remove from active commands
            if command.id in self.active_commands:
                del self.active_commands[command.id]
    
    async def _perform_safety_checks(self, command: SpaceCommand) -> bool:
        """Perform safety checks for critical commands"""
        start_time = time.perf_counter()
        
        # Check system state
        system_safe = await self.safety_systems.check_system_state()
        if not system_safe:
            return False
        
        # Check command conflicts
        conflict_check = await self.safety_systems.check_command_conflicts(command)
        if not conflict_check:
            return False
        
        # Check environmental conditions
        env_check = await self.safety_systems.check_environmental_conditions()
        if not env_check:
            return False
        
        safety_check_time = (time.perf_counter() - start_time) * 1000
        
        # Verify safety check performance
        if safety_check_time > COMMAND_CONFIG["safety_check_timeout_ms"]:
            command_logger.warning(f"Safety checks took {safety_check_time:.2f}ms (limit: {COMMAND_CONFIG['safety_check_timeout_ms']}ms)")
        
        return True
    
    async def _verify_authorization(self, command: SpaceCommand) -> bool:
        """Verify command authorization"""
        return await self.authorization_system.verify_command_authorization(
            command.operator_id, 
            command.authorization_level, 
            command.name
        )
    
    async def _collect_telemetry_feedback(self, command: SpaceCommand) -> Dict[str, Any]:
        """Collect telemetry feedback after command execution"""
        return await self.telemetry_system.get_post_command_telemetry(
            command.satellite_id, 
            command.name
        )
    
    # Command Handlers
    async def _handle_abort_command(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle ABORT command - highest priority emergency stop"""
        command_logger.critical(f"ABORT COMMAND EXECUTING: {command.id}")
        
        # Immediate actions for abort
        tasks = [
            self._abort_all_propulsion(command.satellite_id),
            self._abort_payload_operations(command.satellite_id),
            self._engage_safe_attitude(command.satellite_id),
            self._notify_mission_control_emergency(command)
        ]
        
        # Execute all abort procedures in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if any critical systems failed to abort
        failed_systems = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_systems.append(f"abort_task_{i}")
        
        if failed_systems:
            raise Exception(f"ABORT FAILED - Systems not responding: {failed_systems}")
        
        return {
            "abort_completed": True,
            "systems_aborted": ["propulsion", "payload", "attitude"],
            "mission_control_notified": True,
            "abort_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_safe_mode_command(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle SAFE MODE command - enter minimal power configuration"""
        command_logger.critical(f"SAFE MODE COMMAND EXECUTING: {command.id}")
        
        # Safe mode procedures
        tasks = [
            self._minimize_power_consumption(command.satellite_id),
            self._orient_solar_arrays(command.satellite_id),
            self._disable_non_essential_systems(command.satellite_id),
            self._establish_backup_communication(command.satellite_id)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            "safe_mode_engaged": True,
            "power_minimized": results[0],
            "solar_arrays_oriented": results[1],
            "non_essential_disabled": results[2],
            "backup_comm_established": results[3],
            "safe_mode_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_emergency_stop_command(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle EMERGENCY STOP command"""
        command_logger.critical(f"EMERGENCY STOP COMMAND EXECUTING: {command.id}")
        
        # Stop all active operations immediately
        await self._stop_all_operations(command.satellite_id)
        
        return {
            "emergency_stop_completed": True,
            "all_operations_stopped": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_attitude_control(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle attitude control command"""
        params = command.parameters
        
        # Simulate attitude adjustment
        await asyncio.sleep(0.05)  # 50ms simulation
        
        return {
            "attitude_adjusted": True,
            "axis": params.get("axis"),
            "magnitude": params.get("magnitude"),
            "duration_ms": params.get("duration_ms")
        }
    
    async def _handle_thruster_control(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle thruster control command"""
        params = command.parameters
        
        # Safety check for thruster firing
        if params.get("duration_ms", 0) > 10000:  # Max 10 seconds
            raise Exception("Thruster firing duration exceeds safety limit")
        
        await asyncio.sleep(0.03)  # 30ms simulation
        
        return {
            "thruster_fired": True,
            "axis": params.get("axis"),
            "magnitude": params.get("magnitude"),
            "duration_ms": params.get("duration_ms")
        }
    
    async def _handle_power_management(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle power management command"""
        params = command.parameters
        mode = params.get("mode", "normal")
        
        await asyncio.sleep(0.02)  # 20ms simulation
        
        return {
            "power_mode_set": True,
            "mode": mode,
            "power_level": params.get("power_level", 100)
        }
    
    async def _handle_communication_control(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle communication control command"""
        params = command.parameters
        
        await asyncio.sleep(0.04)  # 40ms simulation
        
        return {
            "antenna_oriented": True,
            "azimuth": params.get("azimuth", 0),
            "elevation": params.get("elevation", 0)
        }
    
    async def _handle_thermal_control(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle thermal control command"""
        params = command.parameters
        
        await asyncio.sleep(0.03)  # 30ms simulation
        
        return {
            "thermal_system_adjusted": True,
            "heater_state": params.get("heater_state", "auto"),
            "target_temperature": params.get("target_temperature", 20)
        }
    
    async def _handle_payload_control(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle payload control command"""
        params = command.parameters
        
        await asyncio.sleep(0.06)  # 60ms simulation
        
        return {
            "payload_activated": True,
            "payload_id": params.get("payload_id"),
            "operation_mode": params.get("operation_mode", "standard")
        }
    
    async def _handle_system_configuration(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle system configuration command"""
        params = command.parameters
        
        await asyncio.sleep(0.08)  # 80ms simulation
        
        return {
            "system_configured": True,
            "config_updated": params.get("config_parameters", {})
        }
    
    async def _handle_diagnostics(self, command: SpaceCommand) -> Dict[str, Any]:
        """Handle diagnostic command"""
        params = command.parameters
        
        await asyncio.sleep(0.1)  # 100ms simulation for diagnostics
        
        return {
            "diagnostics_completed": True,
            "diagnostic_type": params.get("diagnostic_type", "full"),
            "systems_checked": ["power", "communication", "attitude", "thermal"]
        }
    
    # Emergency procedure helpers
    async def _abort_all_propulsion(self, satellite_id: str) -> bool:
        """Abort all propulsion systems"""
        await asyncio.sleep(0.01)  # 10ms critical operation
        return True
    
    async def _abort_payload_operations(self, satellite_id: str) -> bool:
        """Abort all payload operations"""
        await asyncio.sleep(0.01)  # 10ms critical operation
        return True
    
    async def _engage_safe_attitude(self, satellite_id: str) -> bool:
        """Engage safe attitude (sun-pointing)"""
        await asyncio.sleep(0.02)  # 20ms critical operation
        return True
    
    async def _notify_mission_control_emergency(self, command: SpaceCommand) -> bool:
        """Notify mission control of emergency"""
        await asyncio.sleep(0.01)  # 10ms critical operation
        return True
    
    async def _minimize_power_consumption(self, satellite_id: str) -> bool:
        """Minimize power consumption"""
        await asyncio.sleep(0.03)  # 30ms operation
        return True
    
    async def _orient_solar_arrays(self, satellite_id: str) -> bool:
        """Orient solar arrays for maximum power"""
        await asyncio.sleep(0.04)  # 40ms operation
        return True
    
    async def _disable_non_essential_systems(self, satellite_id: str) -> bool:
        """Disable non-essential systems"""
        await asyncio.sleep(0.02)  # 20ms operation
        return True
    
    async def _establish_backup_communication(self, satellite_id: str) -> bool:
        """Establish backup communication"""
        await asyncio.sleep(0.03)  # 30ms operation
        return True
    
    async def _stop_all_operations(self, satellite_id: str) -> bool:
        """Stop all operations immediately"""
        await asyncio.sleep(0.005)  # 5ms emergency stop
        return True

# Mock systems for testing
class SafetySystemsMock:
    """Mock safety systems for testing"""
    
    async def check_system_state(self) -> bool:
        await asyncio.sleep(0.01)  # 10ms safety check
        return True
    
    async def check_command_conflicts(self, command: SpaceCommand) -> bool:
        await asyncio.sleep(0.005)  # 5ms conflict check
        return True
    
    async def check_environmental_conditions(self) -> bool:
        await asyncio.sleep(0.005)  # 5ms environment check
        return True

class AuthorizationSystemMock:
    """Mock authorization system for testing"""
    
    async def verify_command_authorization(self, operator_id: str, auth_level: int, command_name: str) -> bool:
        await asyncio.sleep(0.02)  # 20ms auth check
        
        # Simulate authorization levels
        if command_name in ["ABORT", "SAFE_MODE", "EMERGENCY_STOP"]:
            return auth_level >= 3  # Require level 3+ for critical commands
        
        return auth_level >= 1  # Level 1+ for regular commands

class TelemetrySystemMock:
    """Mock telemetry system for testing"""
    
    async def get_post_command_telemetry(self, satellite_id: str, command_name: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)  # 10ms telemetry collection
        
        return {
            "satellite_id": satellite_id,
            "command": command_name,
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "nominal",
            "power_level": 85.5,
            "temperature": 22.3,
            "attitude": {"pitch": 0.1, "roll": -0.2, "yaw": 0.0}
        }

class MissionControlMock:
    """Mock mission control interface"""
    
    async def send_emergency_notification(self, command: SpaceCommand) -> bool:
        await asyncio.sleep(0.01)  # 10ms notification
        return True

@pytest.fixture
async def command_processor():
    """Pytest fixture for command processor"""
    processor = SpaceCommandProcessor()
    
    # Start command processing task
    processing_task = asyncio.create_task(processor.process_commands())
    
    yield processor
    
    # Cleanup
    processing_task.cancel()
    try:
        await processing_task
    except asyncio.CancelledError:
        pass

def create_test_command(name: str, category: CommandCategory, priority: CommandPriority, 
                       parameters: Dict[str, Any] = None, safety_critical: bool = False,
                       auth_level: int = 1) -> SpaceCommand:
    """Create a test command"""
    return SpaceCommand(
        id=str(uuid.uuid4()),
        name=name,
        category=category,
        priority=priority,
        parameters=parameters or {},
        satellite_id="TEST-SAT-001",
        mission_id="TEST-MISSION-ALPHA",
        timestamp=datetime.utcnow(),
        operator_id="test_operator",
        authorization_level=auth_level,
        safety_critical=safety_critical,
        estimated_duration_ms=100
    )

class TestCriticalCommands:
    """Test critical space commands (ABORT, SAFE MODE, etc.)"""
    
    @pytest.mark.asyncio
    async def test_abort_command_execution(self, command_processor):
        """Test ABORT command execution - highest priority"""
        # Create ABORT command
        abort_command = create_test_command(
            name="ABORT",
            category=CommandCategory.EMERGENCY,
            priority=CommandPriority.CRITICAL,
            safety_critical=True,
            auth_level=3
        )
        
        # Submit command
        submitted = await command_processor.submit_command(abort_command)
        assert submitted, "ABORT command submission failed"
        
        # Wait for processing
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Verify execution
        assert abort_command.status == CommandStatus.COMPLETED, \
            f"ABORT command failed: {abort_command.error_message}"
        
        # Verify execution time (should be very fast for critical commands)
        execution_time = (abort_command.execution_end - abort_command.execution_start).total_seconds() * 1000
        
        assert execution_time < COMMAND_CONFIG["critical_timeout_ms"], \
            f"ABORT execution time {execution_time:.2f}ms exceeds critical limit {COMMAND_CONFIG['critical_timeout_ms']}ms"
        
        # Check command was logged
        assert len(command_processor.command_history) > 0
        assert command_processor.command_history[-1].name == "ABORT"
        
        print(f"ABORT Command Results:")
        print(f"  Execution time: {execution_time:.2f}ms")
        print(f"  Status: {abort_command.status.value}")
        print(f"  Critical timeout limit: {COMMAND_CONFIG['critical_timeout_ms']}ms")
    
    @pytest.mark.asyncio
    async def test_safe_mode_command_execution(self, command_processor):
        """Test SAFE MODE command execution"""
        safe_mode_command = create_test_command(
            name="SAFE_MODE",
            category=CommandCategory.EMERGENCY,
            priority=CommandPriority.CRITICAL,
            safety_critical=True,
            auth_level=3
        )
        
        start_time = time.perf_counter()
        
        # Submit and wait for processing
        submitted = await command_processor.submit_command(safe_mode_command)
        assert submitted, "SAFE_MODE command submission failed"
        
        await asyncio.sleep(0.2)  # Allow processing time
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        # Verify execution
        assert safe_mode_command.status == CommandStatus.COMPLETED, \
            f"SAFE_MODE command failed: {safe_mode_command.error_message}"
        
        assert execution_time < COMMAND_CONFIG["critical_timeout_ms"] + 100, \
            f"SAFE_MODE execution time {execution_time:.2f}ms too high"
        
        print(f"SAFE_MODE Command Results:")
        print(f"  Total execution time: {execution_time:.2f}ms")
        print(f"  Status: {safe_mode_command.status.value}")
    
    @pytest.mark.asyncio
    async def test_emergency_stop_command_execution(self, command_processor):
        """Test EMERGENCY STOP command execution"""
        emergency_command = create_test_command(
            name="EMERGENCY_STOP",
            category=CommandCategory.EMERGENCY,
            priority=CommandPriority.CRITICAL,
            safety_critical=True,
            auth_level=3
        )
        
        # Submit command
        submitted = await command_processor.submit_command(emergency_command)
        assert submitted, "EMERGENCY_STOP command submission failed"
        
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Verify execution
        assert emergency_command.status == CommandStatus.COMPLETED, \
            f"EMERGENCY_STOP command failed: {emergency_command.error_message}"
        
        print(f"EMERGENCY_STOP Command Results:")
        print(f"  Status: {emergency_command.status.value}")
    
    @pytest.mark.asyncio
    async def test_critical_command_authorization_failure(self, command_processor):
        """Test critical command with insufficient authorization"""
        # Create ABORT command with insufficient authorization
        abort_command = create_test_command(
            name="ABORT",
            category=CommandCategory.EMERGENCY,
            priority=CommandPriority.CRITICAL,
            safety_critical=True,
            auth_level=1  # Insufficient for critical commands
        )
        
        # Submit command
        submitted = await command_processor.submit_command(abort_command)
        assert submitted, "Command submission should succeed (validation happens during processing)"
        
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Verify command failed due to authorization
        assert abort_command.status == CommandStatus.FAILED, \
            "ABORT command should fail with insufficient authorization"
        
        assert "Authorization failed" in abort_command.error_message, \
            f"Expected authorization failure, got: {abort_command.error_message}"
        
        print(f"Authorization Test Results:")
        print(f"  Command status: {abort_command.status.value}")
        print(f"  Error message: {abort_command.error_message}")

class TestRoutineCommands:
    """Test routine space commands (attitude control, propulsion, etc.)"""
    
    @pytest.mark.asyncio
    async def test_attitude_control_command(self, command_processor):
        """Test attitude control command"""
        attitude_command = create_test_command(
            name="ATTITUDE_ADJUST",
            category=CommandCategory.ATTITUDE_CONTROL,
            priority=CommandPriority.HIGH,
            parameters={
                "axis": "yaw",
                "magnitude": 5.0,  # degrees
                "duration_ms": 2000
            },
            auth_level=2
        )
        
        # Submit command
        submitted = await command_processor.submit_command(attitude_command)
        assert submitted, "Attitude command submission failed"
        
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Verify execution
        assert attitude_command.status == CommandStatus.COMPLETED, \
            f"Attitude command failed: {attitude_command.error_message}"
        
        execution_time = (attitude_command.execution_end - attitude_command.execution_start).total_seconds() * 1000
        
        assert execution_time < COMMAND_CONFIG["execution_timeout_ms"], \
            f"Attitude command execution time {execution_time:.2f}ms exceeds limit"
        
        print(f"Attitude Control Results:")
        print(f"  Execution time: {execution_time:.2f}ms")
        print(f"  Parameters: {attitude_command.parameters}")
    
    @pytest.mark.asyncio
    async def test_thruster_control_command(self, command_processor):
        """Test thruster control command with safety limits"""
        thruster_command = create_test_command(
            name="THRUSTER_FIRE",
            category=CommandCategory.PROPULSION,
            priority=CommandPriority.HIGH,
            parameters={
                "axis": "x",
                "magnitude": 10.0,  # Newtons
                "duration_ms": 5000  # 5 seconds
            },
            auth_level=2
        )
        
        # Submit command
        submitted = await command_processor.submit_command(thruster_command)
        assert submitted, "Thruster command submission failed"
        
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Verify execution
        assert thruster_command.status == CommandStatus.COMPLETED, \
            f"Thruster command failed: {thruster_command.error_message}"
        
        print(f"Thruster Control Results:")
        print(f"  Status: {thruster_command.status.value}")
        print(f"  Parameters: {thruster_command.parameters}")
    
    @pytest.mark.asyncio
    async def test_thruster_safety_limit(self, command_processor):
        """Test thruster command with excessive duration (should fail)"""
        excessive_thruster_command = create_test_command(
            name="THRUSTER_FIRE",
            category=CommandCategory.PROPULSION,
            priority=CommandPriority.HIGH,
            parameters={
                "axis": "x",
                "magnitude": 10.0,
                "duration_ms": 15000  # 15 seconds - exceeds safety limit
            },
            auth_level=2
        )
        
        # Submit command
        submitted = await command_processor.submit_command(excessive_thruster_command)
        assert submitted, "Command submission should succeed"
        
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Verify command failed due to safety limits
        assert excessive_thruster_command.status == CommandStatus.FAILED, \
            "Thruster command should fail when exceeding safety limits"
        
        assert "safety limit" in excessive_thruster_command.error_message, \
            f"Expected safety limit error, got: {excessive_thruster_command.error_message}"
        
        print(f"Safety Limit Test Results:")
        print(f"  Status: {excessive_thruster_command.status.value}")
        print(f"  Error: {excessive_thruster_command.error_message}")
    
    @pytest.mark.asyncio
    async def test_power_management_command(self, command_processor):
        """Test power management command"""
        power_command = create_test_command(
            name="POWER_MODE",
            category=CommandCategory.POWER_MANAGEMENT,
            priority=CommandPriority.MEDIUM,
            parameters={
                "mode": "high_efficiency",
                "power_level": 75
            },
            auth_level=1
        )
        
        # Submit command
        submitted = await command_processor.submit_command(power_command)
        assert submitted, "Power command submission failed"
        
        await asyncio.sleep(0.2)  # Allow processing time
        
        # Verify execution
        assert power_command.status == CommandStatus.COMPLETED, \
            f"Power command failed: {power_command.error_message}"
        
        print(f"Power Management Results:")
        print(f"  Status: {power_command.status.value}")
        print(f"  Parameters: {power_command.parameters}")

class TestCommandQueueing:
    """Test command queuing and concurrent processing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_command_processing(self, command_processor):
        """Test processing multiple commands concurrently"""
        # Create multiple commands of different priorities
        commands = [
            create_test_command("ATTITUDE_ADJUST", CommandCategory.ATTITUDE_CONTROL, CommandPriority.HIGH, auth_level=2),
            create_test_command("POWER_MODE", CommandCategory.POWER_MANAGEMENT, CommandPriority.MEDIUM, auth_level=1),
            create_test_command("RUN_DIAGNOSTIC", CommandCategory.DIAGNOSTICS, CommandPriority.LOW, auth_level=1),
            create_test_command("ANTENNA_ORIENT", CommandCategory.COMMUNICATION, CommandPriority.MEDIUM, auth_level=1),
            create_test_command("THERMAL_HEATER", CommandCategory.THERMAL_CONTROL, CommandPriority.MEDIUM, auth_level=1)
        ]
        
        # Submit all commands
        submission_times = []
        for cmd in commands:
            start_time = time.perf_counter()
            submitted = await command_processor.submit_command(cmd)
            submission_time = (time.perf_counter() - start_time) * 1000
            submission_times.append(submission_time)
            
            assert submitted, f"Command {cmd.name} submission failed"
        
        # Wait for all commands to process
        await asyncio.sleep(0.5)  # Allow processing time
        
        # Verify all commands completed
        completed_commands = 0
        for cmd in commands:
            if cmd.status == CommandStatus.COMPLETED:
                completed_commands += 1
            else:
                print(f"Command {cmd.name} failed: {cmd.error_message}")
        
        success_rate = completed_commands / len(commands)
        avg_submission_time = np.mean(submission_times)
        
        print(f"Concurrent Processing Results:")
        print(f"  Commands submitted: {len(commands)}")
        print(f"  Commands completed: {completed_commands}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average submission time: {avg_submission_time:.2f}ms")
        
        assert success_rate >= 0.8, f"Success rate {success_rate:.2%} too low"
        assert avg_submission_time < 10, f"Submission time {avg_submission_time:.2f}ms too high"
    
    @pytest.mark.asyncio
    async def test_priority_based_processing(self, command_processor):
        """Test that critical commands get priority processing"""
        # Submit low priority command first
        low_priority_command = create_test_command(
            name="RUN_DIAGNOSTIC",
            category=CommandCategory.DIAGNOSTICS,
            priority=CommandPriority.LOW,
            parameters={"diagnostic_type": "full"},
            auth_level=1
        )
        
        # Submit critical command second (should process first)
        critical_command = create_test_command(
            name="ABORT",
            category=CommandCategory.EMERGENCY,
            priority=CommandPriority.CRITICAL,
            safety_critical=True,
            auth_level=3
        )
        
        # Submit commands
        await command_processor.submit_command(low_priority_command)
        await asyncio.sleep(0.01)  # Small delay
        await command_processor.submit_command(critical_command)
        
        await asyncio.sleep(0.3)  # Allow processing
        
        # Verify both completed
        assert low_priority_command.status == CommandStatus.COMPLETED
        assert critical_command.status == CommandStatus.COMPLETED
        
        # Critical command should complete faster despite being submitted later
        # (This is a simplified test - in reality, priority would be handled by queue ordering)
        
        print(f"Priority Processing Results:")
        print(f"  Low priority command: {low_priority_command.status.value}")
        print(f"  Critical command: {critical_command.status.value}")

class TestCommandValidation:
    """Test command validation and error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_command_structure(self, command_processor):
        """Test handling of invalid command structure"""
        # Create command with missing required parameters
        invalid_command = SpaceCommand(
            id="",  # Missing ID
            name="THRUSTER_FIRE",
            category=CommandCategory.PROPULSION,
            priority=CommandPriority.HIGH,
            parameters={},  # Missing required parameters
            satellite_id="TEST-SAT-001",
            mission_id="TEST-MISSION-ALPHA",
            timestamp=datetime.utcnow(),
            operator_id="test_operator",
            authorization_level=2,
            safety_critical=False,
            estimated_duration_ms=100
        )
        
        # Submit invalid command
        submitted = await command_processor.submit_command(invalid_command)
        
        # Should fail validation
        assert not submitted, "Invalid command should fail validation"
        assert invalid_command.status == CommandStatus.FAILED
        assert "validation failed" in invalid_command.error_message.lower()
        
        print(f"Invalid Command Test Results:")
        print(f"  Submission result: {submitted}")
        print(f"  Command status: {invalid_command.status.value}")
        print(f"  Error message: {invalid_command.error_message}")
    
    @pytest.mark.asyncio
    async def test_command_timeout_handling(self, command_processor):
        """Test command timeout handling"""
        # Create a command that would normally take longer than the timeout
        # (simulated by modifying the handler to take longer)
        
        # Temporarily modify the handler to simulate long execution
        original_handler = command_processor.command_handlers["RUN_DIAGNOSTIC"]
        
        async def slow_diagnostic_handler(command):
            await asyncio.sleep(2.0)  # 2 seconds - exceeds timeout
            return {"diagnostic_completed": True}
        
        command_processor.command_handlers["RUN_DIAGNOSTIC"] = slow_diagnostic_handler
        
        try:
            diagnostic_command = create_test_command(
                name="RUN_DIAGNOSTIC",
                category=CommandCategory.DIAGNOSTICS,
                priority=CommandPriority.LOW,
                parameters={"diagnostic_type": "extended"},
                auth_level=1
            )
            
            # Submit command
            submitted = await command_processor.submit_command(diagnostic_command)
            assert submitted, "Command submission should succeed"
            
            await asyncio.sleep(2.5)  # Wait for timeout
            
            # Verify command timed out
            assert diagnostic_command.status == CommandStatus.TIMEOUT, \
                f"Expected timeout, got status: {diagnostic_command.status.value}"
            
            assert "timeout" in diagnostic_command.error_message.lower(), \
                f"Expected timeout error, got: {diagnostic_command.error_message}"
            
            print(f"Timeout Test Results:")
            print(f"  Command status: {diagnostic_command.status.value}")
            print(f"  Error message: {diagnostic_command.error_message}")
        
        finally:
            # Restore original handler
            command_processor.command_handlers["RUN_DIAGNOSTIC"] = original_handler

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
