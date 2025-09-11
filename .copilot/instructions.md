# Space Telemetry Operations - Copilot Instructions

## Project Context
This is a mission-critical space telemetry operations system that handles real-time spacecraft data processing, analysis, and monitoring. The system must maintain high availability, security, and reliability standards.

## Code Standards
- **Python**: Use type hints, docstrings, and follow PEP 8
- **TypeScript/JavaScript**: Use strict mode, proper error handling
- **Security**: Always validate inputs, use secure coding practices
- **Performance**: Consider memory management and boundary conditions
- **Testing**: Write comprehensive tests for all critical functions

## Domain-Specific Considerations
- Time measurements should use appropriate units (seconds, milliseconds, nanoseconds)
- Handle spacecraft coordinate systems and reference frames properly
- Implement graceful degradation for mission-critical operations
- Ensure data persistence and recovery mechanisms
- Use proper telemetry packet structures and protocols

## Security Requirements
- Follow NIST SP 800-53 guidelines
- Implement proper authentication and authorization
- Secure all API endpoints
- Handle sensitive telemetry data appropriately
- Maintain audit trails for all operations

## Error Handling
- Always provide meaningful error messages
- Implement proper logging for debugging
- Handle edge cases and boundary conditions
- Ensure graceful failure modes
- Include retry mechanisms for transient failures

## Performance Guidelines
- Optimize for real-time telemetry processing
- Consider memory constraints in space systems
- Implement proper caching strategies
- Use appropriate data structures for time-series data
- Monitor resource usage and implement limits
