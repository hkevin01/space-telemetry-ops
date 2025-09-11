# Data Directory

This directory contains data files and datasets used by the Space Telemetry Operations System.

## Structure

```
data/
├── samples/          # Sample telemetry data for testing
├── schemas/          # Data schemas and validation files
├── fixtures/         # Test fixtures and mock data
├── exports/          # Generated reports and exports
├── backups/          # Database backups (gitignored)
├── cache/            # Temporary cache files (gitignored)
└── ml/               # Machine learning datasets and models
    ├── training/     # Training datasets
    ├── models/       # Trained ML models
    └── validation/   # Validation datasets
```

## Usage Guidelines

### Sample Data
- Use sample telemetry data for development and testing
- Ensure no real mission data is stored in this repository
- All sample data should be synthetic or publicly available

### Schemas
- JSON Schema files for telemetry packet validation
- Database schema definitions and migration scripts
- API schema definitions and OpenAPI specifications

### Security Notes
- Never commit sensitive or classified telemetry data
- Use data anonymization for any real-world examples
- Follow data governance policies for your organization

## File Naming Conventions

- Use descriptive names with ISO date format: `telemetry_sample_2024-01-15.json`
- Include mission or spacecraft identifier when applicable: `sat001_orbit_data.csv`
- Use standardized file extensions: `.json`, `.csv`, `.parquet`, `.avro`
