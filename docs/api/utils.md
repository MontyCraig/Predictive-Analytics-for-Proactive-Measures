# Utilities Modules

## Overview

The Utilities module provides essential infrastructure components for the Predictive Analytics framework. It includes robust configuration management with type validation through Pydantic, and a comprehensive logging system. These components ensure the reliability, maintainability, and observability of the application.

## Architecture

The utilities are organized into focused submodules:

1. **Configuration Management**: Pydantic models for type-safe configuration
2. **Logging System**: Structured logging with multiple output options
3. **Path Management**: Utilities for handling file paths consistently

## API Reference

### Configuration Module (`utils.config`)

#### Classes

##### `APIConfig` (Pydantic BaseModel)

API Configuration for data sources.

**Fields:**
- `alpha_vantage_api_key` (SecretStr): Alpha Vantage API key for market data
- `alpha_vantage_base_url` (str): Base URL for Alpha Vantage API (default: "https://www.alphavantage.co/query")

##### `ModelConfig` (Pydantic BaseModel)

Configuration for predictive models.

**Fields:**
- `model_type` (str): Model type (default: "sarima")
- `train_size` (float): Training data proportion (default: 0.8)
- `test_size` (float): Test data proportion (default: 0.2)
- `random_state` (int): Random seed for reproducibility (default: 42)

**Methods:**
- `validate_test_size(v, values)`: Validates that train_size + test_size = 1

##### `SARIMAConfig` (Pydantic BaseModel)

Configuration for SARIMA models.

**Fields:**
- `order` (tuple): SARIMA model order (p, d, q) (default: (1, 1, 1))
- `seasonal_order` (tuple): SARIMA seasonal order (P, D, Q, s) (default: (1, 1, 1, 12))
- `enforce_stationarity` (bool): Whether to enforce stationarity (default: True)
- `enforce_invertibility` (bool): Whether to enforce invertibility (default: True)

##### `Config` (Pydantic BaseModel)

Main configuration for the application.

**Fields:**
- `api` (APIConfig): API configuration
- `model` (ModelConfig): Model configuration
- `sarima` (Optional[SARIMAConfig]): SARIMA-specific configuration
- `data_file` (Optional[str]): Path to data file if using local data
- `output_dir` (str): Directory to store outputs
- `log_level` (str): Logging level (default: "INFO")

**Methods:**
- `validate_sarima_config(v, values)`: Ensures SARIMA config is present when model_type is 'sarima'

#### Functions

##### `get_config(env_file: Optional[str] = None) -> Config`

Load configuration from environment variables or .env file.

**Parameters:**
- `env_file` (str, optional): Optional path to .env file

**Returns:**
- `Config`: Config object with application settings

### Logging Module (`utils.logging_config`)

#### Functions

##### `setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger`

Set up logging configuration.

**Parameters:**
- `level` (int, optional): The logging level (default: logging.INFO)
- `log_file` (str, optional): Optional file path to write logs to

**Returns:**
- `logging.Logger`: A configured logger instance

#### Global Variables

##### `logger`

Default logger instance created when the module is imported.

## Usage Examples

### Configuration Management

#### Basic Configuration

```python
from predictive_analytics.config.settings import get_config

# Load configuration from environment variables or .env file
config = get_config()

# Access configuration values
api_key = config.api.alpha_vantage_api_key.get_secret_value()
model_type = config.model.model_type
output_dir = config.output_dir
```

#### Custom Configuration File

```python
from predictive_analytics.config.settings import get_config

# Load configuration from a specific .env file
config = get_config(env_file="custom_config.env")

# Access model-specific configuration
if config.model.model_type == "sarima":
    order = config.sarima.order
    seasonal_order = config.sarima.seasonal_order
```

### Logging System

#### Basic Logging

```python
from predictive_analytics.utils.logging_config import logger

# Log messages at different levels
logger.debug("Detailed debugging information")
logger.info("General information about program execution")
logger.warning("Warning about potential issues")
logger.error("Error that occurred during execution")
logger.critical("Critical error that requires immediate attention")
```

#### Custom Logger Setup

```python
import logging
from predictive_analytics.utils.logging_config import setup_logging

# Create a custom logger with specific settings
custom_logger = setup_logging(
    level=logging.DEBUG,
    log_file="output/logs/application.log"
)

# Use the custom logger
custom_logger.debug("Detailed debugging information")
custom_logger.info("Processing data for analysis")
```

## Implementation Details

### Configuration Management

The configuration system uses Pydantic models to ensure type safety and validation. It follows these principles:

1. **Type Safety**: All configuration values have explicit types
2. **Validation**: Values are validated during loading, with custom validators for complex relationships
3. **Encapsulation**: Related configuration is grouped into nested models
4. **Security**: Sensitive values like API keys are handled using `SecretStr` to prevent accidental exposure
5. **Flexibility**: Configuration can be loaded from environment variables or `.env` files

### Logging System

The logging system provides structured, configurable logging with these features:

1. **Formatted Output**: Consistent log message format with timestamps and levels
2. **Multiple Destinations**: Logs can be directed to console and/or files
3. **Level Control**: Granular control over logging verbosity
4. **Context Preservation**: Loggers are configured to maintain context information
5. **Reusability**: Common setup logic is encapsulated in reusable functions

## Integration Points

The utilities modules integrate with:
- Environment variables and `.env` files for configuration
- File system for log files and determining paths
- Python's standard logging infrastructure
- External libraries via their configuration settings

## Areas for Enhancement

1. **Remote Configuration**: Add support for loading configuration from remote sources
2. **Configuration Validation**: Implement more sophisticated validation rules
3. **Structured Logging**: Enhance logging with structured JSON output for better analysis
4. **Observability Integration**: Add integration with observability platforms
5. **Dynamic Configuration**: Support for runtime configuration changes
6. **Configuration Migration**: Add tools for handling configuration changes between versions 