# Python CLI Application Standards and Best Practices

A comprehensive guide for building robust, user-friendly command-line interface (CLI) applications in Python.

## Table of Contents

1. **Project Structure**
    - Application Layout
    - Command Organization
    - Configuration Management
    - Plugin Architecture
    - Package Distribution

2. **CLI Framework**
    - Click vs Typer
    - Command Groups
    - Arguments & Options
    - Input Validation
    - Help Documentation

3. **User Experience**
    - Progress Feedback
    - Error Handling
    - Color & Formatting
    - Interactive Features
    - Shell Completion

4. **Advanced Features**
    - Configuration Files
    - Environment Variables
    - Logging
    - Plugin System
    - Shell Integration

5. **Testing & Quality**
    - Unit Testing
    - Integration Testing
    - Documentation
    - Distribution
    - Maintenance

---

## 1. Project Structure

### Basic Application Layout
```
cli_project/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── cli.py
│       ├── commands/
│       │   ├─��� __init__.py
│       │   ├── base.py
│       │   └── specific_commands.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_commands.py
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

### Command Organization
```python
# src/myapp/cli.py
import typer
from typing import Optional
from pathlib import Path
from .commands import users, config, utils

app = typer.Typer(
    name="myapp",
    help="CLI application description",
    add_completion=True
)

# Add command groups
app.add_typer(users.app, name="users")
app.add_typer(config.app, name="config")
app.add_typer(utils.app, name="utils")

@app.callback()
def callback():
    """
    MyApp CLI - A tool for managing something awesome.
    """
    pass

def main():
    app()

if __name__ == "__main__":
    main()
```

---

## 2. CLI Framework

### Command Implementation
```python
# src/myapp/commands/users.py
import typer
from typing import Optional, List
from pathlib import Path
from ..utils.console import console

app = typer.Typer(help="User management commands")

@app.command()
def create(
    username: str = typer.Argument(..., help="Username for the new user"),
    email: str = typer.Option(..., "--email", "-e", help="User's email"),
    admin: bool = typer.Option(False, "--admin", help="Grant admin privileges"),
    groups: Optional[List[str]] = typer.Option(
        None,
        "--group", "-g",
        help="Groups to add user to"
    )
):
    """
    Create a new user with the specified details.
    """
    try:
        # Implementation
        console.print(f"Creating user: {username}")
        # ... user creation logic ...
        console.print(f"✓ User {username} created successfully", style="green")
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(1)

@app.command()
def list(
    active: bool = typer.Option(True, help="Show only active users"),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format (table, json, csv)"
    )
):
    """
    List all users in the system.
    """
    try:
        # Implementation
        users = [{"username": "user1", "email": "user1@example.com"}]
        if format == "table":
            # Use rich tables for display
            table = Table(show_header=True)
            table.add_column("Username")
            table.add_column("Email")
            for user in users:
                table.add_row(user["username"], user["email"])
            console.print(table)
        elif format == "json":
            console.print_json(data=users)
        elif format == "csv":
            # Output CSV format
            import csv
            import sys
            writer = csv.DictWriter(sys.stdout, fieldnames=["username", "email"])
            writer.writeheader()
            writer.writerows(users)
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        raise typer.Exit(1)
```

### Input Validation
```python
# src/myapp/utils/validators.py
from typing import Any, Optional
import re
import typer

def validate_email(ctx: typer.Context, param: typer.Parameter, value: str) -> str:
    """Validate email format."""
    if not re.match(r"[^@]+@[^@]+\.[^@]+", value):
        raise typer.BadParameter("Invalid email format")
    return value

def validate_path(
    ctx: typer.Context,
    param: typer.Parameter,
    value: Optional[Path]
) -> Optional[Path]:
    """Validate path exists if provided."""
    if value and not value.exists():
        raise typer.BadParameter(f"Path does not exist: {value}")
    return value

# Usage in commands
@app.command()
def create(
    email: str = typer.Option(
        ...,
        "--email", "-e",
        callback=validate_email,
        help="User's email"
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        callback=validate_path,
        help="Path to config file"
    )
):
    pass
```

---

## 3. User Experience

### Progress Feedback
```python
# src/myapp/utils/console.py
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from functools import wraps

console = Console()

def with_spinner(message: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with console.status(message, spinner="dots") as status:
                try:
                    result = func(*args, **kwargs)
                    status.stop()
                    return result
                except Exception as e:
                    status.stop()
                    console.print(f"Error: {str(e)}", style="red")
                    raise
        return wrapper
    return decorator

# Usage
@app.command()
@with_spinner("Processing data...")
def process_data(file: Path):
    # Long-running operation
    process_file(file)
    console.print("✓ Data processed successfully", style="green")
```

### Interactive Features
```python
# src/myapp/utils/interactive.py
from rich.prompt import Prompt, Confirm
from rich.console import Console

console = Console()

def interactive_config():
    """Interactive configuration setup."""
    console.print("=== Configuration Setup ===", style="bold blue")
    
    # Get user input
    host = Prompt.ask(
        "Enter database host",
        default="localhost"
    )
    port = int(Prompt.ask(
        "Enter database port",
        default="5432"
    ))
    
    # Confirm dangerous operations
    if Confirm.ask("Do you want to reset the database?"):
        console.print("Resetting database...", style="yellow")
        # ... implementation ...
    
    return {
        "host": host,
        "port": port
    }

# Usage in commands
@app.command()
def configure():
    """
    Interactive configuration setup.
    """
    config = interactive_config()
    save_config(config)
    console.print("✓ Configuration saved", style="green")
```

---

## 4. Advanced Features

### Configuration Management
```python
# src/myapp/config/settings.py
from pathlib import Path
from typing import Optional, Dict, Any
import tomli
import tomli_w
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    app_name: str = "myapp"
    debug: bool = False
    api_key: Optional[str] = None
    config_path: Path = Field(
        default_factory=lambda: Path.home() / ".myapp" / "config.toml"
    )
    
    class Config:
        env_prefix = "MYAPP_"
        env_file = ".env"

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    if config_path is None:
        config_path = Settings().config_path
    
    if not config_path.exists():
        return {}
    
    with config_path.open("rb") as f:
        return tomli.load(f)

def save_config(config: Dict[str, Any], config_path: Optional[Path] = None):
    """Save configuration to TOML file."""
    if config_path is None:
        config_path = Settings().config_path
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("wb") as f:
        tomli_w.dump(config, f)
```

### Plugin System
```python
# src/myapp/plugins/base.py
from typing import Protocol, Dict, Any
from pathlib import Path
import importlib.util
import pkg_resources

class PluginProtocol(Protocol):
    """Protocol defining the plugin interface."""
    
    @property
    def name(self) -> str:
        """Plugin name."""
        ...
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        ...
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality."""
        ...

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, PluginProtocol] = {}
    
    def load_plugins(self):
        """Load all installed plugins."""
        for entry_point in pkg_resources.iter_entry_points("myapp.plugins"):
            plugin_class = entry_point.load()
            plugin = plugin_class()
            self.plugins[plugin.name] = plugin
    
    def get_plugin(self, name: str) -> PluginProtocol:
        """Get plugin by name."""
        if name not in self.plugins:
            raise ValueError(f"Plugin not found: {name}")
        return self.plugins[name]
```

---

## 5. Testing

### Command Testing
```python
# tests/test_commands.py
from typer.testing import CliRunner
from myapp.cli import app
import pytest

runner = CliRunner()

def test_create_user():
    """Test user creation command."""
    result = runner.invoke(
        app,
        [
            "users",
            "create",
            "testuser",
            "--email",
            "test@example.com"
        ]
    )
    assert result.exit_code == 0
    assert "User testuser created successfully" in result.stdout

def test_create_user_invalid_email():
    """Test user creation with invalid email."""
    result = runner.invoke(
        app,
        [
            "users",
            "create",
            "testuser",
            "--email",
            "invalid-email"
        ]
    )
    assert result.exit_code == 2
    assert "Invalid email format" in result.stdout

@pytest.mark.parametrize("format", ["table", "json", "csv"])
def test_list_users_format(format):
    """Test user list command with different formats."""
    result = runner.invoke(
        app,
        ["users", "list", "--format", format]
    )
    assert result.exit_code == 0
```

---

## Best Practices

1. **Command Design**
   - Use clear, descriptive command names
   - Provide sensible defaults
   - Implement proper help documentation
   - Support both interactive and non-interactive modes
   - Follow the principle of least surprise

2. **User Experience**
   - Provide clear feedback
   - Use colors and formatting judiciously
   - Implement progress indicators
   - Handle errors gracefully
   - Support shell completion

3. **Configuration**
   - Use configuration files
   - Support environment variables
   - Implement secure credential handling
   - Provide configuration validation
   - Support multiple environments

4. **Testing**
   - Test command-line parsing
   - Test input validation
   - Test output formatting
   - Mock external dependencies
   - Test error conditions

5. **Documentation**
   - Provide clear command help
   - Document configuration options
   - Include usage examples
   - Document installation process
   - Maintain changelog

---

## Conclusion

Following these CLI standards ensures:
- User-friendly command-line interfaces
- Robust error handling
- Consistent user experience
- Maintainable code structure
- Comprehensive testing

Remember to:
- Follow CLI best practices
- Provide comprehensive documentation
- Implement proper testing
- Handle errors gracefully
- Consider user experience

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at http://www.apache.org/licenses/LICENSE-2.0.
