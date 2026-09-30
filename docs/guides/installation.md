# Installation Guide

This guide will walk you through the process of setting up the Predictive Analytics for Proactive Measures framework on your system.

## Prerequisites

Before installation, ensure you have the following:

1. **Python 3.10+**: The framework is built on Python 3.10 or higher
2. **Git**: To clone the repository
3. **Conda** (recommended) or **virtualenv**: For environment management
4. **Alpha Vantage API Key**: Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/MontyCraig/Predictive-Analytics-for-Proactive-Measures.git
cd Predictive-Analytics-for-Proactive-Measures
```

### Step 2: Create a Conda Environment

**Option A: Using Conda (Recommended)**

```bash
conda create -n predictive_analytics python=3.10
conda activate predictive_analytics
```

**Option B: Using virtualenv**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\\Scripts\\activate
```

### Step 3: Install Dependencies

```bash
pip install -e ".[dev]"
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
cp .env-example .env
```

Edit the `.env` file and add your Alpha Vantage API key:

```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### Step 5: Verify Installation

Run the following command to verify your installation:

```bash
predictive-analytics --help
```

You should see the help message with available commands and options.

## Installation with Docker

If you prefer using Docker, we provide a Docker configuration:

### Step 1: Build the Docker Image

```bash
docker build -t predictive-analytics .
```

### Step 2: Run the Container

```bash
docker run -it --env ALPHA_VANTAGE_API_KEY=your_api_key_here predictive-analytics
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**

   If you encounter errors about missing packages, try reinstalling the dependencies:

   ```bash
   pip install -e ".[dev]" --force-reinstall
   ```

2. **API Key Issues**

   If you encounter errors related to the Alpha Vantage API, make sure:
   - You have set the API key correctly in the `.env` file
   - Your API key is valid and hasn't reached its request limit

3. **Environment Activation Problems**

   If you're having issues with your environment:
   
   ```bash
   # For Conda
   conda deactivate
   conda activate predictive_analytics
   
   # For virtualenv
   deactivate
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

### Getting Help

If you continue to experience issues, please:

1. Check the [project documentation](../index.md)
2. Look for similar issues in our issue tracker
3. Create a new issue with details about your problem
4. Reach out to the maintainers directly

## Next Steps

After installation, you may want to:

- Read the [Basic Usage Guide](./basic_usage.md)
- Explore [Advanced Configuration](./advanced_configuration.md)
- Check out the [API Documentation](../api/index.md) 