# Installation Guide

This guide will walk you through the process of setting up the Predictive Analytics for Proactive Measures framework on your system.

## Prerequisites

Before installation, ensure you have the following:

<<<<<<< HEAD
1. **Python 3.9+**: The framework is built on Python 3.9 or higher

=======
1. **Python 3.10+**: The framework is built on Python 3.10 or higher
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5
2. **Git**: To clone the repository

3. **Conda** (recommended) or **virtualenv**: For environment management

4. **Alpha Vantage API Key**: Get a free API key from [Alpha Vantage](<https://www.alphavantage.co/support/#api-key)>

## Installation Steps

### Step 1: Clone the Repository

```bash
<<<<<<< HEAD
git clone <https://github.com/your-org/Predictive-Analytics-for-Proactive-Measures.git>
=======
git clone https://github.com/MontyCraig/Predictive-Analytics-for-Proactive-Measures.git
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5
cd Predictive-Analytics-for-Proactive-Measures

```text
### Step 2: Create a Conda Environment

**Option A: Using Conda (Recommended)**

```bash
conda create -n predictive_analytics python=3.10
conda activate predictive_analytics

```text
**Option B: Using virtualenv**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\\Scripts\\activate

```text
### Step 3: Install Dependencies

```bash
<<<<<<< HEAD
pip install -r requirements.txt
=======
pip install -e ".[dev]"
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
### Step 4: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
cp .env-example .env

```text
Edit the `.env` file and add your Alpha Vantage API key:

```text
ALPHA_VANTAGE_API_KEY=your_api_key_here

```text
### Step 5: Verify Installation

Run the following command to verify your installation:

```bash
<<<<<<< HEAD
python main.py --help
=======
predictive-analytics --help
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
You should see the help message with available commands and options.

## Installation with Docker

If you prefer using Docker, we provide a Docker configuration:

### Step 1: Build the Docker Image

```bash
docker build -t predictive-analytics .

```text
### Step 2: Run the Container

```bash
docker run -it --env ALPHA_VANTAGE_API_KEY=your_api_key_here predictive-analytics

```text
## Troubleshooting

### Common Issues

1. **Missing Dependencies**

   If you encounter errors about missing packages, try reinstalling the dependencies:

   ```bash
<<<<<<< HEAD
   pip install -r requirements.txt --force-reinstall
   ```text
=======
   pip install -e ".[dev]" --force-reinstall
   ```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

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

   ```text
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
