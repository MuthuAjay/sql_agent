#!/usr/bin/env python3
"""Setup script for SQL Agent development."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {command}")
    return subprocess.run(command, shell=True, check=check)


def main():
    """Main setup function."""
    print("🚀 Setting up SQL Agent development environment...")
    
    # Check if Poetry is installed
    try:
        run_command("poetry --version")
    except subprocess.CalledProcessError:
        print("❌ Poetry is not installed. Please install Poetry first:")
        print("   curl -sSL https://install.python-poetry.org | python3 -")
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    try:
        run_command("poetry install")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating .env file from template...")
        try:
            run_command("cp env.example .env")
            print("✅ Created .env file. Please edit it with your configuration.")
        except subprocess.CalledProcessError:
            print("❌ Failed to create .env file")
            sys.exit(1)
    else:
        print("✅ .env file already exists")
    
    # Install pre-commit hooks
    print("\n🔧 Installing pre-commit hooks...")
    try:
        run_command("poetry run pre-commit install")
    except subprocess.CalledProcessError:
        print("⚠️  Failed to install pre-commit hooks (optional)")
    
    # Run tests
    print("\n🧪 Running tests...")
    try:
        run_command("poetry run pytest tests/ -v")
    except subprocess.CalledProcessError:
        print("⚠️  Some tests failed (this is expected without proper configuration)")
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys and database configuration")
    print("2. Start the development environment:")
    print("   poetry run uvicorn sql_agent.api.main:app --reload")
    print("3. Or use Docker Compose:")
    print("   docker-compose up --build")


if __name__ == "__main__":
    main() 