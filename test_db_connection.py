#!/usr/bin/env python3
"""Database connection test script for SQL Agent."""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sql_agent.core.config import settings
from sql_agent.core.database import DatabaseManager
from sql_agent.utils.logging import get_logger


async def test_database_connection():
    """Test the database connection with current settings."""
    logger = get_logger("db_test")
    
    print("🔍 Testing Database Connection")
    print("=" * 40)
    
    # Display current configuration
    print(f"Database Type: {settings.database_type}")
    print(f"Database URL: {settings.database_url}")
    print(f"Pool Size: {settings.database_pool_size}")
    print(f"Max Overflow: {settings.database_max_overflow}")
    print()
    
    # Create database manager
    db_manager = DatabaseManager()
    
    try:
        print("🔄 Initializing database connection...")
        await db_manager.initialize()
        print("✅ Database connection established successfully!")
        
        # Test basic query
        print("\n🧪 Testing basic query...")
        result = await db_manager.execute_query("SELECT 1 as test_value")
        
        if result.error:
            print(f"❌ Query test failed: {result.error}")
            return False
        else:
            print(f"✅ Query test successful!")
            print(f"   Execution time: {result.execution_time:.3f}s")
            print(f"   Result: {result.data}")
        
        # Test schema information
        print("\n📋 Testing schema information...")
        schema_info = await db_manager.get_schema_info()
        
        if schema_info:
            print(f"✅ Schema information retrieved successfully!")
            print(f"   Tables found: {len(schema_info)}")
            
            for table_name, table_info in schema_info.items():
                print(f"   - {table_name}: {len(table_info['columns'])} columns")
        else:
            print("⚠️  No schema information available (database might be empty)")
        
        # Test sample data if tables exist
        if schema_info:
            first_table = list(schema_info.keys())[0]
            print(f"\n📊 Testing sample data from '{first_table}'...")
            
            try:
                sample_data = await db_manager.get_sample_data(first_table, limit=3)
                if sample_data:
                    print(f"✅ Sample data retrieved successfully!")
                    print(f"   Records found: {len(sample_data)}")
                    print(f"   Sample: {sample_data[0] if sample_data else 'No data'}")
                else:
                    print("ℹ️  Table exists but contains no data")
            except Exception as e:
                print(f"⚠️  Could not retrieve sample data: {e}")
        
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure PostgreSQL is running on localhost:5432")
        print("2. Verify the database 'sql_agent_db' exists")
        print("3. Check that user 'postgres' with password 'yourpassword' has access")
        print("4. Ensure the .env file is in the project root with correct DATABASE_URL")
        print("5. If using Docker, run: docker-compose up postgres")
        return False
        
    finally:
        # Clean up
        if db_manager:
            await db_manager.close()
            print("\n🔒 Database connection closed")


async def test_with_docker():
    """Test database connection using Docker setup."""
    print("\n🐳 Testing with Docker setup...")
    print("=" * 40)
    
    print("Starting PostgreSQL container...")
    print("Run this command in another terminal:")
    print("docker-compose up postgres -d")
    print()
    
    print("Waiting for database to be ready...")
    print("(This may take a few seconds)")
    
    # Wait a bit for Docker to start
    await asyncio.sleep(5)
    
    # Test connection
    return await test_database_connection()


def main():
    """Main function."""
    print("🚀 SQL Agent Database Connection Test")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your database configuration.")
        print("You can copy from env.example:")
        print("cp env.example .env")
        return
    
    print("✅ .env file found")
    
    # Ask user preference
    print("\nChoose test method:")
    print("1. Test with local PostgreSQL (if already running)")
    print("2. Test with Docker setup")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(test_with_docker())
    else:
        asyncio.run(test_database_connection())


if __name__ == "__main__":
    main() 