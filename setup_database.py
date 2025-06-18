#!/usr/bin/env python3
"""Database setup script for SQL Agent."""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sql_agent.core.config import settings
from sql_agent.core.database import DatabaseManager
from sql_agent.utils.logging import get_logger


async def setup_database():
    """Set up the database with sample data."""
    logger = get_logger("db_setup")
    
    print("🗄️  Setting up SQL Agent Database")
    print("=" * 40)
    
    # Display configuration
    print(f"Database: {settings.database_url}")
    print()
    
    # Create database manager
    db_manager = DatabaseManager()
    
    try:
        print("🔄 Initializing database connection...")
        await db_manager.initialize()
        print("✅ Database connected")
        
        # Read and execute initialization SQL
        init_sql_file = Path("scripts/init.sql")
        if init_sql_file.exists():
            print(f"\n📄 Reading initialization script: {init_sql_file}")
            with open(init_sql_file, 'r') as f:
                init_sql = f.read()
            
            # Split SQL into individual statements
            statements = [stmt.strip() for stmt in init_sql.split(';') if stmt.strip()]
            
            print(f"Executing {len(statements)} SQL statements...")
            
            for i, statement in enumerate(statements, 1):
                if statement:
                    try:
                        result = await db_manager.execute_query(statement)
                        if result.error:
                            print(f"⚠️  Statement {i} had an issue: {result.error}")
                        else:
                            print(f"✅ Statement {i} executed successfully")
                    except Exception as e:
                        print(f"❌ Statement {i} failed: {e}")
            
            print("\n🎉 Database setup completed!")
            
            # Verify setup
            print("\n🔍 Verifying setup...")
            schema_info = await db_manager.get_schema_info()
            
            if schema_info:
                print(f"✅ Found {len(schema_info)} tables:")
                for table_name, table_info in schema_info.items():
                    print(f"   - {table_name}: {len(table_info['columns'])} columns")
                
                # Show sample data
                for table_name in schema_info.keys():
                    try:
                        sample_data = await db_manager.get_sample_data(table_name, limit=2)
                        if sample_data:
                            print(f"   📊 {table_name}: {len(sample_data)} sample records")
                    except Exception as e:
                        print(f"   ⚠️  {table_name}: Could not retrieve sample data")
            else:
                print("⚠️  No tables found after setup")
                
        else:
            print(f"❌ Initialization script not found: {init_sql_file}")
            print("Please ensure scripts/init.sql exists")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure PostgreSQL is running")
        print("2. Verify database 'sql_agent_db' exists")
        print("3. Check user permissions")
        print("4. Ensure .env file has correct DATABASE_URL")
        return False
        
    finally:
        # Clean up
        if db_manager:
            await db_manager.close()
            print("\n🔒 Database connection closed")


def main():
    """Main function."""
    print("🚀 SQL Agent Database Setup")
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
    
    # Check if init.sql exists
    init_file = Path("scripts/init.sql")
    if not init_file.exists():
        print("❌ scripts/init.sql not found!")
        print("Please ensure the initialization script exists.")
        return
    
    print("✅ Initialization script found")
    
    # Confirm setup
    print(f"\nThis will set up the database: {settings.database_url}")
    print("This may create tables and insert sample data.")
    
    confirm = input("\nContinue? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Setup cancelled.")
        return
    
    # Run setup
    success = asyncio.run(setup_database())
    
    if success:
        print("\n🎉 Database setup completed successfully!")
        print("You can now run the SQL Agent application.")
    else:
        print("\n❌ Database setup failed.")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main() 