#!/usr/bin/env python3
"""
Simple Test Script to List All Tables from Database

This script tests the schema endpoint to list all tables.
Usage: python list_tables_test.py
"""

import requests
import json
import sys
from typing import Dict, Any


def test_list_tables(base_url: str = "http://localhost:8000"):
    """Test listing all tables from the database."""
    
    print("🔍 Testing SQL Agent - List Tables")
    print(f"📡 API URL: {base_url}")
    print("-" * 50)
    
    try:
        # Test 1: Health check first
        print("1. Testing API health...")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ API Status: {health_data.get('status', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test 2: List databases
        print("\n2. Testing database listing...")
        response = requests.get(f"{base_url}/api/v1/schema/databases", timeout=15)
        
        if response.status_code == 200:
            databases = response.json()
            print(f"   ✅ Found {len(databases)} database(s)")
            for db in databases:
                print(f"      - {db.get('name', 'unknown')} ({db.get('status', 'unknown')})")
        else:
            print(f"   ❌ Database listing failed: {response.status_code}")
            print(f"      Error: {response.text}")
            return False
        
        # Test 3: List tables (main test)
        print("\n3. Testing table listing...")
        response = requests.get(f"{base_url}/api/v1/schema/tables", timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            tables = data.get("tables", [])
            
            print(f"   ✅ Found {len(tables)} table(s)")
            print(f"   📊 Database: {data.get('database_name', 'unknown')}")
            print(f"   🔄 Extraction Method: {data.get('extraction_method', 'unknown')}")
            
            if tables:
                print("\n   📋 Tables:")
                print("   " + "-" * 80)
                print(f"   {'Name':<20} {'Columns':<10} {'Rows':<15} {'Size':<12} {'Description':<20}")
                print("   " + "-" * 80)
                
                for table in tables:
                    name = table.get("name", "unknown")[:19]
                    cols = str(table.get("column_count", 0))
                    rows = str(table.get("row_count", "unknown"))[:14]
                    size = format_bytes(table.get("size_bytes")) if table.get("size_bytes") else "unknown"
                    desc = (table.get("description", "")[:19] + "...") if len(table.get("description", "")) > 19 else table.get("description", "")
                    
                    print(f"   {name:<20} {cols:<10} {rows:<15} {size:<12} {desc:<20}")
                
                print("   " + "-" * 80)
            else:
                print("   ⚠️  No tables found")
            
            return True
            
        else:
            print(f"   ❌ Table listing failed: {response.status_code}")
            print(f"      Error: {response.text}")
            
            # Try to get more details about the error
            try:
                error_data = response.json()
                print(f"      Detail: {error_data.get('detail', 'No details')}")
            except:
                pass
                
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the SQL Agent API running?")
        print("   Try starting it with: uvicorn sql_agent.api.main:app --reload")
        return False
    
    except requests.exceptions.Timeout:
        print("❌ Request timed out. API might be slow or overloaded.")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def format_bytes(bytes_val: int) -> str:
    """Format bytes into human readable format."""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f}KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.1f}MB"
    else:
        return f"{bytes_val/(1024**3):.1f}GB"


def test_specific_table(base_url: str, table_name: str):
    """Test getting info for a specific table."""
    print(f"\n4. Testing specific table: {table_name}")
    
    try:
        response = requests.get(f"{base_url}/api/v1/schema/tables/{table_name}", timeout=15)
        
        if response.status_code == 200:
            table_info = response.json()
            print(f"   ✅ Table info retrieved")
            print(f"   📊 Columns: {len(table_info.get('columns', []))}")
            print(f"   📝 Description: {table_info.get('description', 'No description')}")
            
            # Show first few columns
            columns = table_info.get("columns", [])[:5]
            if columns:
                print("   🔍 Sample columns:")
                for col in columns:
                    print(f"      - {col.get('name', 'unknown')} ({col.get('type', 'unknown')})")
            
        elif response.status_code == 404:
            print(f"   ❌ Table '{table_name}' not found")
        else:
            print(f"   ❌ Failed to get table info: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error getting table info: {e}")


def test_sample_data(base_url: str, table_name: str):
    """Test getting sample data from a table."""
    print(f"\n5. Testing sample data from: {table_name}")
    
    try:
        response = requests.get(f"{base_url}/api/v1/schema/sample/{table_name}?limit=3", timeout=15)
        
        if response.status_code == 200:
            sample_data = response.json()
            data = sample_data.get("data", {})
            columns = data.get("columns", [])
            rows = data.get("rows", [])
            
            print(f"   ✅ Sample data retrieved")
            print(f"   📊 Columns: {len(columns)}")
            print(f"   📋 Rows: {len(rows)}")
            
            if columns and rows:
                print("   🔍 Sample data:")
                print("      Columns:", ", ".join(columns[:5]))
                print("      First row:", rows[0] if rows else "No data")
            
        else:
            print(f"   ❌ Failed to get sample data: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error getting sample data: {e}")


def main():
    """Main function."""
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    # Test table listing
    success = test_list_tables(base_url)
    
    if success:
        # Ask if user wants to test a specific table
        print("\n" + "="*50)
        table_name = input("Enter table name to test (or press Enter to skip): ").strip()
        
        if table_name:
            test_specific_table(base_url, table_name)
            test_sample_data(base_url, table_name)
    
    print("\n" + "="*50)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
        print("\nTroubleshooting tips:")
        print("1. Make sure SQL Agent API is running")
        print("2. Check if database is connected")
        print("3. Verify schema routes are working")
        print("4. Check API logs for errors")


if __name__ == "__main__":
    main()