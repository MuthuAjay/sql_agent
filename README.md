# SQL Agent

An AI-powered SQL Agent with multi-agent architecture, RAG integration, and MCP (Model Context Protocol) support for natural language database interactions.

## 🚀 Features

### ✅ Phase 1: Foundation & Core Infrastructure
- **Project Setup**: Poetry configuration, Docker setup, environment management
- **Core Infrastructure**: Configuration, state management, LLM providers, database management
- **Development Environment**: PostgreSQL, ChromaDB, sample data, structured logging

### ✅ Phase 2: Multi-Agent System
- **Router Agent**: Intelligent query routing with LLM-based intent analysis
- **SQL Agent**: Natural language to SQL conversion with validation and execution
- **Analysis Agent**: Statistical analysis, business insights, and data quality scoring
- **Visualization Agent**: Automatic chart selection and data visualization
- **Agent Orchestrator**: LangGraph-based workflow coordination

### ✅ Phase 4: MCP Integration
- **MCP Server**: Standard Model Context Protocol implementation
- **Database Tools**: Query execution, sample data retrieval, SQL validation
- **Schema Tools**: Table listing, column information, schema search, relationship detection
- **Visualization Tools**: Chart creation, export, and data analysis
- **11 Specialized Tools**: Complete MCP toolset for database operations

### 🚧 Phase 3: RAG & Context Management (In Progress)
- Vector database integration (ChromaDB/Qdrant)
- Schema embedding and retrieval
- Context-aware query processing

### 🚧 Phase 5: API & User Interface (Planned)
- FastAPI REST API
- Web interface
- Real-time query processing

## 📋 MCP Tools Available

### Database Tools
- `execute_query`: Execute SQL queries with validation and formatting
- `get_sample_data`: Retrieve sample data from tables
- `validate_sql`: Validate SQL syntax without execution

### Schema Tools
- `get_tables`: List all tables with column counts
- `get_columns`: Get detailed column information for specific tables
- `search_schema`: Search schema by keywords (tables and columns)
- `get_relationships`: Detect potential foreign key relationships

### Visualization Tools
- `create_chart`: Create data visualizations with configuration
- `get_chart_types`: Get available chart types and descriptions
- `export_chart`: Export charts in various formats (JSON, HTML, PNG, SVG)
- `analyze_data_for_visualization`: Suggest appropriate chart types based on data

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Poetry
- Docker and Docker Compose

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd sql_agent

# Install dependencies
poetry install

# Start development environment
docker-compose up -d

# Run the setup script
./scripts/setup.sh

# Test the installation
poetry run pytest
```

### Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Configure your environment variables
DATABASE_URL=postgresql+asyncpg://user:password@localhost/sql_agent
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
```

## 🚀 Usage

### Multi-Agent System
```python
import asyncio
from sql_agent.agents import AgentOrchestrator

async def main():
    orchestrator = AgentOrchestrator()
    
    # Process a natural language query
    result = await orchestrator.process_query(
        "Show me the top 10 customers by revenue"
    )
    
    print(f"Generated SQL: {result.generated_sql}")
    print(f"Results: {result.query_result.row_count} rows")
    print(f"Insights: {len(result.analysis_result.insights)}")
    print(f"Chart: {result.visualization_config.chart_type}")

asyncio.run(main())
```

### MCP Integration
```python
import asyncio
from sql_agent.mcp import MCPClient

async def main():
    client = MCPClient()
    
    # Execute a query
    result = await client.call_tool("execute_query", {
        "sql": "SELECT * FROM customers LIMIT 5"
    })
    print(result)
    
    # Get schema information
    result = await client.call_tool("get_tables", {})
    print(result)
    
    # Create a visualization
    data = json.dumps([
        {"name": "John", "value": 100},
        {"name": "Jane", "value": 200}
    ])
    
    result = await client.call_tool("create_chart", {
        "chart_type": "bar",
        "data": data,
        "title": "Sample Chart"
    })
    print(result)

asyncio.run(main())
```

### Running Examples
```bash
# Multi-agent system example
poetry run python examples/multi_agent_example.py

# MCP integration example
poetry run python examples/mcp_example.py
```

## 🧪 Testing

### Run All Tests
```bash
poetry run pytest
```

### Run Specific Test Suites
```bash
# Multi-agent system tests
poetry run pytest tests/test_agents.py -v

# MCP integration tests
poetry run pytest tests/test_mcp.py -v

# Core functionality tests
poetry run pytest tests/test_core.py -v
```

### Test Coverage
```bash
poetry run pytest --cov=sql_agent --cov-report=html
```

## 📊 Project Status

| Phase | Status | Progress | Key Features |
|-------|--------|----------|--------------|
| 1 | ✅ Complete | 100% | Foundation, core infrastructure |
| 2 | ✅ Complete | 100% | Multi-agent system (4 agents + orchestrator) |
| 3 | 🚧 In Progress | 0% | RAG integration, vector database |
| 4 | ✅ Complete | 100% | MCP integration (11 tools) |
| 5 | 🚧 Planned | 0% | REST API, web interface |
| 6 | 🚧 Planned | 0% | Security, performance optimization |
| 7 | 🚧 Planned | 0% | Testing, documentation |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Router Agent   │───▶│  SQL Agent      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Analysis Agent  │    │ MCP Database    │
                       └─────────────────┘    │    Server       │
                                │              └─────────────────┘
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │Visualization    │    │ Vector Database │
                       │   Agent         │    │  (ChromaDB/     │
                       └─────────────────┘    │   Qdrant)       │
                                              └─────────────────┘
```

## 📚 Documentation

- [Project Plan](docs/PROJECT_PLAN.md) - Comprehensive development roadmap
- [Phase 2 Summary](docs/PHASE2_SUMMARY.md) - Multi-agent system details
- [Phase 4 Summary](docs/PHASE4_SUMMARY.md) - MCP integration details
- [API Documentation](docs/API.md) - REST API reference (coming soon)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the multi-agent framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [MCP](https://modelcontextprotocol.io/) for the Model Context Protocol standard
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Pydantic](https://pydantic.dev/) for data validation

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the examples in the `examples/` folder

---

**SQL Agent** - Making database interactions as natural as conversation. 🗄️✨ 