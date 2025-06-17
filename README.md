# SQL Agent

An AI-powered SQL Agent with multi-agent architecture, RAG (Retrieval-Augmented Generation), and MCP (Model Context Protocol) integration.

## Features

- **Multi-Agent System**: Router, SQL, Analysis, and Visualization agents working together
- **Natural Language to SQL**: Convert user queries to optimized SQL statements
- **RAG Integration**: Intelligent context retrieval using vector databases
- **MCP Support**: Model Context Protocol for database interactions
- **Multiple LLM Providers**: Support for OpenAI, Google, and local models
- **Database Support**: PostgreSQL, MySQL, SQLite with async operations
- **Vector Databases**: ChromaDB and Qdrant for schema context storage
- **REST API**: FastAPI-based endpoints for easy integration
- **Security**: Query validation, rate limiting, and audit logging

## Architecture

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

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Docker (optional, for development)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sql_agent
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
poetry run uvicorn sql_agent.api.main:app --reload
```

## Configuration

Create a `.env` file with the following variables:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
DATABASE_TYPE=postgresql

# Vector Database
VECTOR_DB_TYPE=chromadb
VECTOR_DB_URL=http://localhost:8000

# MCP Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=3000

# Security
RATE_LIMIT_PER_MINUTE=100
QUERY_TIMEOUT_SECONDS=30
MAX_ROWS_RETURNED=1000
```

## Usage

### REST API

The SQL Agent provides a REST API with the following endpoints:

- `POST /api/v1/query` - Convert natural language to SQL and execute
- `POST /api/v1/sql/generate` - Generate SQL from natural language
- `POST /api/v1/sql/execute` - Execute SQL query
- `POST /api/v1/analyze` - Analyze query results
- `POST /api/v1/visualize` - Create visualizations

### Example Usage

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/query",
        json={
            "query": "Show me the top 10 customers by revenue",
            "database": "sales_db"
        }
    )
    
    result = response.json()
    print(f"Generated SQL: {result['sql']}")
    print(f"Results: {result['results']}")
```

### MCP Integration

The SQL Agent can be used as an MCP server:

```bash
poetry run python -m sql_agent.mcp.server
```

## Development

### Project Structure

```
sql_agent/
├── agents/           # Multi-agent system
│   ├── base.py      # Base agent abstract class
│   ├── router.py    # Router agent
│   ├── sql.py       # SQL generation agent
│   ├── analysis.py  # Analysis agent
│   └── viz.py       # Visualization agent
├── core/            # Core functionality
│   ├── config.py    # Configuration management
│   ├── database.py  # Database abstractions
│   ├── llm.py       # LLM provider management
│   └── state.py     # State management
├── mcp/             # MCP integration
│   ├── server.py    # MCP server implementation
│   └── tools.py     # MCP tool definitions
├── rag/             # RAG functionality
│   ├── embeddings.py # Embedding services
│   ├── vector_store.py # Vector database operations
│   └── schema.py    # Schema context management
├── api/             # REST API
│   ├── main.py      # FastAPI application
│   ├── routes/      # API routes
│   └── models.py    # API models
└── utils/           # Utilities
    ├── logging.py   # Structured logging
    ├── security.py  # Security utilities
    └── validation.py # Input validation
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=sql_agent

# Run specific test file
poetry run pytest tests/test_agents.py
```

### Code Quality

```bash
# Format code
poetry run black sql_agent/
poetry run isort sql_agent/

# Lint code
poetry run ruff check sql_agent/

# Type checking
poetry run mypy sql_agent/
```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run specific services
docker-compose up database vector-db
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

- All SQL queries are validated and sanitized
- Rate limiting is implemented for API endpoints
- Audit logging tracks all operations
- Environment variables are used for sensitive configuration
- Input validation prevents injection attacks

## Performance

- Async/await for all I/O operations
- Connection pooling for database connections
- Caching for frequently accessed schema information
- Streaming for large query results
- Background tasks for vector indexing

## Monitoring

The application includes comprehensive logging and monitoring:

- Structured logging with request IDs
- Health checks for all external services
- Performance metrics collection
- Error tracking and alerting
- Query execution time monitoring 