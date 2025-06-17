# SQL Agent Quick Start Guide

Get up and running with the SQL Agent project in minutes!

## Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Docker & Docker Compose (optional, for full development environment)

## Quick Setup

### Option 1: Local Development (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sql_agent
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   
   This will:
   - Install all dependencies with Poetry
   - Create a `.env` file from the template
   - Install pre-commit hooks
   - Run initial tests

3. **Configure your environment**
   ```bash
   # Edit the .env file with your API keys
   nano .env
   ```
   
   At minimum, you need:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql+asyncpg://user:password@localhost/sql_agent
   ```

4. **Start the development server**
   ```bash
   poetry run uvicorn sql_agent.api.main:app --reload
   ```

### Option 2: Docker Development

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd sql_agent
   cp env.example .env
   # Edit .env with your configuration
   ```

2. **Start with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   
   This will start:
   - PostgreSQL database (port 5432)
   - ChromaDB vector database (port 8000)
   - SQL Agent application (port 8080)

## Testing the Setup

### 1. Health Check
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "0.1.0"
}
```

### 2. Basic Query Test
```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me the top 5 customers by revenue",
    "database": "sql_agent"
  }'
```

## Project Structure

```
sql_agent/
├── agents/           # Multi-agent system
│   ├── base.py      # Base agent abstract class
│   ├── router.py    # Router agent (TODO)
│   ├── sql.py       # SQL generation agent (TODO)
│   ├── analysis.py  # Analysis agent (TODO)
│   └── viz.py       # Visualization agent (TODO)
├── core/            # Core functionality
│   ├── config.py    # Configuration management ✅
│   ├── database.py  # Database abstractions ✅
│   ├── llm.py       # LLM provider management ✅
│   └── state.py     # State management ✅
├── mcp/             # MCP integration (TODO)
├── rag/             # RAG functionality (TODO)
├── api/             # REST API (TODO)
└── utils/           # Utilities
    └── logging.py   # Structured logging ✅
```

## Development Workflow

### 1. Code Quality
```bash
# Format code
poetry run black sql_agent/
poetry run isort sql_agent/

# Lint code
poetry run ruff check sql_agent/

# Type checking
poetry run mypy sql_agent/
```

### 2. Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=sql_agent

# Run specific test file
poetry run pytest tests/test_config.py
```

### 3. Database Operations
```bash
# Connect to PostgreSQL
docker exec -it sql_agent_postgres psql -U sql_agent_user -d sql_agent

# View sample data
SELECT * FROM customers LIMIT 5;
SELECT * FROM orders LIMIT 5;
SELECT * FROM products LIMIT 5;
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | Yes (if using OpenAI) |
| `GOOGLE_API_KEY` | Google API key | - | Yes (if using Google) |
| `DATABASE_URL` | Database connection string | - | Yes |
| `VECTOR_DB_URL` | Vector database URL | `http://localhost:8000` | No |
| `LLM_PROVIDER` | LLM provider to use | `openai` | No |
| `DEBUG` | Enable debug mode | `false` | No |

### Database Setup

The project includes sample data for testing:

- **customers**: Customer information with revenue
- **orders**: Order details with amounts and status
- **products**: Product catalog with prices and categories

## Common Issues

### 1. Poetry Installation
If Poetry is not installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Database Connection
If you can't connect to the database:
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs sql_agent_postgres
```

### 3. API Key Issues
If you get API key errors:
- Ensure your API key is correctly set in `.env`
- Check that the key has sufficient credits/quota
- Verify the key is for the correct provider (OpenAI/Google)

### 4. Port Conflicts
If ports are already in use:
```bash
# Check what's using the ports
lsof -i :8080
lsof -i :5432
lsof -i :8000

# Kill processes or change ports in docker-compose.yml
```

## Next Steps

1. **Explore the codebase**: Start with `sql_agent/core/` to understand the foundation
2. **Implement agents**: Begin with the Router Agent in `sql_agent/agents/router.py`
3. **Add RAG functionality**: Implement vector database integration
4. **Build the API**: Create FastAPI endpoints for the agents
5. **Add tests**: Write comprehensive tests for each component

## Getting Help

- **Documentation**: Check `docs/PROJECT_PLAN.md` for detailed architecture
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `poetry run pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

Happy coding! 🚀 