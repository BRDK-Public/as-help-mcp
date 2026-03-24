# Tests Directory

This directory contains the comprehensive test suite for the as-help-mcp project.

## Quick Start

```bash
# Install test dependencies
uv sync --group test

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test category
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/e2e/
```

## Directory Structure

- `unit/` - Unit tests for individual modules
  - `test_indexer.py` - Tests for XML parsing and indexing logic
  - `test_search_engine.py` - Tests for LanceDB hybrid search with RRF
  - `test_server.py` - Tests for MCP server tools
  
- `integration/` - Integration tests for components working together
  - `test_indexer_search.py` - Tests for indexer + search engine integration
  
- `e2e/` - End-to-end tests for complete workflows
  - `test_full_workflows.py` - Complete user journey tests
  
- `conftest.py` - Shared fixtures and test utilities

## Test Fixtures

### Temporary Help Directory
Creates a temporary directory with sample HTML files and brhelpcontent.xml:
```python
def test_something(temp_help_dir, sample_xml):
    # temp_help_dir has sample HTML files
    # sample_xml is the brhelpcontent.xml path
    pass
```

### Mock Indexer
Provides a pre-configured mock indexer with sample data:
```python
def test_with_mock(mock_indexer):
    # Use mock_indexer.pages, mock_indexer.help_id_map, etc.
    pass
```

### Initialized Indexer
Provides a real indexer with parsed XML:
```python
def test_with_real_indexer(initialized_indexer):
    # Full functional indexer
    pass
```

## Running Tests

### Run All Tests
```bash
uv run pytest
```

### Run with Verbose Output
```bash
uv run pytest -v
```

### Run Specific Test File
```bash
uv run pytest tests/unit/test_indexer.py
```

### Run Specific Test
```bash
uv run pytest tests/unit/test_indexer.py::TestXMLAttributeExtraction::test_process_section_full_format
```

### Run Tests Matching Pattern
```bash
uv run pytest -k "breadcrumb"
```

### Skip Slow Tests
```bash
uv run pytest -m "not slow"
```

### Stop on First Failure
```bash
uv run pytest -x
```

### Show Local Variables on Failure
```bash
uv run pytest -l
```

## Coverage Reports

### Generate HTML Coverage Report
```bash
uv run pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Terminal Coverage Report
```bash
uv run pytest --cov=src --cov-report=term-missing
```

### XML Coverage Report (for CI)
```bash
uv run pytest --cov=src --cov-report=xml
```

## Test Categories

### Unit Tests (Fast)
Test individual functions and methods in isolation using mocks.
- Run time: ~3-5 seconds
- Coverage focus: Custom business logic

### Integration Tests (Medium)
Test components working together with real implementations.
- Run time: ~10-20 seconds
- Coverage focus: Component interaction

### E2E Tests (Slow)
Test complete user workflows from start to finish.
- Run time: ~30-60 seconds
- Coverage focus: Full system behavior
- Some tests marked with `@pytest.mark.slow`

## Writing New Tests

### Test Naming Convention
- Test files: `test_<module>.py`
- Test classes: `Test<Feature>`
- Test methods: `test_<what_it_tests>`

### Example Unit Test
```python
def test_breadcrumb_simple_hierarchy(temp_help_dir, sample_xml):
    """Verify breadcrumb for a 3-level hierarchy."""
    indexer = HelpContentIndexer(temp_help_dir)
    indexer.parse_xml_structure()
    
    breadcrumb = indexer.get_breadcrumb("mc_moveabs_page")
    
    assert len(breadcrumb) == 3
    assert breadcrumb[0].text == "Motion"
    assert breadcrumb[1].text == "mapp Motion"
    assert breadcrumb[2].text == "MC_BR_MoveAbsolute"
```

### Example Integration Test
```python
def test_search_then_get_page_workflow(app_context):
    """Verify search_help -> get_page_by_id workflow."""
    # Search
    search_results = search_help(ctx, query="X20DI9371")
    assert search_results.total > 0
    
    # Get page
    page_content = get_page_by_id(
        page_id=search_results.results[0].page_id,
        ctx=ctx
    )
    assert page_content is not None
```

## Debugging Tests

### Run with PDB on Failure
```bash
uv run pytest --pdb
```

### Print Statements
```bash
uv run pytest -s
```

### Show Captured Output
```bash
uv run pytest --capture=no
```

## CI/CD Integration

Tests run automatically on:
- Every push to main
- All pull requests
- Multiple Python versions (3.12, 3.13)

See `.github/workflows/test.yml` for CI configuration.

## Troubleshooting

### Import Errors
Make sure you're running tests with `uv run pytest` to use the virtual environment.

### Fixture Not Found
Check that the fixture is defined in `conftest.py` or imported properly.

### Slow Tests
Use `-m "not slow"` to skip tests marked as slow.

### Coverage Too Low
Run `uv run pytest --cov=src --cov-report=html` and open the HTML report to see which lines need coverage.
