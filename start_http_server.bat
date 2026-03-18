@echo off
set AS_HELP_ROOT=C:\Program Files (x86)\BrAutomation\AS6\help-en\Data
set AS_HELP_DB_PATH=C:\projects\as-help-mcp\data\as6\.ashelp\search.db
set AS_HELP_METADATA_DIR=C:\projects\as-help-mcp\data\as6\.ashelp_metadata
set AS_HELP_VERSION=6
python "C:\projects\as-help-mcp\run_mcp.py" --http --port 3838
