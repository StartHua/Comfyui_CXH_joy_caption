@echo off

set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR%../../../python_embeded"


python.exe -m pip install -r "%SCRIPT_DIR%requirements.txt"

pause
