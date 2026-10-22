@echo off
rem Uso: quant.bat <url-do-huggingface>  (sem argumento, ele pergunta)
rem        quant.bat <url> --list-all   para ver todas as quants no menu
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python quant.py %*
if errorlevel 1 pause
