
## Basics
```bat
rem variable assignment
rem Do not use whitespace between te name and value
SET foo=bar

rem for loop
for %%f in (absa_transformers.py, absa_utils.py, get_asc_input.py) do copy /Y .\%%f \\redmond.corp.microsoft.com\data\absa\code\%%f
```