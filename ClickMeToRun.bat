@echo off

SET PYTHON_PATH_Loc=Python\python.exe
SET PYTHON_PATH_Sys=C:\Python35\python.exe

echo Starting SkyEye...

:Local
%PYTHON_PATH_Loc%  run.py || goto :error
echo Stared Local...
exit

:Sys
%PYTHON_PATH_Sys%  run.py || goto :error
echo Stared System...
exit

:error
echo Environment is NOT ready
echo Please prepare the environment.
set /p DUMMY=Hit ENTER to exit...