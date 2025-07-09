@echo off
echo Activando entorno Visual Studio 2022...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Compilando con nvcc...
nvcc pseudoinversa.cu -o pseudoinversa

if %errorlevel% equ 0 (
    echo ✓ Compilacion exitosa!
    echo.
    echo Ejecutando programa...
    pseudoinversa.exe
    echo.
    echo Verificando resultado...
    if exist salida.sal (
        echo ✓ Archivo salida.sal generado
        type salida.sal
    ) else (
        echo ✗ No se genero salida.sal
    )
) else (
    echo ✗ Error en compilacion
)
pause
