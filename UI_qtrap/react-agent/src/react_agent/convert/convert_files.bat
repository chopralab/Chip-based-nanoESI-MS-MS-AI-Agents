@echo on
setlocal enabledelayedexpansion

SET MS_CONVERT="C:\Users\ChopraLab\AppData\Local\Apps\ProteoWizard 3.0.25155.379d87c 64-bit\msconvert.exe"

echo Converting all .wiff files in raw_data directory...
for %%f in ("C:\Users\ChopraLab\Desktop\laptop\convert_raw\raw_data\*.wiff") do (
    echo Processing %%f...
    %MS_CONVERT% ^
        --text ^
        --zlib ^
        --filter "peakPicking vendor msLevel=1-2" ^
        --filter "titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> File:<SourcePath>, NativeID:<Id>" ^
        -o "C:\Users\ChopraLab\Desktop\laptop\convert_raw\converted_files" ^
        "%%f"
    
    set "EXIT_CODE=!ERRORLEVEL!"
    if !EXIT_CODE! EQU 0 (
        echo Successfully converted %%f
    ) else (
        echo Failed to convert %%f with error code !EXIT_CODE!
    )
)

echo All conversions complete.