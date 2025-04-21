@echo on
setlocal enabledelayedexpansion

SET MS_CONVERT="C:\Users\iyer95\AppData\Local\Apps\ProteoWizard 3.0.23054.b585bc2 64-bit\msconvert.exe"

echo Converting all .wiff files in raw_data directory...
for %%f in ("C:\Users\iyer95\OneDrive - purdue.edu\Desktop\MSConvert\raw_data\*.wiff") do (
    echo Processing %%f...
    %MS_CONVERT% ^
        --text ^
        --zlib ^
        --filter "peakPicking vendor msLevel=1-2" ^
        --filter "titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> File:<SourcePath>, NativeID:<Id>" ^
        -o "C:\Users\iyer95\OneDrive - purdue.edu\Desktop\MSConvert\converted_files" ^
        "%%f"
    
    set "EXIT_CODE=!ERRORLEVEL!"
    if !EXIT_CODE! EQU 0 (
        echo Successfully converted %%f
    ) else (
        echo Failed to convert %%f with error code !EXIT_CODE!
    )
)

echo All conversions complete.