@echo off
chcp 65001 >nul
cd /d "%~dp0"
title MICRA

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo   Chua co moi truong ao. Dang tao va cai thu vien...
    echo   Buoc nay chi chay mot lan, mat 3-10 phut.
    echo.
    python -m venv .venv
    call ".venv\Scripts\activate.bat"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install google-genai matplotlib
) else (
    call ".venv\Scripts\activate.bat"
)

if not exist ".env" (
    if exist ".env.example" copy ".env.example" ".env" >nul
    echo   [!] Da tao file .env. Mo ra dien GEMINI_API_KEY neu muon dung LLM that.
)

cls
echo ==========================================================
echo    MICRA - MOI TRUONG AO DA BAT SAN
echo ==========================================================
echo.
echo    python chay_tat_ca.py      Kiem tra toan bo 36 muc
echo    python chan_doan.py        Chan doan ket noi API
echo    python -m src.train        Huan luyen mo hinh
echo    python kiem_tra_auc.py     Bao cao AUC chi tiet
echo    python kiem_thu.py         16 muc kiem thu
echo    python run_cli.py HKD023   Tham dinh mot ho so
echo    streamlit run app.py       Mo giao dien web
echo.
echo ==========================================================
echo.
cmd /k
