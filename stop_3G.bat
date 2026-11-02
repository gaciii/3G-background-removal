@echo off
setlocal
cd /d "%~dp0"
echo [3G] Dang tim may chu 3G tren cong 7866...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=Get-NetTCPConnection -LocalPort 7866 -State Listen -ErrorAction SilentlyContinue; foreach($id in @($c|Select-Object -ExpandProperty OwningProcess -Unique)){if(!$id){continue}; $pr=Get-CimInstance Win32_Process -Filter ('ProcessId='+$id) -ErrorAction SilentlyContinue; if($pr.CommandLine -match 'app\.py' -and $pr.ExecutablePath -match 'python'){Stop-Process -Id $id -Force; Write-Host '[3G] Da dung may chu.'}else{Write-Host '[3G] Cong 7866 co tien trinh khac, khong dung nham.'}}"
echo [3G] Hoan tat.
endlocal
