$pythonw = 'C:\Users\Admin\AppData\Local\Programs\Python\Python312\pythonw.exe'
$args = 'C:\projects\as-help-mcp\run_mcp.py --http --port 3838 --help-root "C:\Program Files (x86)\BrAutomation\AS6\help-en\Data" --db-path C:\projects\as-help-mcp\data\as6\.ashelp\search.db --metadata-dir C:\projects\as-help-mcp\data\as6\.ashelp_metadata --as-version 6'

$action = New-ScheduledTaskAction -Execute $pythonw -Argument $args -WorkingDirectory 'C:\projects\as-help-mcp'
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 0) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName 'as-help-mcp' -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force
Write-Host "Done. Task registered."
Get-ScheduledTask -TaskName 'as-help-mcp' | Select-Object TaskName, State
