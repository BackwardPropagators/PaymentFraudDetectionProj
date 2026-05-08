# Payment Fraud Detection

This project runs three fraud-detection experiments, one per dataset. Each dataset has its own Docker image and Python entry point.

## Run The Project

Open PowerShell in the project root.

```powershell
./dev.ps1 up
```

Run one dataset directly:

```powershell
./dev.ps1 run Dataset1
./dev.ps1 run Dataset2
./dev.ps1 run Dataset3
```

Stop everything:

```powershell
./dev.ps1 down
```

## Commands

```text
up        Start and build all services
down      Stop all services
restart   Restart all services
run       Run one service
logs      Show logs
clean     Remove project containers, images, and volumes
status    Show container status
shell     Open a shell inside a service
help      Show command help
```

Valid services are `Dataset1`, `Dataset2`, and `Dataset3`.

## PowerShell Setup

If PowerShell blocks the script, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```