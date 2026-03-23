param(
    [string]$Command = "help",
    [string]$Service = ""
)

$Services = @("Dataset1", "Dataset2", "Dataset3")


function Assert-Service{
    param([string]$Svc)
    if(-not $Svc){
        Write-Host "Please specify a service. Available services: $($Services -join ', ')" -ForegroundColor Red
        exit 1
    }
    if ($Services -notcontains $Svc) {
        Write-Host "Invalid service '$Svc'. Available services: $($Services -join ', ')" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Up {
    Write-Host "Starting and building all services..." -ForegroundColor Green
    docker-compose up --build
}

function Invoke-Down {
    Write-Host "Stopping all services..." -ForegroundColor Yellow
    docker-compose down
}

function Invoke-Restart {
    Write-Host "Restarting all services..." -ForegroundColor Yellow
    docker-compose down
    docker-compose up --build
}

function Invoke-Run{
    param([string]$Svc)
    Assert-Service $Svc
    Write-Host "Running service '$Svc'..." -ForegroundColor Green
    docker-compose run $Svc
}

function Invoke-Logs {
  param([string]$Svc)
  if (-not $Svc) {
    Write-Host "Tailing logs for all services..." -ForegroundColor Cyan
    docker-compose logs -f
  } else {
    Assert-Service $Svc
    Write-Host "Tailing logs for $Svc..." -ForegroundColor Cyan
    docker-compose logs -f $Svc
  }
}

function Invoke-Clean {
  Write-Host "WARNING: This will remove all containers, images, and volumes." -ForegroundColor Red
  $confirm = Read-Host "Are you sure? (y/N)"
  if ($confirm -match "^[Yy]$") {
    Write-Host "Cleaning up..." -ForegroundColor Red
    docker-compose down --rmi all --volumes --remove-orphans
    Write-Host "Done." -ForegroundColor Green
  } else {
    Write-Host "Cancelled."
  }
}

function Invoke-Status {
  Write-Host "Container status:" -ForegroundColor Cyan
  docker-compose ps
}
 
function Invoke-Shell {
  param([string]$Svc)
  Assert-Service $Svc
  Write-Host "Opening shell in $Svc..." -ForegroundColor Cyan
  # Try bash first, fall back to sh
  docker-compose exec $Svc /bin/bash
  if ($LASTEXITCODE -ne 0) {
    docker-compose exec $Svc /bin/sh
  }
}

switch ($Command.ToLower()) {
    "up" { Invoke-Up }
    "down" { Invoke-Down }
    "restart" { Invoke-Restart }
    "run" { Invoke-Run -Svc $Service }
    "logs" { Invoke-Logs -Svc $Service }
    "clean" { Invoke-Clean }
    "status" { Invoke-Status }
    "shell" { Invoke-Shell -Svc $Service }
    "help" {
        Write-Host "Usage: .\dev.ps1 <command> [service]" -ForegroundColor Green
        Write-Host "Commands:" -ForegroundColor Green
        Write-Host "  up       - Start and build all services" -ForegroundColor Green
        Write-Host "  down     - Stop all services" -ForegroundColor Green
        Write-Host "  restart  - Restart all services" -ForegroundColor Green
        Write-Host "  run      - Run a one-off command in a service (requires service name)" -ForegroundColor Green
        Write-Host "  logs     - Tail logs for a service or all services (optional service name)" -ForegroundColor Green
        Write-Host "  clean    - Remove all containers, images, and volumes (with confirmation)" -ForegroundColor Green
        Write-Host "  status   - Show container status" -ForegroundColor Green
        Write-Host "  shell    - Open a shell in a service (requires service name)" -ForegroundColor Green
    }
    default {
        Write-Host "Unknown command '$Command'. Use 'help' for usage information." -ForegroundColor Red
        exit 1
    }
}