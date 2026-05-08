param(
    [string]$Command = "help",
    [string]$Service = ""
)

$Services = @("Dataset1", "Dataset2", "Dataset3")

function Assert-Service {
    param([string]$Name)

    if (-not $Name) {
        Write-Host "Please choose a service: $($Services -join ', ')" -ForegroundColor Red
        exit 1
    }

    if ($Services -notcontains $Name) {
        Write-Host "Unknown service '$Name'. Choose one of: $($Services -join ', ')" -ForegroundColor Red
        exit 1
    }
}

function Invoke-Up {
    Write-Host "Starting services..." -ForegroundColor Green
    docker-compose up --build
}

function Invoke-Down {
    Write-Host "Stopping services..." -ForegroundColor Yellow
    docker-compose down
}

function Invoke-Restart {
    Invoke-Down
    Invoke-Up
}

function Invoke-Run {
    param([string]$Name)

    Assert-Service $Name
    Write-Host "Running $Name..." -ForegroundColor Green
    docker-compose run $Name
}

function Invoke-Logs {
    param([string]$Name)

    if ($Name) {
        Assert-Service $Name
        Write-Host "Showing logs for $Name..." -ForegroundColor Cyan
        docker-compose logs -f $Name
        return
    }

    Write-Host "Showing logs for all services..." -ForegroundColor Cyan
    docker-compose logs -f
}

function Invoke-Clean {
    Write-Host "This removes the project containers, images, and volumes." -ForegroundColor Red
    $Answer = Read-Host "Continue? (y/N)"

    if ($Answer -match "^[Yy]$") {
        docker-compose down --rmi all --volumes --remove-orphans
        Write-Host "Clean complete." -ForegroundColor Green
        return
    }

    Write-Host "Cancelled."
}

function Invoke-Status {
    docker-compose ps
}

function Invoke-Shell {
    param([string]$Name)

    Assert-Service $Name
    Write-Host "Opening shell in $Name..." -ForegroundColor Cyan
    docker-compose exec $Name /bin/bash

    if ($LASTEXITCODE -ne 0) {
        docker-compose exec $Name /bin/sh
    }
}

function Show-Help {
    Write-Host "Usage: .\dev.ps1 <command> [service]" -ForegroundColor Green
    Write-Host "Commands:" -ForegroundColor Green
    Write-Host "  up       Start and build all services" -ForegroundColor Green
    Write-Host "  down     Stop all services" -ForegroundColor Green
    Write-Host "  restart  Restart all services" -ForegroundColor Green
    Write-Host "  run      Run one service" -ForegroundColor Green
    Write-Host "  logs     Show logs" -ForegroundColor Green
    Write-Host "  clean    Remove project containers, images, and volumes" -ForegroundColor Green
    Write-Host "  status   Show container status" -ForegroundColor Green
    Write-Host "  shell    Open a shell in a service" -ForegroundColor Green
}

switch ($Command.ToLower()) {
    "up" { Invoke-Up }
    "down" { Invoke-Down }
    "restart" { Invoke-Restart }
    "run" { Invoke-Run -Name $Service }
    "logs" { Invoke-Logs -Name $Service }
    "clean" { Invoke-Clean }
    "status" { Invoke-Status }
    "shell" { Invoke-Shell -Name $Service }
    "help" { Show-Help }
    default {
        Write-Host "Unknown command '$Command'. Use 'help' to see the commands." -ForegroundColor Red
        exit 1
    }
}