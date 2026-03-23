# CreditcardFraudDetectionProj

## Development Commands

This project uses a PowerShell script (`dev.ps1`) to manage Docker services for each dataset. Run the following commands from the project root in a PowerShell terminal:

### Usage

```
./dev.ps1 <command> [service]
```

### Available Commands

- `up`       - Start and build all services
- `down`     - Stop all services
- `restart`  - Restart all services
- `run`      - Run a one-off command in a service (requires service name)
- `logs`     - Tail logs for a service or all services (optional service name)
- `clean`    - Remove all containers, images, and volumes (with confirmation)
- `status`   - Show container status
- `shell`    - Open a shell in a service (requires service name)

**Example:**

```
./dev.ps1 run Dataset1
```

### Services

- Dataset1
- Dataset2
- Dataset3

## Troubleshooting

If you get an error about running scripts (e.g., "running scripts is disabled on this system"), you need to allow script execution for your user. Run this command in a PowerShell terminal:

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try running the script again.