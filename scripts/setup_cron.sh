#!/bin/bash

# Setup cron job for nightly index refresh
# This script sets up a cron job to refresh the QueryGenie index every night

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Make the refresh script executable
chmod +x "$SCRIPT_DIR/refresh_index.py"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Add cron job (runs every day at 2 AM)
# The cron job will:
# 1. Change to the project directory
# 2. Run the refresh script
# 3. Log output to the logs directory

CRON_JOB="0 2 * * * cd $PROJECT_DIR && python3 $SCRIPT_DIR/refresh_index.py >> $PROJECT_DIR/logs/cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "refresh_index.py"; then
    echo "Cron job already exists for QueryGenie refresh"
    echo "Current cron jobs:"
    crontab -l | grep "refresh_index.py"
else
    # Add the cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "Cron job added successfully!"
    echo "QueryGenie index will refresh every day at 2:00 AM"
fi

echo ""
echo "Cron job details:"
echo "Command: $CRON_JOB"
echo "Logs will be saved to: $PROJECT_DIR/logs/cron.log"
echo ""
echo "To view current cron jobs: crontab -l"
echo "To remove the cron job: crontab -e (then delete the line)"
echo "To test the refresh script: python3 $SCRIPT_DIR/refresh_index.py --test"
