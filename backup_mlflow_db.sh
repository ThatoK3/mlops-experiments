#!/bin/bash

# Load environment variables from .env
set -a
source .env
set +a

# Validate required environment variables
if [ -z "$MYSQL_ROOT_USER" ] || [ -z "$MYSQL_HOST" ]; then
  echo "Error: MYSQL_ROOT or MYSQL_HOST is not set in .env"
  exit 1
fi

# Run mysqldump
echo "Starting MySQL backup..."
mysqldump \
  --single-transaction \
  --routines \
  --triggers \
  --events \
  -u "$MYSQL_ROOT_USER" \
  -p"$MYSQL_ROOT_PASSWORD" \
  -h "$MYSQL_HOST" \
  mlflow_db > mlflow_backup.sql

if [ $? -eq 0 ]; then
  echo "Backup completed successfully"
else
  echo "Backup failed!"
  exit 1
fi

