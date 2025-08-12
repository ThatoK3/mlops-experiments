#!/bin/bash

# Load environment variables from .env
set -a
source .env
set +a

# Validate required environment variables
if [ -z "$MYSQL_ROOT_USER" ] || [ -z "$MYSQL_HOST" ]; then
  echo "Error: MYSQL_ROOT_USER or MYSQL_HOST is not set in .env"
  exit 1
fi

# Path to backup file
BACKUP_FILE="mlflow_backup.sql"

if [ ! -f "$BACKUP_FILE" ]; then
  echo "Error: Backup file '$BACKUP_FILE' not found!"
  exit 1
fi

echo "Restoring database from $BACKUP_FILE..."

# Drop existing database
mysql -u root -p"$MYSQL_ROOT_PASSWORD" -h "$MYSQL_HOST" -e "DROP DATABASE IF EXISTS mlflow_db;"

# Create new database
mysql -u root -p"$MYSQL_ROOT_PASSWORD" -h "$MYSQL_HOST" -e "CREATE DATABASE mlflow_db;"

# Restore from backup file
mysql -u root -p"$MYSQL_ROOT_PASSWORD" -h "$MYSQL_HOST" mlflow_db < "$BACKUP_FILE"

if [ $? -eq 0 ]; then
  echo "Database restore completed successfully."
else
  echo "Database restore failed!"
  exit 1
fi

