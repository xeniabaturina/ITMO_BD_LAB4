-- Create the production schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS public;

-- Create the test schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS test;

-- Set permissions
GRANT ALL ON SCHEMA public TO current_user;
GRANT ALL ON SCHEMA test TO current_user;

-- Set search path to include both schemas
ALTER DATABASE CURRENT_USER SET search_path TO public, test;
