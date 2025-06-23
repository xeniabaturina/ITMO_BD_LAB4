-- Create the production schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS public;

-- Create the test schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS test;

-- Set permissions
GRANT ALL ON SCHEMA public TO PUBLIC;
GRANT ALL ON SCHEMA test TO PUBLIC;
