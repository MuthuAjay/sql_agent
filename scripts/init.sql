-- SQL Agent Database Initialization Script

-- Create sample tables for testing
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revenue DECIMAL(10,2) DEFAULT 0.00
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    stock_quantity INTEGER DEFAULT 0
);

-- Insert sample data
INSERT INTO customers (name, email, revenue) VALUES
    ('John Doe', 'john@example.com', 1500.00),
    ('Jane Smith', 'jane@example.com', 2300.00),
    ('Bob Johnson', 'bob@example.com', 800.00),
    ('Alice Brown', 'alice@example.com', 3200.00),
    ('Charlie Wilson', 'charlie@example.com', 1100.00);

INSERT INTO products (name, price, category, stock_quantity) VALUES
    ('Laptop', 999.99, 'Electronics', 50),
    ('Mouse', 29.99, 'Electronics', 100),
    ('Keyboard', 79.99, 'Electronics', 75),
    ('Desk Chair', 199.99, 'Furniture', 25),
    ('Coffee Mug', 9.99, 'Kitchen', 200);

INSERT INTO orders (customer_id, total_amount, status) VALUES
    (1, 1029.98, 'completed'),
    (2, 79.99, 'completed'),
    (3, 209.98, 'pending'),
    (4, 199.99, 'completed'),
    (5, 39.98, 'shipped');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category); 