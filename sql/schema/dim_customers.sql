CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id          TEXT    PRIMARY KEY,
    customer_unique_id   TEXT    NOT NULL,
    city                 TEXT,
    state                TEXT,
    zip_code             TEXT,
    first_purchase_date  DATE,
    customer_segment     TEXT    DEFAULT 'new'
);

CREATE INDEX IF NOT EXISTS idx_cust_state   ON dim_customers(state);
CREATE INDEX IF NOT EXISTS idx_cust_segment ON dim_customers(customer_segment);