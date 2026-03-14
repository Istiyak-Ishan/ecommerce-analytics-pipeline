CREATE TABLE IF NOT EXISTS fact_orders (
    order_id           TEXT    PRIMARY KEY,
    customer_id        TEXT    NOT NULL,
    product_id         TEXT,
    seller_id          TEXT,
    date_id            INTEGER NOT NULL,
    channel_id         TEXT    DEFAULT 'CH006',
    quantity           INTEGER NOT NULL DEFAULT 1,
    unit_price         REAL    NOT NULL,
    freight_value      REAL    DEFAULT 0,
    discount_amount    REAL    DEFAULT 0,
    revenue            REAL,
    gross_profit       REAL,
    order_status       TEXT    DEFAULT 'delivered',
    payment_type       TEXT,
    installments       INTEGER DEFAULT 1,
    review_score       INTEGER,
    order_purchase_ts  TEXT,
    order_approved_ts  TEXT,
    order_delivered_ts TEXT,
    estimated_delivery TEXT,

    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id),
    FOREIGN KEY (product_id)  REFERENCES dim_products(product_id),
    FOREIGN KEY (date_id)     REFERENCES dim_time(date_id),
    FOREIGN KEY (channel_id)  REFERENCES dim_channels(channel_id)
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON fact_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date     ON fact_orders(date_id);
CREATE INDEX IF NOT EXISTS idx_orders_status   ON fact_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_orders_product  ON fact_orders(product_id);