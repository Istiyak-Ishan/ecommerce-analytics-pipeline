CREATE TABLE IF NOT EXISTS dim_products (
    product_id            TEXT    PRIMARY KEY,
    category_name         TEXT,
    category_english      TEXT,
    product_name_length   INTEGER,
    product_desc_length   INTEGER,
    photos_qty            INTEGER,
    weight_g              REAL,
    length_cm             REAL,
    height_cm             REAL,
    width_cm              REAL,
    price                 REAL,
    is_active             INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_prod_category ON dim_products(category_english);