CREATE TABLE IF NOT EXISTS dim_channels (
    channel_id    TEXT    PRIMARY KEY,
    channel_name  TEXT    NOT NULL,
    channel_type  TEXT,
    cac           REAL
);

INSERT OR IGNORE INTO dim_channels VALUES
    ('CH001', 'Google Ads',       'paid',     45.00),
    ('CH002', 'Organic Search',   'organic',   5.00),
    ('CH003', 'Email Campaign',   'owned',    12.00),
    ('CH004', 'Social Media Ads', 'paid',     38.00),
    ('CH005', 'Referral',         'referral', 18.00),
    ('CH006', 'Direct',           'direct',    0.00);