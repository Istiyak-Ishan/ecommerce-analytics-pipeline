CREATE TABLE IF NOT EXISTS dim_time (
    date_id      INTEGER      PRIMARY KEY,
    full_date    DATE         NOT NULL UNIQUE,
    year         INTEGER      NOT NULL,
    quarter      INTEGER      NOT NULL,
    month        INTEGER      NOT NULL,
    month_name   TEXT         NOT NULL,
    week         INTEGER      NOT NULL,
    day_of_week  TEXT         NOT NULL,
    is_weekend   INTEGER      NOT NULL DEFAULT 0,
    is_holiday   INTEGER      NOT NULL DEFAULT 0
);