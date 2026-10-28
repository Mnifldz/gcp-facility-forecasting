WITH daily AS (

    SELECT
        facility_id,
        booking_date AS prediction_date,
        new_bookings

    FROM {{ ref('facility_daily_complete') }}

),

features AS (

    SELECT
        facility_id,
        prediction_date,

        -- Calendar features
        EXTRACT(DAYOFWEEK FROM prediction_date) AS day_of_week,
        EXTRACT(MONTH FROM prediction_date) AS month,

        -- Current and lagged demand
        new_bookings AS bookings_today,

        LAG(new_bookings, 1) OVER w AS bookings_lag_1,
        LAG(new_bookings, 7) OVER w AS bookings_lag_7,
        LAG(new_bookings, 14) OVER w AS bookings_lag_14,

        -- Rolling averages
        AVG(new_bookings) OVER (
            PARTITION BY facility_id
            ORDER BY prediction_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS bookings_rolling_7,

        AVG(new_bookings) OVER (
            PARTITION BY facility_id
            ORDER BY prediction_date
            ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
        ) AS bookings_rolling_28,

        -- Forecasting target: bookings over the next 7 days
        SUM(new_bookings) OVER (
            PARTITION BY facility_id
            ORDER BY prediction_date
            ROWS BETWEEN 1 FOLLOWING AND 7 FOLLOWING
        ) AS bookings_next_7_days,

        -- Check that the complete target window exists
        COUNT(new_bookings) OVER (
            PARTITION BY facility_id
            ORDER BY prediction_date
            ROWS BETWEEN 1 FOLLOWING AND 7 FOLLOWING
        ) AS future_days_available

    FROM daily

    WINDOW w AS (
        PARTITION BY facility_id
        ORDER BY prediction_date
    )

)

SELECT
    * EXCEPT(future_days_available)

FROM features

WHERE
    bookings_lag_14 IS NOT NULL
    AND future_days_available = 7