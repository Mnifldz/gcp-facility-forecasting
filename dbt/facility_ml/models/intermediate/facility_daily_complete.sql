WITH calendar AS (

    SELECT
        day AS booking_date

    FROM UNNEST(
        GENERATE_DATE_ARRAY(
            DATE '2025-01-01',
            DATE '2025-12-31'
        )
    ) AS day

),

facilities AS (

    SELECT DISTINCT
        facility_id

    FROM {{ source('storage_raw', 'facilities') }}

),

complete_calendar AS (

    SELECT
        f.facility_id,
        c.booking_date

    FROM facilities f
    CROSS JOIN calendar c

)

SELECT
    c.facility_id,
    c.booking_date,

    COALESCE(
        d.new_bookings,
        0
    ) AS new_bookings,

    d.avg_booking_price,

    COALESCE(
        d.booked_monthly_revenue,
        0
    ) AS booked_monthly_revenue

FROM complete_calendar c

LEFT JOIN
    {{ source('storage_analytics', 'daily_facility_metrics') }} d

ON
    c.facility_id = d.facility_id
    AND c.booking_date = d.booking_date