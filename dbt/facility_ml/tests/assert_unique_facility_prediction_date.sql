SELECT
    facility_id,
    prediction_date,
    COUNT(*) AS duplicate_count

FROM {{ ref('facility_forecasting_features') }}

GROUP BY
    facility_id,
    prediction_date

HAVING COUNT(*) > 1
