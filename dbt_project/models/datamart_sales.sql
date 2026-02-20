{{ config(materialized='table') }}

with source_data as (

    select *
    from fact_sales as s
    left join dim_holiday as h
    on s.order_date = h.date
)

select *
from source_data
