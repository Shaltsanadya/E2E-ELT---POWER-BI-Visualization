from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from prophet import Prophet
import psycopg2


default_args = {
    "owner": "yourname",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
} 
holday = pd.read_excel("/opt/airflow/dags/US HOLIDAY.xlsx")
sales =  pd.read_excel("/opt/airflow/dags/US SALES.xlsx")
df = pd.read_excel("/opt/airflow/dags/Sales_Data_Full.xlsx")
def extract_data():
    conn = psycopg2.connect(
        user="postgres",
        password="yourpassword",
        host="postgres",
        port="5432",
        database="postgres",

    )
    cur = conn.cursor()
    cur.execute("SET search_path TO public;")
    #dimension
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dim_holiday (
            date DATE NOT NULL,
            holiday VARCHAR(100)
        );
    """)
    cur.execute("TRUNCATE TABLE dim_holiday;")
    for _, row in holday.iterrows():
        date = row['Date'].strftime('%d-%m-%Y') if not pd.isna(row['Date']) else '01-01-1970'
        holiday = '' if pd.isna(row['Holiday']) else str(row['Holiday']).replace("'", "''")

        cur.execute(f"""
            INSERT INTO dim_holiday
            (date, holiday)
            VALUES (
                TO_DATE('{date}', 'DD-MM-YYYY'),
                '{holiday}'
            );
        """)
    #fact
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fact_sales(
            order_date DATE NOT NULL,
            subject_category VARCHAR(100),
            sales_amount NUMERIC,
            sales_quantity INT
        );
    """)
    cur.execute("TRUNCATE TABLE fact_sales;")
    for _, row in sales.iterrows():
        order_date = row['Order Date'].strftime('%d-%m-%Y') if not pd.isna(row['Order Date']) else '01-01-1970'
        subject_category = '' if pd.isna(row['Subject Category']) else str(row['Subject Category']).replace("'", "''")
        sales_amount = 0 if pd.isna(row['Sales Amount']) else row['Sales Amount']
        sales_quantity = 0 if pd.isna(row['Sales Quantity']) else row['Sales Quantity']

        cur.execute(f"""
            INSERT INTO fact_sales
            (order_date, subject_category, sales_amount, sales_quantity)
            VALUES (
                TO_DATE('{order_date}', 'DD-MM-YYYY'),
                '{subject_category}',
                {sales_amount},
                {sales_quantity}
            );
        """)
    #SALES
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales_data (
            id SERIAL PRIMARY KEY,
            order_date DATE NOT NULL,
            subject_category VARCHAR(100),
            sales_amount NUMERIC,
            sales_quantity INT,
            isholiday INT,
            holiday_name VARCHAR(100)
        );
    """)
    cur.execute("TRUNCATE TABLE sales_data;")
    for _, row in df.iterrows():
        order_date = row['Order Date'].strftime('%d-%m-%Y') if not pd.isna(row['Order Date']) else '01-01-1970'
        subject_category = '' if pd.isna(row['Subject Category']) else str(row['Subject Category']).replace("'", "''")
        holiday_name = '' if pd.isna(row['holiday_name']) else str(row['holiday_name']).replace("'", "''")
        sales_amount = 0 if pd.isna(row['Sales Amount']) else row['Sales Amount']
        sales_quantity = 0 if pd.isna(row['Sales Quantity']) else row['Sales Quantity']
        isholiday = 0 if pd.isna(row['isholiday']) else int(row['isholiday'])

        cur.execute(f"""
            INSERT INTO sales_data
            (order_date, subject_category, sales_amount, sales_quantity, isholiday, holiday_name)
            VALUES (
                TO_DATE('{order_date}', 'DD-MM-YYYY'),
                '{subject_category}',
                {sales_amount},
                {sales_quantity},
                {isholiday},
                '{holiday_name}'
            );
        """)
    conn.commit()
    cur.close()
    conn.close()

def forecast_data():
    conn = psycopg2.connect(
        user="postgres",
        password="yourpassword",
        host="postgres",
        port="5432",
        database="postgres"
    )
    cur = conn.cursor()

    cur.execute("SET search_path TO public;")
    cur.execute("SELECT order_date, sales_quantity, sales_amount FROM sales_data;")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    # ============================================
    # SARIMAX PREP
    # ============================================
    df = pd.DataFrame(rows, columns=['Order Date', 'Sales Quantity', 'Sales Amount'])
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    df[['Sales Quantity', 'Sales Amount']] = df[['Sales Quantity', 'Sales Amount']].apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    )

    df['category'] = 'HISTORY'

    weekly = (
        df[['Order Date', 'Sales Quantity', 'Sales Amount']]
        .set_index('Order Date')
        .resample('W')
        .sum()
    ).asfreq('W')

    weekly = weekly.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    y_qty = weekly['Sales Quantity']
    y_amt = weekly['Sales Amount']

    order = (1, 1, 0)
    seasonal_order = (0, 1, 1, 52)

    def sarimax_forecast(series, steps=52):
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False)
        fc = result.get_forecast(steps=steps)
        idx = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=steps, freq='W')
        return pd.DataFrame({'Forecast': fc.predicted_mean}, index=idx)

    forecast_qty = sarimax_forecast(y_qty)
    forecast_amt = sarimax_forecast(y_amt)

    forecast_qty['Forecast'] = forecast_qty['Forecast'].round(2)
    forecast_amt['Forecast'] = forecast_amt['Forecast'].round(2)

    sarimax = pd.merge(
        forecast_qty, forecast_amt,
        left_index=True, right_index=True
    ).reset_index().rename(columns={
        'index': 'Order Date',
        'Forecast_x': 'Sales Quantity',
        'Forecast_y': 'Sales Amount'
    })

    sarimax['category'] = 'SARIMAX'

    historical = df[['Order Date', 'Sales Quantity', 'Sales Amount', 'category']]
    sarimax = pd.concat([historical, sarimax], ignore_index=True)

    conn = psycopg2.connect(
        user="postgres",
        password="yourpassword",
        host="postgres",
        port="5432",
        database="postgres"
    )
    cur = conn.cursor()
    cur.execute("SET search_path TO public;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sarimax_fc (
            order_date DATE NOT NULL,
            sales_quantity NUMERIC,
            sales_amount NUMERIC,
            category VARCHAR(100)
        );
    """)
    cur.execute("TRUNCATE TABLE sarimax_fc;")

    insert_query = """
        INSERT INTO sarimax_fc (order_date, sales_quantity, sales_amount, category)
        VALUES (TO_DATE(%s, 'YYYY-MM-DD'), %s, %s, %s);
    """

    for _, row in sarimax.iterrows():
        cur.execute(insert_query, (
            row['Order Date'].strftime('%Y-%m-%d'),
            row['Sales Quantity'],
            row['Sales Amount'],
            row['category']
        ))

    conn.commit()
    cur.close()
    conn.close()

    # ============================================
    # PROPHET
    # ============================================
    df = pd.DataFrame(rows, columns=['Order Date', 'Sales Quantity', 'Sales Amount'])
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['category'] = 'HISTORY'

    # Quantity
    fc = df.set_index('Order Date')
    fc = fc.resample('W-MON')[['Sales Quantity']].sum().round(2)

    train_data = fc.reset_index()
    train_data.columns = ['ds', 'y']

    m = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='additive'
    )

    model = m.fit(train_data)
    future_data = model.make_future_dataframe(periods=52, freq='W-MON')
    predict = model.predict(future_data)

    predict1 = (
        predict.set_index('ds')
               .resample('W-MON')
               .sum()[['yhat']]
               .reset_index()
    )
    predict1 = predict1.iloc[len(train_data):].rename(columns={'ds': 'Order Date', 'yhat': 'Sales Quantity'})

    # Amount
    fc = df.set_index('Order Date')
    fc = fc.resample('W-MON')[['Sales Amount']].sum().round(2)

    train_data = fc.reset_index()
    train_data.columns = ['ds', 'y']

    m = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='additive'
    )

    model = m.fit(train_data)
    future_data = model.make_future_dataframe(periods=52, freq='W-MON')
    predict = model.predict(future_data)

    predict2 = (
        predict.set_index('ds')
               .resample('W-MON')
               .sum()[['yhat']]
               .reset_index()
    )
    predict2 = predict2.iloc[len(train_data):].rename(columns={'ds': 'Order Date', 'yhat': 'Sales Amount'})

    fc = fc.reset_index()
    predictfull = pd.merge(predict2, predict1, on='Order Date', how='inner')
    predictfull['category'] = 'PROPHET'
    predictall = pd.concat([df, predictfull], axis=0)


    conn = psycopg2.connect(
        user="postgres",
        password="yourpassword",
        host="postgres",
        port="5432",
        database="postgres"
    )
    cur = conn.cursor()

    cur.execute("SET search_path TO public;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prophet_fc (
            order_date DATE NOT NULL,
            sales_quantity NUMERIC,
            sales_amount NUMERIC,
            category VARCHAR(100)
        );
    """)
    cur.execute("TRUNCATE TABLE prophet_fc;")

    insert_query = """
        INSERT INTO prophet_fc (order_date, sales_quantity, sales_amount, category)
        VALUES (TO_DATE(%s, 'YYYY-MM-DD'), %s, %s, %s);
    """

    for _, row in predictall.iterrows():
        cur.execute(insert_query, (
            row['Order Date'].strftime('%Y-%m-%d'),
            row['Sales Quantity'],
            row['Sales Amount'],
            row['category']
        ))

    conn.commit()
    cur.close()
    conn.close()

with DAG(
    dag_id="etl_test",
    start_date=datetime(2026, 1, 6),
    end_date =datetime(2026, 2, 28),
    schedule=None,
    catchup=False,
    default_args=default_args,
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable= extract_data
    )
    transform_task =BashOperator(
    task_id="transform_data",
    bash_command= 'dbt run --select datamart_sales --profiles-dir /home/airflow/.dbt --project-dir /opt/airflow/dbt_project')

    forecast_task = PythonOperator(
        task_id="forecast_data",
        python_callable= forecast_data
    )   


    extract_task >> transform_task >> forecast_task
