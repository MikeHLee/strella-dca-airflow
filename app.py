import streamlit as st
import psycopg2 as pg
import polars as pl

from typing import Optional, List

def create_connection_string(user, password, host, port, database):
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require&options=-c%20statement_timeout%3D60s"

op_db = create_connection_string(
            st.secrets.database.OP_USER, 
            st.secrets.database.OP_PASSWORD, 
            st.secrets.database.OP_HOST, 
            st.secrets.database.OP_PORT, 
            st.secrets.database.OP_DATABASE
        )

ts_db = create_connection_string(
            st.secrets.database.TS_USER, 
            st.secrets.database.TS_PASSWORD, 
            st.secrets.database.TS_HOST, 
            st.secrets.database.TS_PORT, 
            st.secrets.database.TS_DATABASE
        )

ALL_ROOMS_QUERY = '''
    SELECT "Room".id, "Room".name, "Room"."isOrganic", "Variety".name, "Customer".name, "Room"."scoringStartDate", "Room"."scoringEndDate", "Room"."scoringStartDateOverride", "Room"."scoringEndDateOverride", "Room"."openedDate"
    FROM "Variety" 
    RIGHT JOIN "RoomVariety" ON "RoomVariety"."varietyId" = "Variety".id 
    RIGHT JOIN "Room" ON "RoomVariety"."roomId" = "Room".id 
    LEFT JOIN "CustomerSite" ON "CustomerSite"."id" = "Room"."customerSiteId" 
    LEFT JOIN "Customer" ON "Customer".id = "CustomerSite"."customerId"
'''

CO2_QUERY = '''
    SELECT dt, avg_co2_ppm as co2_ppm, avg_o2_ppm as o2_ppm, avg_temp_c as temp_c, device_id
    FROM
    measurements_interpolated_10_minute
    WHERE room_id = %s
    AND dt > %s
    AND dt < %s
    ORDER BY dt ASC;
'''

C2H4_QUERY = '''
    SELECT dt, avg_c2h4_ppm as c2h4_ppm, device_id
    FROM
    measurements_interpolated_30_minute
    WHERE room_id = %s
    AND dt > %s
    AND dt < %s
    ORDER BY dt ASC;
'''

def fetch_customer_rooms() -> pl.DataFrame:
    op_conn = pg.connect(op_db)
    op_conn.set_session(autocommit=True, readonly=True)
    op_cursor = op_conn.cursor()

    all_rooms: Optional[pl.DataFrame] = None

    with st.spinner("Loading customers and rooms..."):
        op_cursor.execute(ALL_ROOMS_QUERY)
        all_room_data = op_cursor.fetchall()

        for entry in all_room_data:
            room = pl.DataFrame(
                            data={
                                "room_id": entry[0],
                                "room_name": entry[1],
                                "variety": ("Organic " if entry[2] else "") + (entry[3] if entry[3] else "Others"),
                                "customer": entry[4],
                                "scoring_start_date": entry[5],
                                "scoring_end_date": entry[6],
                                "scoring_start_date_override": entry[7],
                                "scoring_end_date_override": entry[8],
                                "opened_date": entry[9] 
                            }
                        )
            
            if all_rooms is None:
                all_rooms = room
            else:
                all_rooms = pl.concat([all_rooms,room])
    
    return all_rooms

def get_room_data(room_id: str, score_start_date: str, score_end_date: str):
    """
    CO2_QUERY: A SQL query string that selects the date-time, average CO2 ppm, average O2 ppm, average temperature in Celsius, and device ID from the 'measurements_interpolated_10_minute' table. The query filters the data based on room ID and a date-time range, and orders the results in ascending order by date-time.

    C2H4_QUERY: A SQL query string that selects the date-time, average C2H4 ppm, and device ID from the 'measurements_interpolated_30_minute' table. The query filters the data based on room ID and a date-time range, and orders the results in ascending order by date-time.

    get_room_data: A function that retrieves and processes room data from a database.

    Args:
        ts_cursor: A database cursor object used to execute SQL commands.
        room_id (str): The ID of the room for which data is to be retrieved.
        score_start_date (str): The start date for the data retrieval in the format "YYYY-MM-DD HH:MM:SS".
        score_end_date (str): The end date for the data retrieval in the format "YYYY-MM-DD HH:MM:SS". If None, the current date and time is used.

    Returns:
        df: A Polars DataFrame containing the retrieved data. The DataFrame includes columns for time, CO2 ppm, O2 ppm, temperature in Celsius, C2H4 ppm, and device ID. The data is joined on both time and device_id.

    Note:
        Any data marked as 'should_drop' or not in the 'CAProd' status will have been dropped before this function is called. See the setup for the continuous aggregate 'measurements_time_weight_10_minute' and 'measurements_time_weight_30_minute' for more details.
    """

    ts_conn = pg.connect(ts_db)
    ts_conn.set_session(autocommit=True, readonly=True)
    ts_cursor = ts_conn.cursor()

    if score_end_date is None:
        score_end_date = datetime.datetime.now(
            datetime.UTC).strftime("%Y-%m-%d 00:00:00")

    # at this point any data marked as should_drop or not in the CAProd status will have been dropped
    # see the setup for the continuous aggregate measurements_time_weight_10_minute
    ts_cursor.execute(CO2_QUERY, (room_id, score_start_date, score_end_date))

    results = ts_cursor.fetchall()

    # put the results into a polars data frame
    df_co2 = pl.DataFrame(results, schema={"time": pl.Datetime("ns"),
                                           "co2_ppm": pl.Float64,
                                           "o2_ppm": pl.Float64,
                                           "temp_c": pl.Float64,
                                           "device_id": pl.String})

    # at this point any data marked as should_drop or not in the CAProd status will have been dropped
    # see the setup for the continuous aggregate measurements_time_weight_30_minute
    ts_cursor.execute(C2H4_QUERY, (room_id, score_start_date, score_end_date))

    results = ts_cursor.fetchall()

    # put the results into a polars data frame
    df_c2h4 = pl.DataFrame(results, schema={"time": pl.Datetime("ns"),
                                            "c2h4_ppm": pl.Float64,
                                            "device_id": pl.String})

    # left join the two data frames (on both time and device_id)
    df = df_co2.join(df_c2h4, on=['time', 'device_id'], how='left')

    return df

def run():
    st.title("🍎 Room Shared Air Analysis Tool")

    if "all_rooms" not in st.session_state:
        st.session_state["all_rooms"] = fetch_customer_rooms()

    all_rooms: pl.DataFrame = st.session_state.all_rooms

    st.success("Customers and rooms loaded.")

    selected_customer = st.selectbox(
        label = "Select a customer.",
        options = all_rooms["customer"].unique(maintain_order=True)
    )

    selected_customers_rooms = all_rooms.filter(pl.col("customer") == selected_customer)

    selected_rooms = st.multiselect(label = "Select the rooms you'd like to compare.", options = selected_customers_rooms["room_name"].unique(maintain_order=True))

    if st.button(label = "**Compare**"):
        selected_rooms_data = {}
        with st.spinner("Fetching sensor readings..."):
            for room_name in selected_rooms:
                room = selected_customers_rooms.filter((pl.col("room_name") == room_name) & (pl.col("customer") == selected_customer))
                room_id = room["room_id"].first()
                scoring_start_date = room["scoring_start_date"].first()
                scoring_end_date = room["scoring_end_date"].first()
                scoring_start_date_override = room["scoring_start_date_override"].first()
                scoring_end_date_override = room["scoring_end_date_override"].first()
                opened_date = room["opened_date"].first()

                scoring_start_date = (
                    scoring_start_date_override
                    if scoring_start_date_override
                    else scoring_start_date
                )

                if scoring_end_date_override:
                    scoring_end_date = scoring_end_date_override
                elif opened_date:
                    scoring_end_date = opened_date
                else:
                    scoring_end_date = scoring_end_date

                this_rooms_data = get_room_data(
                    room_id=room_id,
                    score_start_date=scoring_start_date,
                    score_end_date=scoring_end_date
                )

                selected_rooms_data[room_name] = this_rooms_data.group_by_dynamic(
                                                                    "time", every="1hr"
                                                                ).agg(
                                                                    pl.col("co2_ppm"),
                                                                    pl.col("o2_ppm"),
                                                                    pl.col("temp_c"),
                                                                    pl.col("c2h4_ppm")
                                                                )

        #Raw values correlation (co2, c2h4, temp)
        ordered_room_names = selected_rooms_data.keys()
        co2_correlations = []
        c2h4_correlations = []
        o2_correlations = []
        temp_c_correlations = []
        for room_name in selected_rooms_data:
            #correlate its core 4 measurements with those of all other rooms to create 4 lists with len() == num keys)
            #append these lists to each of the lists above to create correlation matrices of each

            #take sample frame

            for room_name in selected_rooms_data:
                #Join these into a comparison frame and drop nulls
                
                #Raw values correlation (co2, c2h4, temp)
                ...

                #Delta values correlation (d_co2, d_c2h4, d_temp)
                ...   

                #Raw granger causality (co2, c2h4, temp) ie. likelihood of granger non-causality, independence
                ...

                #Delta granger causality (d_co2, d_c2h4, d_temp)
                ...

        #Likely groups (collections of rooms likely to be sharing air / scrubbing / ventilation)

run()