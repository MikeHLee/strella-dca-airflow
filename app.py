import streamlit as st
import psycopg2 as pg
import polars as pl
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import datetime
import plotly.express as px

st.set_page_config(page_title="Room Similarity", 
				   page_icon = "./images/strella_logo.jpeg") 

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

    all_rooms = pl.DataFrame(
                            schema={
                                "room_id": pl.String,
                                "room_name": pl.String,
                                "variety": pl.String,
                                "customer": pl.String,
                                "scoring_start_date": pl.Datetime,
                                "scoring_end_date": pl.Datetime,
                                "scoring_start_date_override": pl.Datetime,
                                "scoring_end_date_override": pl.Datetime,
                                "opened_date": pl.Datetime 
                            }
                        )

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
    df_co2 = pl.DataFrame(
                results,
                orient="row",
                schema={"time": pl.Datetime("ns"),
                        "co2_ppm": pl.Float64,
                        "o2_ppm": pl.Float64,
                        "temp_c": pl.Float64,
                        "device_id": pl.String})

    # at this point any data marked as should_drop or not in the CAProd status will have been dropped
    # see the setup for the continuous aggregate measurements_time_weight_30_minute
    ts_cursor.execute(C2H4_QUERY, (room_id, score_start_date, score_end_date))

    results = ts_cursor.fetchall()

    # put the results into a polars data frame
    df_c2h4 = pl.DataFrame(
                results,
                orient="row",
                schema={"time": pl.Datetime("ns"),
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
        if len(selected_rooms) < 2:
            st.warning("You must select at least 2 rooms to compare!")
        else:
            selected_rooms_data = {}
            with st.spinner("Fetching sensor readings..."):
                for room_name in selected_rooms:
                    st.toast("Fetching data for " + room_name + " ...", icon="🍏")
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
                                                                        "time", every="30m"
                                                                        ).agg(
                                                                                pl.col("co2_ppm").mean().round(3),
                                                                                pl.col("o2_ppm").mean().round(3),
                                                                                pl.col("temp_c").mean().round(3),
                                                                                pl.col("c2h4_ppm").mean().round(3)
                                                                            ).with_columns(
                                                                                    co2_ppm_delta = pl.col("co2_ppm").diff().round(3),
                                                                                    o2_ppm_delta = pl.col("o2_ppm").diff().round(3),
                                                                                    temp_c_delta = pl.col("temp_c").diff().round(3),
                                                                                    c2h4_ppm_delta = pl.col("c2h4_ppm").diff().round(3)
                                                                                )
                    
                    st.toast("Obtained, filtered, and transformed data for " + room_name + " .", icon="🍏")

            #Raw values correlation (co2, c2h4, temp)
            ordered_room_names = list(selected_rooms_data.keys())

            co2_correlations = []
            c2h4_correlations = []
            o2_correlations = []
            temp_c_correlations = []

            delta_co2_correlations = []
            delta_c2h4_correlations = []
            delta_o2_correlations = []
            delta_temp_c_correlations = []

            co2_p_independence_30m_lag = []
            c2h4_p_independence_30m_lag = []
            o2_p_independence_30m_lag = []
            temp_c_p_independence_30m_lag = []

            delta_co2_p_independence_30m_lag = []
            delta_c2h4_p_independence_30m_lag = []
            delta_o2_p_independence_30m_lag = []
            delta_temp_c_p_independence_30m_lag = []

            co2_p_independence_1hr_lag = []
            c2h4_p_independence_1hr_lag = []
            o2_p_independence_1hr_lag = []
            temp_c_p_independence_1hr_lag = []

            delta_co2_p_independence_1hr_lag = []
            delta_c2h4_p_independence_1hr_lag = []
            delta_o2_p_independence_1hr_lag = []
            delta_temp_c_p_independence_1hr_lag = []
            
            #correlate its core 4 measurements with those of all other rooms to create 4 lists with len() == num keys)
            #append these lists to each of the lists above to create correlation matrices of each
            with st.spinner("Calculating independence metrics..."):
                for iteration_room_name in ordered_room_names:

                    #take sample frame
                    iteration_base_frame: pl.DataFrame = selected_rooms_data[iteration_room_name].filter(
                                                                                                    pl.col("co2_ppm").is_finite(),
                                                                                                    pl.col("o2_ppm").is_finite(),
                                                                                                    pl.col("temp_c").is_finite(),
                                                                                                    pl.col("c2h4_ppm").is_finite(),
                                                                                                    pl.col("co2_ppm_delta").is_finite(),
                                                                                                    pl.col("o2_ppm_delta").is_finite(),
                                                                                                    pl.col("temp_c_delta").is_finite(),
                                                                                                    pl.col("c2h4_ppm_delta").is_finite(),
                                                                                                )

                    this_iterations_co2_correlations = []
                    this_iterations_c2h4_correlations = []
                    this_iterations_o2_correlations = []
                    this_iterations_temp_c_correlations = []

                    this_iterations_delta_co2_correlations = []
                    this_iterations_delta_c2h4_correlations = []
                    this_iterations_delta_o2_correlations = []
                    this_iterations_delta_temp_c_correlations = []

                    this_iterations_co2_p_independence_30m_lag = []
                    this_iterations_c2h4_p_independence_30m_lag = []
                    this_iterations_o2_p_independence_30m_lag = []
                    this_iterations_temp_c_p_independence_30m_lag = []

                    this_iterations_delta_co2_p_independence_30m_lag = []
                    this_iterations_delta_c2h4_p_independence_30m_lag = []
                    this_iterations_delta_o2_p_independence_30m_lag = []
                    this_iterations_delta_temp_c_p_independence_30m_lag = []

                    this_iterations_co2_p_independence_1hr_lag = []
                    this_iterations_c2h4_p_independence_1hr_lag = []
                    this_iterations_o2_p_independence_1hr_lag = []
                    this_iterations_temp_c_p_independence_1hr_lag = []

                    this_iterations_delta_co2_p_independence_1hr_lag = []
                    this_iterations_delta_c2h4_p_independence_1hr_lag = []
                    this_iterations_delta_o2_p_independence_1hr_lag = []
                    this_iterations_delta_temp_c_p_independence_1hr_lag = []

                    for comparison_room_name in ordered_room_names:
                        #Join these into a comparison frame and drop nulls
                        iteration_comparison_frame: pl.DataFrame = selected_rooms_data[comparison_room_name].filter(
                                                                                                                pl.col("co2_ppm").is_finite(),
                                                                                                                pl.col("o2_ppm").is_finite(),
                                                                                                                pl.col("temp_c").is_finite(),
                                                                                                                pl.col("c2h4_ppm").is_finite(),
                                                                                                                pl.col("co2_ppm_delta").is_finite(),
                                                                                                                pl.col("o2_ppm_delta").is_finite(),
                                                                                                                pl.col("temp_c_delta").is_finite(),
                                                                                                                pl.col("c2h4_ppm_delta").is_finite(),
                                                                                                            )

                        joint_frame = iteration_base_frame.join(iteration_comparison_frame, on="time", how="inner").fill_nan(None).drop_nulls()

                        #Raw values correlation (co2, c2h4, temp)
                        co2_raw_correlation = np.corrcoef(joint_frame["co2_ppm"],joint_frame["co2_ppm_right"])[0][1]
                        this_iterations_co2_correlations.append(round(float(co2_raw_correlation),3))
                        c2h4_raw_correlation = np.corrcoef(joint_frame["o2_ppm"],joint_frame["o2_ppm_right"])[0][1] 
                        this_iterations_c2h4_correlations.append(round(float(c2h4_raw_correlation),3))
                        o2_raw_correlation = np.corrcoef(joint_frame["temp_c"],joint_frame["temp_c_right"])[0][1] 
                        this_iterations_o2_correlations.append(round(float(o2_raw_correlation),3))
                        temp_c_raw_correlation = np.corrcoef(joint_frame["c2h4_ppm"],joint_frame["c2h4_ppm_right"])[0][1]
                        this_iterations_temp_c_correlations.append(round(float(temp_c_raw_correlation),3))

                        #Delta values correlation (d_co2, d_c2h4, d_temp)
                        delta_co2_correlation = np.corrcoef(joint_frame["co2_ppm_delta"],joint_frame["co2_ppm_delta_right"])[0][1]
                        this_iterations_delta_co2_correlations.append(round(float(delta_co2_correlation),3))
                        delta_c2h4_correlation = np.corrcoef(joint_frame["o2_ppm_delta"],joint_frame["o2_ppm_delta_right"])[0][1]
                        this_iterations_delta_c2h4_correlations.append(round(float(delta_c2h4_correlation),3))
                        delta_o2_correlation = np.corrcoef(joint_frame["temp_c_delta"],joint_frame["temp_c_delta_right"])[0][1]
                        this_iterations_delta_o2_correlations.append(round(float(delta_o2_correlation),3))
                        delta_temp_c_correlation = np.corrcoef(joint_frame["c2h4_ppm_delta"],joint_frame["c2h4_ppm_delta_right"])[0][1]
                        this_iterations_delta_temp_c_correlations.append(round(float(delta_temp_c_correlation),3))

                        #Raw granger causality (co2, c2h4, temp) ie. likelihood of granger non-causality, independence
                        co2_ppm_independence_likelihood = grangercausalitytests(joint_frame[["co2_ppm", "co2_ppm_right"]], maxlag=2)
                        o2_ppm_independence_likelihood = grangercausalitytests(joint_frame[["o2_ppm", "o2_ppm_right"]], maxlag=2)
                        temp_c_independence_likelihood = grangercausalitytests(joint_frame[["temp_c", "temp_c_right"]], maxlag=2)
                        c2h4_ppm_independence_likelihood = grangercausalitytests(joint_frame[["c2h4_ppm", "c2h4_ppm_right"]], maxlag=2)

                        #Delta granger causality (d_co2, d_c2h4, d_temp)
                        delta_co2_ppm_independence_likelihood = grangercausalitytests(joint_frame[["co2_ppm_delta", "co2_ppm_delta_right"]], maxlag=2)
                        delta_o2_ppm_independence_likelihood = grangercausalitytests(joint_frame[["o2_ppm_delta", "o2_ppm_delta_right"]], maxlag=2)
                        delta_temp_c_independence_likelihood = grangercausalitytests(joint_frame[["temp_c_delta", "temp_c_delta_right"]], maxlag=2)
                        delta_c2h4_ppm_independence_likelihood = grangercausalitytests(joint_frame[["c2h4_ppm_delta", "c2h4_ppm_delta_right"]], maxlag=2)

                        for lag in co2_ppm_independence_likelihood:
                            test_results = co2_ppm_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_co2_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_co2_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                        
                        for lag in o2_ppm_independence_likelihood:
                            test_results = o2_ppm_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_o2_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_o2_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))

                        for lag in temp_c_independence_likelihood:
                            test_results = temp_c_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_temp_c_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_temp_c_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))

                        for lag in c2h4_ppm_independence_likelihood:
                            test_results = c2h4_ppm_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_c2h4_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_c2h4_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))

                        for lag in delta_co2_ppm_independence_likelihood:
                            test_results = delta_co2_ppm_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_delta_co2_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_delta_co2_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                        
                        for lag in delta_o2_ppm_independence_likelihood:
                            test_results = delta_o2_ppm_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_delta_o2_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_delta_o2_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))

                        for lag in delta_temp_c_independence_likelihood:
                            test_results = delta_temp_c_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_delta_temp_c_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_delta_temp_c_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))

                        for lag in delta_c2h4_ppm_independence_likelihood:
                            test_results = delta_c2h4_ppm_independence_likelihood[lag][0]
                            probabilities_of_independence = []
                            for test in test_results:
                                if comparison_room_name == iteration_room_name:
                                    p_independence = 0.0
                                else:
                                    p_independence = float(test_results[test][1])
                                probabilities_of_independence.append(p_independence)
                            if lag == 1:
                                this_iterations_delta_c2h4_p_independence_30m_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                            if lag == 2:
                                this_iterations_delta_c2h4_p_independence_1hr_lag.append(round(float(np.mean(probabilities_of_independence)),3))
                    
                    co2_correlations.append(this_iterations_co2_correlations)
                    c2h4_correlations.append(this_iterations_c2h4_correlations)
                    o2_correlations.append(this_iterations_o2_correlations)
                    temp_c_correlations.append(this_iterations_temp_c_correlations)
                    delta_co2_correlations.append(this_iterations_delta_co2_correlations)
                    delta_c2h4_correlations.append(this_iterations_delta_c2h4_correlations)
                    delta_o2_correlations.append(this_iterations_delta_o2_correlations)
                    delta_temp_c_correlations.append(this_iterations_delta_temp_c_correlations)
                    co2_p_independence_30m_lag.append(this_iterations_co2_p_independence_30m_lag)
                    co2_p_independence_1hr_lag.append(this_iterations_co2_p_independence_1hr_lag)
                    o2_p_independence_30m_lag.append(this_iterations_o2_p_independence_30m_lag)
                    o2_p_independence_1hr_lag.append(this_iterations_o2_p_independence_1hr_lag)
                    temp_c_p_independence_30m_lag.append(this_iterations_temp_c_p_independence_30m_lag)
                    temp_c_p_independence_1hr_lag.append(this_iterations_temp_c_p_independence_1hr_lag)
                    c2h4_p_independence_30m_lag.append(this_iterations_c2h4_p_independence_30m_lag)
                    c2h4_p_independence_1hr_lag.append(this_iterations_c2h4_p_independence_1hr_lag)
                    delta_co2_p_independence_30m_lag.append(this_iterations_delta_co2_p_independence_30m_lag)
                    delta_co2_p_independence_1hr_lag.append(this_iterations_delta_co2_p_independence_1hr_lag)
                    delta_o2_p_independence_30m_lag.append(this_iterations_delta_o2_p_independence_30m_lag)
                    delta_o2_p_independence_1hr_lag.append(this_iterations_delta_o2_p_independence_1hr_lag)
                    delta_temp_c_p_independence_30m_lag.append(this_iterations_delta_temp_c_p_independence_30m_lag)
                    delta_temp_c_p_independence_1hr_lag.append(this_iterations_delta_temp_c_p_independence_1hr_lag)
                    delta_c2h4_p_independence_30m_lag.append(this_iterations_delta_c2h4_p_independence_30m_lag)
                    delta_c2h4_p_independence_1hr_lag.append(this_iterations_delta_c2h4_p_independence_1hr_lag)
            
                #raw corr
                print(co2_correlations, "\n")
                print(c2h4_correlations, "\n")
                print(o2_correlations, "\n")
                print(temp_c_correlations, "\n")

                #delta corr
                print(delta_co2_correlations, "\n")
                print(delta_c2h4_correlations, "\n")
                print(delta_o2_correlations, "\n")
                print(delta_temp_c_correlations, "\n")

                #30m raw independence
                print(co2_p_independence_30m_lag, "\n")
                print(o2_p_independence_30m_lag, "\n")
                print(temp_c_p_independence_30m_lag, "\n")
                print(c2h4_p_independence_30m_lag, "\n")
                
                #1hr raw independence
                print(co2_p_independence_1hr_lag, "\n")
                print(o2_p_independence_1hr_lag, "\n")
                print(temp_c_p_independence_1hr_lag, "\n")
                print(c2h4_p_independence_1hr_lag, "\n")
                
                #30m delta independence
                print(delta_co2_p_independence_30m_lag, "\n")
                print(delta_o2_p_independence_30m_lag, "\n")
                print(delta_temp_c_p_independence_30m_lag, "\n")
                print(delta_c2h4_p_independence_30m_lag, "\n")
                
                #1hr raw independence
                print(delta_co2_p_independence_1hr_lag, "\n")
                print(delta_o2_p_independence_1hr_lag, "\n")
                print(delta_temp_c_p_independence_1hr_lag, "\n")
                print(delta_c2h4_p_independence_1hr_lag, "\n")

            #Likely groups (collections of rooms likely to be sharing air / scrubbing / ventilation)
            st.subheader("Raw Measurement Correlations")
            st.info("These charts show the correlations between sensor readings in all pairs of rooms at all 30m intervals in the measurement history.\
                    A value of 1 indicates a perfect correlation between sampled measurements, while a value of -1 indicates an inverse correlation between the samples")
            
            
            fig = px.imshow(co2_correlations,
                            labels=dict(x="Co2 Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(o2_correlations,
                            labels=dict(x="O2 Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            
            fig = px.imshow(c2h4_correlations,
                            labels=dict(x="C2H4 Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(temp_c_correlations,
                            labels=dict(x="Temp Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            st.subheader("Measurement Change Correlations")
            st.info("These charts show the correlations between the change in sensor values for all selected pairs of rooms at all 30m intervals in the measurement history.\
                    A value of 1 indicated perfect correlation (the measurements move perfectly together), while a value of -1 indicates inverse correlation (the measurements move in exactly opposite directions)")


            fig = px.imshow(delta_co2_correlations,
                            labels=dict(x="Delta Co2 Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(delta_o2_correlations,
                            labels=dict(x="Delta O2 Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)


            fig = px.imshow(delta_c2h4_correlations,
                            labels=dict(x="Delta C2H4 Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(delta_temp_c_correlations,
                            labels=dict(x="Delta Temp Correlations", y="", color="Relationship Strength"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            st.subheader("Raw Measurement Granger Causality (30m time lag)")
            st.info("These charts provide the likelihood that the raw measurement values in each pair of rooms is independent within the given time range. \
                    Independence in this context means that the values of one do not contain any predictive information about the values of the other within the given time lag.\
                    Values closer to 0 suggest that there is almost no chance that the rooms are not influencing each-other or otherwise related due to some latent factor")
            
            fig = px.imshow(co2_p_independence_30m_lag,
                            labels=dict(x="CO2 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(o2_p_independence_30m_lag,
                            labels=dict(x="O2 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)


            fig = px.imshow(c2h4_p_independence_30m_lag,
                            labels=dict(x="C2H4 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(temp_c_p_independence_30m_lag,
                            labels=dict(x="Temp 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            st.subheader("Raw Measurement Granger Causality (60m time lag")
            st.info("These charts provide the likelihood that the raw measurement values in each pair of rooms is independent within the given time range. \
                    Independence in this context means that the values of one do not contain any predictive information about the values of the other within the given time lag.\
                    Values closer to 0 suggest that there is almost no chance that the rooms are not influencing each-other or somehow related due to a hidden factor")

            fig = px.imshow(co2_p_independence_1hr_lag,
                            labels=dict(x="CO2 60m,", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(o2_p_independence_1hr_lag,
                            labels=dict(x="O2 60m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)


            fig = px.imshow(c2h4_p_independence_1hr_lag,
                            labels=dict(x="C2H4 60m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(temp_c_p_independence_1hr_lag,
                            labels=dict(x="Temp 60m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            st.subheader("Measurement Change Granger Causality (30m time lag)")
            st.info("These charts provide the likelihood that the changes in measurement values for each pair of rooms are independent from one another within the given time range. \
                    Independence in this context means that the values of one do not contain any predictive information about the values of the other within the given time lag.\
                    Values closer to 0 suggest that there is almost no chance that the rooms are not influencing each-other or somehow related due to some hidden factor")


            fig = px.imshow(delta_co2_p_independence_30m_lag,
                            labels=dict(x="Delta CO2 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(delta_o2_p_independence_30m_lag,
                            labels=dict(x="Delta O2 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            
            fig = px.imshow(delta_c2h4_p_independence_30m_lag,
                            labels=dict(x="Delta C2H4 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(delta_temp_c_p_independence_30m_lag,
                            labels=dict(x="Delta Temp 30m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            st.subheader("Measurement Change Granger Causality (60m time lag)")
            st.info("These charts provide the likelihood that the changes in measurement values for each pair of rooms are independent from one another within the given time range. \
                    Independence in this context means that the values of one do not contain any predictive information about the values of the other within the given time lag.\
                    Values closer to 0 suggest that there is almost no chance that the rooms are not influencing each-other or somehow related due to some hidden factor")


            fig = px.imshow(delta_co2_p_independence_1hr_lag,
                            labels=dict(x="Delta CO2 60m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(delta_o2_p_independence_1hr_lag,
                            labels=dict(x="Delta O2 60m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)


            fig = px.imshow(delta_c2h4_p_independence_1hr_lag,
                            labels=dict(x="Delta C2H4 60m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

            fig = px.imshow(delta_temp_c_p_independence_1hr_lag,
                            labels=dict(x="Delta Temp 60m", y="", color="Likelihood of Independence"),
                            x=["Room " + str(name) for name in ordered_room_names],
                            y=["Room " + str(name) for name in ordered_room_names],
                            text_auto=True
                    )
            fig.update_xaxes(side="top")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

if "authorized" not in st.session_state:
    st.session_state["authorized"] = False
if st.session_state["authorized"]:
    run()
else:
    st.subheader("Security Question")
    answer = st.text_input(label="Approximately many 🍎s does the Strella Maturity Scoring Algorithm supervise?")
    if answer == st.secrets.authorization.SECURITY_QUESTION_ANSWER:
        st.success("You got it!")
        st.session_state["authorized"] = True
        st.rerun()