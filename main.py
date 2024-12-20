import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Streamlit app
def main():
    st.title("ECU Map Extension Tool")
    st.markdown("""
    Paste your ECU map data, preview it, and generate an extended map up to a specified torque value.
    """)

    # Step 1: Input data
    st.header("1. Paste Your Map Data")
    st.markdown("Input two axes (RPM and Airflow) and the corresponding Torque values as a table.")
    input_data = st.text_area("Paste data in CSV format (columns: RPM, Airflow, Torque):")
    
    if input_data:
        try:
            # Parse the input data
            map_data = pd.read_csv(pd.compat.StringIO(input_data))
            
            if {'RPM', 'Airflow', 'Torque'}.issubset(map_data.columns):
                st.success("Data loaded successfully!")
                st.write("Preview of the data:")
                st.dataframe(map_data.head())
            else:
                st.error("Ensure your data has columns: 'RPM', 'Airflow', and 'Torque'.")
                return
        except Exception as e:
            st.error(f"Error reading data: {e}")
            return

        # Step 2: Validate and process data
        st.header("2. Validate and Extrapolate")
        max_torque = st.number_input(
            "Enter the maximum torque for extrapolation (Nm):", min_value=0, value=800, step=10
        )

        if st.button("Generate Extended Map"):
            # Extract the columns
            rpm = map_data['RPM'].values
            airflow = map_data['Airflow'].values
            torque = map_data['Torque'].values

            # Define new ranges for extrapolation
            rpm_new = np.linspace(rpm.min(), rpm.max() * 2, 100)  # Extend RPM range
            airflow_new = np.linspace(airflow.min(), airflow.max() * 2, 100)  # Extend airflow
            rpm_grid, airflow_grid = np.meshgrid(rpm_new, airflow_new)

            # Interpolate and extrapolate torque values
            torque_grid = griddata(
                (rpm, airflow),  # Known points
                torque,          # Known values
                (rpm_grid, airflow_grid),  # Grid points
                method='linear',  # Use linear interpolation
                fill_value=max_torque  # Extrapolate with max torque
            )

            # Create a DataFrame for the extended map
            extended_map = pd.DataFrame({
                'RPM': rpm_grid.flatten(),
                'Airflow': airflow_grid.flatten(),
                'Torque': torque_grid.flatten()
            })

            # Step 3: Display and download
            st.success("Extended map generated!")
            st.write("Preview of the extended map:")
            st.dataframe(extended_map.head(20))

            # Provide download link
            csv = extended_map.to_csv(index=False)
            st.download_button(
                label="Download Extended Map as CSV",
                data=csv,
                file_name='extended_map.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
