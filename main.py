import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import CubicSpline

def main():
    st.title("ECU Map Extension Tool")
    st.markdown("""
    Extend your ECU map dynamically by pasting axes and torque data. The app interpolates and extrapolates the map smoothly.
    """)

    # Step 1: Input X-axis (RPM)
    st.header("Step 1: Paste X-axis (RPM) Values")
    x_axis_input = st.text_area("Paste the X-axis (RPM) values, separated by spaces:")

    if x_axis_input:
        try:
            x_axis = np.array([float(val) for val in x_axis_input.split()])
            st.success("X-axis (RPM) values loaded successfully!")
            st.write("X-axis:", x_axis)
        except Exception as e:
            st.error(f"Error parsing X-axis values: {e}")
            return

    # Step 2: Input Y-axis (Airflow)
    st.header("Step 2: Paste Y-axis (Airflow) Values")
    y_axis_input = st.text_area("Paste the Y-axis (Airflow) values, separated by spaces:")

    if y_axis_input:
        try:
            y_axis = np.array([float(val) for val in y_axis_input.split()])
            st.success("Y-axis (Airflow) values loaded successfully!")
            st.write("Y-axis:", y_axis)
        except Exception as e:
            st.error(f"Error parsing Y-axis values: {e}")
            return

    # Step 3: Input Torque Values (Matrix)
    st.header("Step 3: Paste Torque Values (Matrix)")
    torque_matrix_input = st.text_area(
        "Paste the Torque values as rows, separated by tabs or spaces. Each row represents a Y-axis value."
    )

    if torque_matrix_input:
        try:
            torque_matrix = np.array([
                [float(val) for val in row.split()]
                for row in torque_matrix_input.strip().split("\n")
            ])

            if torque_matrix.shape == (len(y_axis), len(x_axis)):
                st.success("Torque matrix loaded successfully!")
                st.write("Torque Matrix (Preview):")
                st.dataframe(pd.DataFrame(torque_matrix, index=y_axis, columns=x_axis))
            else:
                st.error(
                    f"Matrix dimensions do not match! Expected ({len(y_axis)} rows, {len(x_axis)} columns)."
                )
                return
        except Exception as e:
            st.error(f"Error parsing Torque matrix: {e}")
            return

    # Step 4: Specify maximum torque and intervals for extrapolation
    if x_axis_input and y_axis_input and torque_matrix_input:
        st.header("Step 4: Specify Extrapolation Settings")
        max_torque = st.number_input(
            "Enter the maximum torque value for extrapolation (Nm):", min_value=0, value=800, step=10
        )
        rpm_extension_factor = st.slider(
            "Extend RPM axis by this factor:", min_value=1.0, max_value=2.0, value=1.5, step=0.1
        )
        airflow_extension_factor = st.slider(
            "Extend Airflow axis by this factor:", min_value=1.0, max_value=2.0, value=1.5, step=0.1
        )

        if st.button("Generate Extended Map"):
            # Extend axes
            x_axis_new = np.linspace(x_axis.min(), x_axis.max() * rpm_extension_factor, 100)
            y_axis_new = np.linspace(y_axis.min(), y_axis.max() * airflow_extension_factor, 100)
            x_grid, y_grid = np.meshgrid(x_axis_new, y_axis_new)

            # Interpolate torque matrix
            points = np.array([[x, y] for y in y_axis for x in x_axis])
            torque_flat = torque_matrix.flatten()
            torque_grid = griddata(
                points, torque_flat, (x_grid, y_grid), method='cubic', fill_value=np.nan
            )

            # Fill NaN values using smooth extrapolation
            for i in range(torque_grid.shape[0]):
                nan_indices = np.isnan(torque_grid[i])
                valid_indices = ~np.isnan(torque_grid[i])
                if np.any(valid_indices):
                    cs = CubicSpline(x_axis_new[valid_indices], torque_grid[i][valid_indices])
                    torque_grid[i][nan_indices] = cs(x_axis_new[nan_indices])

            # Clip values at max torque
            torque_grid = np.clip(torque_grid, 0, max_torque)

            # Create DataFrame for the extended map
            extended_map = pd.DataFrame(
                torque_grid, index=y_axis_new, columns=x_axis_new
            )

            # Display the extended map
            st.success("Extended map generated!")
            st.write("Preview of the extended map:")
            st.dataframe(extended_map)

            # Provide download link
            csv = extended_map.to_csv(index=True, header=True)
            st.download_button(
                label="Download Extended Map as CSV",
                data=csv,
                file_name='extended_map.csv',
                mime='text/csv'
            )


if __name__ == "__main__":
    main()
