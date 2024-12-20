import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    st.title("ECU Map Extension Tool with Linear Regression")
    st.markdown("""
    Paste your ECU map data (X-axis, Y-axis, and Torque matrix), validate it, 
    and extend the map dynamically using linear regression.
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

    # Step 4: Specify maximum torque and factors for extrapolation
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

            # Perform linear regression on the last few rows of torque matrix
            extended_torque_matrix = []
            for i in range(len(torque_matrix)):
                last_x = x_axis[-5:]  # Use the last 5 points for regression
                last_torque = torque_matrix[i, -5:]

                # Fit linear regression model
                reg = LinearRegression()
                reg.fit(last_x.reshape(-1, 1), last_torque)

                # Predict for the new X-axis
                new_torque = reg.predict(x_axis_new.reshape(-1, 1))
                extended_torque_matrix.append(new_torque)

            # Clip values at max torque
            extended_torque_matrix = np.clip(np.array(extended_torque_matrix), 0, max_torque)

            # Create DataFrame for the extended map
            extended_map = pd.DataFrame(
                extended_torque_matrix, index=y_axis_new, columns=x_axis_new
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
