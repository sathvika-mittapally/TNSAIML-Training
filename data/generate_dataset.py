"""
Dataset Generator for Air Quality Index (AQI) Prediction
Generates a synthetic environmental dataset.
"""

import os
import numpy as np
import pandas as pd

def calculate_sub_index(conc, breakpoints):
    """EPA piecewise linear sub‑index calculation."""
    if conc < 0:
        conc = 0
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= conc <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
    # Extrapolate above highest breakpoint, cap at 500
    c_lo, c_hi, i_lo, i_hi = breakpoints[-1]
    extrapolated = ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
    return min(500.0, max(0.0, extrapolated))

# EPA breakpoint tables (conc units: µg/m³ for PM, ppm for gases)
PM25_BP = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)]
PM10_BP = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)]
NO2_BP = [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)]
SO2_BP = [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)]
CO_BP = [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)]
O3_BP = [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200), (106, 200, 201, 300), (201, 400, 301, 500)]

def generate_data(samples=3000, seed=42):
    np.random.seed(seed)
    # Weather
    temp = np.clip(np.random.normal(26, 7.5, samples), 2, 48)
    hum = np.clip(np.random.normal(55, 18, samples), 15, 98)
    wind = np.clip(np.random.lognormal(2.2, 0.55, samples), 1, 45)
    solar = np.random.beta(2, 2, samples)
    stagnation = (1/(wind+1))*15 + (hum/100)*1.5
    # Pollutants
    pm25 = np.clip(np.random.exponential(35, samples)*(stagnation*0.4) + np.random.normal(5,3,samples), 3, 450)
    pm10 = np.clip(pm25*np.random.uniform(1.3,2.4,samples) + np.random.normal(8,4,samples), 8,600)
    no2 = np.clip(np.random.gamma(3,12,samples)*(1+(35-temp)*0.015), 2,250)
    so2 = np.clip(np.random.gamma(2.5,8,samples),1,180)
    co = np.clip(np.random.gamma(2,2.5,samples)*(stagnation*0.25),0.2,38)
    o3 = np.clip(temp*1.8 + solar*40 + no2*0.15 + np.random.normal(0,6,samples),5,280)
    aqi_vals, categories = [], []
    for i in range(samples):
        idx = max(
            calculate_sub_index(pm25[i], PM25_BP),
            calculate_sub_index(pm10[i], PM10_BP),
            calculate_sub_index(no2[i], NO2_BP),
            calculate_sub_index(so2[i], SO2_BP),
            calculate_sub_index(co[i], CO_BP),
            calculate_sub_index(o3[i], O3_BP),
        )
        idx = round(float(np.clip(idx + np.random.normal(0,1.5),5,500)),1)
        aqi_vals.append(idx)
        if idx <= 50: cat = "Good"
        elif idx <=100: cat = "Moderate"
        elif idx <=150: cat = "Unhealthy for Sensitive Groups"
        elif idx <=200: cat = "Unhealthy"
        elif idx <=300: cat = "Very Unhealthy"
        else: cat = "Hazardous"
        categories.append(cat)
    df = pd.DataFrame({
        "PM2.5": np.round(pm25,2),
        "PM10": np.round(pm10,2),
        "NO2": np.round(no2,2),
        "SO2": np.round(so2,2),
        "CO": np.round(co,2),
        "O3": np.round(o3,2),
        "Temperature": np.round(temp,1),
        "Humidity": np.round(hum,1),
        "Wind_Speed": np.round(wind,1),
        "AQI": aqi_vals,
        "AQI_Category": categories,
    })
    return df

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "air_quality_data.csv")
    print("Generating dataset …")
    data = generate_data(3500)
    data.to_csv(out, index=False)
    print(f"Saved to {out} ({len(data)} rows)")
