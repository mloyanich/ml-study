import streamlit as st
from car_price_predictor.car_price import CarPrice


def main():
    st.title("Car Price Predictor")

    # Input fields for car features
    engine_hp = st.number_input("Engine HP", min_value=0.0, value=150.0)
    engine_cylinders = st.number_input("Engine Cylinders", min_value=1.0, value=4.0)
    highway_mpg = st.number_input("Highway MPG", min_value=0, value=30)
    city_mpg = st.number_input("City MPG", min_value=0, value=25)
    popularity = st.number_input("Popularity", min_value=0, value=1000)
    year = st.number_input("Year", min_value=1900, max_value=2025, value=2020)
    number_of_doors = st.number_input(
        "Number of Doors", min_value=2, max_value=5, value=4
    )
    make = st.text_input("Make", value="toyota")
    model = st.text_input("Model", value="camry")
    engine_fuel_type = st.text_input("Engine Fuel Type", value="regular_unleaded")
    transmission_type = st.text_input("Transmission Type", value="automatic")
    driven_wheels = st.text_input("Driven Wheels", value="front_wheel_drive")
    market_category = st.text_input("Market Category", value="sedan")
    vehicle_size = st.text_input("Vehicle Size", value="midsize")
    vehicle_style = st.text_input("Vehicle Style", value="sedan")

    if st.button("Predict Price"):
        car_features = {
            "engine_hp": engine_hp,
            "engine_cylinders": engine_cylinders,
            "highway_mpg": highway_mpg,
            "city_mpg": city_mpg,
            "popularity": popularity,
            "year": year,
            "number_of_doors": number_of_doors,
            "make": make.lower(),
            "model": model.lower(),
            "engine_fuel_type": engine_fuel_type.lower(),
            "transmission_type": transmission_type.lower(),
            "driven_wheels": driven_wheels.lower(),
            "market_category": market_category.lower(),
            "vehicle_size": vehicle_size.lower(),
            "vehicle_style": vehicle_style.lower(),
        }
        predictor = CarPrice()
        price = predictor.predict(car_features)
        st.success(f"Predicted price: ${price[0]:,.2f}")


if __name__ == "__main__":
    main()
