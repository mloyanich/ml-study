import typer
from car_price_predictor.car_price import CarPrice


app = typer.Typer()


@app.command()
def train_model():
    """Train the car price prediction model."""
    predictor = CarPrice()
    predictor.train_linear_regression(r=0.01)
    typer.echo("Model trained successfully.")


@app.command()
def predict_price():
    """Predict car price based on input features."""
    predictor = CarPrice()
    ad = {
        "city_mpg": 18,
        "driven_wheels": "all_wheel_drive",
        "engine_cylinders": 6.0,
        "engine_fuel_type": "regular_unleaded",
        "engine_hp": 268.0,
        "highway_mpg": 25,
        "make": "toyota",
        "market_category": "crossover,performance",
        "model": "venza",
        "number_of_doors": 4.0,
        "popularity": 2031,
        "transmission_type": "automatic",
        "vehicle_size": "midsize",
        "vehicle_style": "wagon",
        "year": 2013,
    }
    price = predictor.predict(ad)
    typer.echo(f"Predicted price: ${price[0]:,.2f}")


if __name__ == "__main__":
    app()
