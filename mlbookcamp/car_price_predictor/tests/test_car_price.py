import unittest
import pandas as pd
from car_price_predictor.car_price import CarPrice


class TestCarPricePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_data = pd.DataFrame(
            [
                {
                    "make": "toyota",
                    "year": 2015,
                    "engine_hp": 200,
                    "engine_cylinders": 4,
                    "highway_mpg": 30,
                    "city_mpg": 25,
                    "popularity": 1000,
                    "transmission_type": "automatic",
                    "engine_fuel_type": "regular_unleaded",
                    "number_of_doors": 4,
                    "msrp": 25000,
                }
            ]
            * 50
        )

    def test_feature_extraction(self):
        model = CarPrice()
        self.assertIn("age", model.features)
        for f in model.features:
            self.assertTrue(
                f.startswith("is_") or f.startswith("num_") or f in model.df.columns
            )

    def test_training(self):
        model = CarPrice()
        model.train()
        self.assertTrue(hasattr(model, "w0"))
        self.assertTrue(hasattr(model, "w"))
        self.assertIsInstance(model.w0, float)
        self.assertEqual(model.w.shape[0], model.X_train.shape[1])

    def test_prediction(self):
        model = CarPrice()
        model.train()
        sample_car = self.sample_data.iloc[0].drop("msrp").to_dict()
        price = model.predict(sample_car)
        self.assertIsInstance(price[0], float)
        self.assertGreater(price[0], 0)

    def test_rmse(self):
        model = CarPrice()
        model.train()
        rmse_val = model.validate()
        rmse_test = model.test()
        self.assertGreaterEqual(rmse_val, 0)
        self.assertGreaterEqual(rmse_test, 0)


if __name__ == "__main__":
    unittest.main()
