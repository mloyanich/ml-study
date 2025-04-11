import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_FEATURES = [
    "engine_hp",
    "engine_cylinders",
    "highway_mpg",
    "city_mpg",
    "popularity",
]


class CarPrice:
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler("car_price.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)

        # Set the style for the plots to dark theme
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        self.features = []

        # Create a PDF file to save the plots
        output_pdf = "plots.pdf"
        self.pdf = PdfPages(output_pdf)
        self.logger.debug("Plots will be saved as %s", output_pdf)

        df = pd.read_csv("data/data.csv")
        self.logger.debug("Data loaded successfully from '%s'", "data/data.csv")

        df.columns = df.columns.str.lower().str.replace(" ", "_")
        self.logger.debug("Columns after normalization: %s", df.columns.tolist())

        string_columns = list(df.dtypes[df.dtypes == "object"].index)
        for col in string_columns:
            df[col] = df[col].str.lower().str.replace(" ", "_")
        self.logger.debug("String columns normalized: %s", string_columns)

        self.df = self.extract_features(df)

        self.logger.debug("Dataframe length: %d", len(self.df))
        self.logger.debug("Dataframe head: %s", self.df.head())
        self.logger.debug("Dataframe columns: %s", self.df.columns.tolist())
        self.logger.debug("Initial features: %s", self.features)
        self.prep_data()

    def plot(self, title=None, xlabel=None, ylabel=None, print_result=False):
        if title:
            plt.title(title, color="white")
        if xlabel:
            plt.xlabel(xlabel, color="white")
        if ylabel:
            plt.ylabel(ylabel, color="white")
        plt.tick_params(axis="x", colors="white")
        plt.tick_params(axis="y", colors="white")
        self.pdf.savefig()  # Save the current figure to the PDF
        if print_result:
            plt.show()
        plt.close()  # Close the plot to free up memory
        self.logger.debug("Plot saved with title: %s", title)

    def plt_fig(self):
        plt.figure(figsize=(10, 6))

    def train_linear_reg(self, r=0.0):
        X = self.X_train
        y = self.y_train
        # adding the dummy column
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        # normal equation formula
        XTX = X.T.dot(X)
        reg = r * np.eye(XTX.shape[0])
        XTX = XTX + reg
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        self.logger.debug("Linear regression trained with regularization: %f", r)
        return w[0], w[1:]

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error**2).mean()
        rmse_value = np.sqrt(mse)
        self.logger.debug("Computed RMSE: %f", rmse_value)
        return rmse_value

    def feature_eng(self, df, column_name, prefix, n_features=3):
        feature_set = df[column_name].value_counts().head(n_features).index
        features = []
        for f in feature_set:
            feature = f"{prefix}_{f}"
            if self.features and feature not in self.features:
                self.logger.debug("Skipping feature: %s", feature)
                continue
            df[feature] = (df[column_name] == f).astype(int)
            features.append(feature)
        self.logger.debug(
            "Features engineered for column '%s': %s", column_name, features
        )
        return features

    def extract_features(self, df_input):
        df = df_input.copy()
        if self.features:
            features = self.features.copy()
        else:
            features = BASE_FEATURES.copy()

        df["age"] = 2017 - df.year
        features.append("age")

        df["number_of_doors"] = df["number_of_doors"].fillna(0).astype(int)
        door_features = self.feature_eng(df, "number_of_doors", "num_doors", 3)
        features.extend(door_features)

        make_features = self.feature_eng(df, "make", "is_make", 10)
        features.extend(make_features)

        fuel_features = self.feature_eng(df, "engine_fuel_type", "is_type", 4)
        features.extend(fuel_features)

        transmission_features = self.feature_eng(
            df, "transmission_type", "is_transmission", 3
        )
        features.extend(transmission_features)
        features = self.dedup(features)
        if self.features:
            for feature in self.features:
                if feature not in df.columns:
                    df[feature] = 0
        if not self.features:
            self.features = features
        self.logger.debug("Extracted features: %s", self.features)
        return df

    def prepare_X(self, df):
        if self.features:
            features = self.features.copy()
        else:
            features = BASE_FEATURES.copy()

        df_num = df[features]
        df_num = df_num.fillna(0)

        self.logger.debug("Feature standard deviations: %s", df_num.std())
        X = df_num.values
        return X, features

    def dedup(self, array):
        # Remove duplicates using a loop
        unique_array = []
        for item in array:
            if item not in unique_array:
                unique_array.append(item)

        self.logger.debug("Deduplicated features: %s", unique_array)
        return unique_array

    def predict(self, car_listing):
        df_test = pd.DataFrame([car_listing])
        if self.features:
            for feature in self.features:
                if feature not in df_test.columns:
                    df_test[feature] = 0
        cols = []
        cols.extend(self.features)
        cols.extend(self.df_train.columns)
        for col in df_test.columns:
            if col not in cols:
                del df_test[col]

        self.logger.debug("Test dataframe columns: %s", df_test.columns)
        X_test, features = self.prepare_X(df_test)
        self.logger.debug("Features used for prediction: %s", features)
        y_pred = self.w0 + X_test.dot(self.w)
        self.logger.debug("Prediction made for input: %s", car_listing)
        return np.expm1(y_pred)

    def train_and_validate(self, r=0.0):
        w0, w = self.train_linear_reg(r)
        self.logger.debug("Trained weights: w0=%f, w=%s", w0, w)
        y_pred = w0 + self.X_train.dot(w)

        self.plt_fig()
        sns.histplot(y_pred, label="prediction")
        sns.histplot(self.y_train, label="target")
        plt.legend()
        self.plot(title="prediction vs actual")

        y_pred_val = w0 + self.X_val.dot(w)

        self.plt_fig()
        sns.histplot(y_pred_val, label="Predictions")
        sns.histplot(self.y_val, label="Validation")
        plt.legend()
        self.plot(title="validation dataset")
        self.logger.info("Validation RMSE: %f", self.rmse(self.y_val, y_pred_val))

    def test_reg(self):
        for r in [0, 0.001, 0.01, 0.1, 1, 10]:
            self.logger.info("Testing regularization parameter: %f", r)
            self.train_and_validate(r)

    def initial_histplot(self):
        # Histogram of all cars MSRP with 40 bins
        self.plt_fig()
        sns.histplot(data=self.df, x="msrp", bins=40)
        self.plot(title="Histplot of all cars MSRP", xlabel="MSRP", ylabel="Count")

        # Histogram of cars with MSRP < $100k without bins
        self.plt_fig()
        sns.histplot(data=self.df[self.df.msrp < 100000], x="msrp", kde=True)
        self.plot(
            title="Histplot of cars with MSRP < $100k", xlabel="MSRP", ylabel="Count"
        )

        log_price = np.log1p(self.df.msrp)
        self.plt_fig()
        sns.histplot(log_price)
        self.plot(title="Histplot of log price", xlabel="log(MSRP)", ylabel="Count")

    def prep_data(self):
        self.logger.debug("Preparing the data")
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)
        np.random.seed(2)
        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]
        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train : n_train + n_val].copy()
        df_test = df_shuffled.iloc[n_train + n_val :].copy()

        self.y_train = np.log1p(df_train.msrp.values)
        self.y_val = np.log1p(df_val.msrp.values)
        self.y_test = np.log1p(df_test.msrp.values)

        del df_train["msrp"]
        del df_val["msrp"]
        del df_test["msrp"]

        self.df_train = df_train
        self.logger.debug("Training data prepared")
        self.X_train, features = self.prepare_X(df_train)
        self.features = features
        self.logger.debug("Validation data prepared")
        self.X_val, features = self.prepare_X(df_val)
        self.logger.debug("Test data prepared")
        self.X_test, features = self.prepare_X(df_test)
        self.logger.debug("All data prepared")

    def train(self, r=0.0):
        w0, w = self.train_linear_reg(r)

        y_pred = self.w0 + self.X_val.dot(self.w)
        self.logger.debug("Validation RMSE: %f", self.rmse(self.y_val, y_pred))

        y_pred = self.w0 + self.X_test.dot(self.w)
        self.logger.debug("Test RMSE: %f", self.rmse(self.y_test, y_pred))
        return self.w0, self.w

    def validate(self):
        y_pred = self.w0 + self.X_val.dot(self.w)
        rmse_val = self.rmse(self.y_val, y_pred)
        self.logger.debug("Validation RMSE: %f", rmse_val)
        return rmse_val

    def test(self):
        y_pred = self.w0 + self.X_test.dot(self.w)
        rmse_test = self.rmse(self.y_test, y_pred)
        self.logger.debug("Test RMSE: %f", rmse_test)
        return rmse_test


def main():
    logging.info("Starting CarPrice Predictor")
    price_predictor = CarPrice()

    price_predictor.initial_histplot()

    price_predictor.test_reg()
    price_predictor.train(r=0.01)
    price_predictor.validate()
    price_predictor.test()

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
    predicted_price = price_predictor.predict(ad)[0]
    print(f"Predicted price for Toyota Vanza: {predicted_price:.2f}")

    jeep_wrangler_2016 = {
        "city_mpg": 17,
        "driven_wheels": "four_wheel_drive",
        "engine_cylinders": 6.0,
        "engine_fuel_type": "regular_unleaded",
        "engine_hp": 285.0,
        "highway_mpg": 21,
        "make": "jeep",
        "market_category": "suv,off_road",
        "model": "wrangler",
        "number_of_doors": 4.0,
        "popularity": 1500,
        "transmission_type": "manual",
        "vehicle_size": "large",
        "vehicle_style": "suv",
        "year": 2016,
    }
    predicted_price = price_predictor.predict(jeep_wrangler_2016)[0]
    print(f"Prediction for Jeep Wrangler 2016: {predicted_price:.2f}")
    price_predictor.pdf.close()


if __name__ == "__main__":
    main()
