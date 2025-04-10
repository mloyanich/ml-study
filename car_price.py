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
        # Set the style for the plots to dark theme
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        self.features = []

        # Create a PDF file to save the plots
        output_pdf = "plots.pdf"
        self.pdf = PdfPages(output_pdf)
        print(f"Plots will be saved as {output_pdf}")

        df = pd.read_csv("data.csv")

        df.columns = df.columns.str.lower().str.replace(" ", "_")

        string_columns = list(df.dtypes[df.dtypes == "object"].index)
        for col in string_columns:
            df[col] = df[col].str.lower().str.replace(" ", "_")

        self.df = self.extract_features(df)

        print(f"len(df): {len(self.df)}")
        print(f"df.head(): {self.df.head()}")
        print(f"df.columns: {self.df.columns}")
        print(f"self.features: {self.features}")
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

        return w[0], w[1:]

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error**2).mean()
        return np.sqrt(mse)

    def feature_eng(self, df, column_name, prefix, n_features=3):
        feature_set = df[column_name].value_counts().head(n_features).index
        features = []
        for f in feature_set:
            feature = f"{prefix}_{f}"
            if self.features and feature not in self.features:
                print(f"skipping feature {feature}")
                continue
            df[feature] = (df[column_name] == f).astype(int)
            features.append(feature)
        return features

    def extract_features(self, df_input):
        df = df_input.copy()
        if self.features:
            features = self.features.copy()
        else:
            features = BASE_FEATURES.copy()

        df["age"] = 2025 - df.year
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
        return df

    def prepare_X(self, df):
        if self.features:
            features = self.features.copy()
        else:
            features = BASE_FEATURES.copy()

        df_num = df[features]
        df_num = df_num.fillna(0)

        print(df_num.std())
        X = df_num.values
        return X, features

    def dedup(self, array):
        # Remove duplicates using a loop
        unique_array = []
        for item in array:
            if item not in unique_array:
                unique_array.append(item)

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

        print(df_test.columns)
        X_test, features = self.prepare_X(df_test)
        print(features)
        y_pred = self.w0 + X_test.dot(self.w)
        return np.expm1(y_pred)

    def train_and_validate(self, r):
        w0, w = self.train_linear_reg(r)
        print(self.features)
        print(w)
        print("%5s, %.2f, %.2f, %.2f" % (r, w0, w[13], w[20]))
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
        print("RMSE: %s", self.rmse(self.y_val, y_pred_val))

    def test_reg(self, r_list=[0, 0.001, 0.01, 0.1, 1, 10]):
        for r in r_list:
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
        self.plot(title="Histplot of log pricel", xlabel="log(MSRP)", ylabel="Count")

    def prep_data(self):
        print("prepping the data")
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
        print("prep train data")
        self.X_train, features = self.prepare_X(df_train)
        self.features = features
        print("prep validation data")
        self.X_val, features = self.prepare_X(df_val)
        print("prep test data")
        self.X_test, features = self.prepare_X(df_test)
        print("all data prepped")

    def train_linear_regression(self, r):
        self.w0, self.w = self.train_linear_reg(r)

        y_pred = self.w0 + self.X_val.dot(self.w)
        print("validation:", self.rmse(self.y_val, y_pred))

        y_pred = self.w0 + self.X_test.dot(self.w)
        print("test:", self.rmse(self.y_test, y_pred))


def main():
    price_predictor = CarPrice()

    price_predictor.initial_histplot()

    price_predictor.test_reg()
    price_predictor.train_linear_regression(r=0.01)

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
    print(price_predictor.predict(ad))

    jeep_wrangler_2016 = {
        "city_mpg": 17,  # Typical city mileage for a 2016 Jeep Wrangler
        "driven_wheels": "four_wheel_drive",  # Jeep Wranglers are known for their 4WD capability
        "engine_cylinders": 6.0,  # Assuming a 6-cylinder engine
        "engine_fuel_type": "regular_unleaded",  # Common fuel type
        "engine_hp": 285.0,  # Horsepower for a typical 2016 Wrangler
        "highway_mpg": 21,  # Typical highway mileage
        "make": "jeep",  # Manufacturer
        "market_category": "suv,off_road",  # Market category
        "model": "wrangler",  # Model name
        "number_of_doors": 4.0,  # Typically a 2-door or 4-door option
        "popularity": 1500,  # Hypothetical popularity score
        "transmission_type": "manual",  # Assuming manual transmission
        "vehicle_size": "large",  # Vehicle size category
        "vehicle_style": "suv",  # Vehicle style
        "year": 2016,  # Model year
    }

    print(price_predictor.predict(jeep_wrangler_2016))


if __name__ == "__main__":
    main()
