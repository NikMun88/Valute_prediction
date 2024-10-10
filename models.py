import pandas as pd
import numpy as np
from keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class AutoRegressionModel:
    def __init__(self, data, train_size=0.9, max_iters=50, mse_threshold=1e-4, no_improvement_threshold=50):
        """
        Инициализация класса с опциональными параметрами.
        data: Временной ряд (вектор-столбец)
        train_size: Доля данных для тренировки (по умолчанию 0.8)
        max_iters: Максимальное количество итераций для поиска лагов (по умолчанию 50)
        mse_threshold: Порог для минимального значения MSE (по умолчанию 1e-4)
        no_improvement_threshold: Число итераций без улучшений MSE перед остановкой (по умолчанию 5)
        """
        self.data = data
        self.train_size = train_size
        self.max_iters = max_iters
        self.mse_threshold = mse_threshold
        self.no_improvement_threshold = no_improvement_threshold

        # Внутренние параметры
        self.train = None
        self.test = None
        self.best_lag = None
        self.best_model = None

    def split_data(self):
        """Разделение данных на тренировочные и тестовые выборки."""
        train_size = int(len(self.data) * self.train_size)
        self.train, self.test = self.data[:train_size], self.data[train_size:]

    def find_best_lag(self):
        """Поиск оптимального числа лагов с минимальным значением MSE."""
        no_improvement_counter = 0
        best_mse = float('inf')
        lag = 1
        iteration = 0

        while iteration < self.max_iters and no_improvement_counter < self.no_improvement_threshold:
            # Строим модель с текущим числом лагов
            model = AutoReg(self.train, lags=lag)
            model_fitted = model.fit()

            # Предсказание на тестовой выборке
            predictions = model_fitted.predict(start=len(self.train), end=len(self.data) - 1, dynamic=False)

            # Оценка MSE
            mse = mean_squared_error(self.test, predictions)

            # Проверка на лучшее MSE
            if mse < best_mse:
                best_mse = mse
                self.best_lag = lag
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Условие выхода, если достигнут порог MSE
            if mse < self.mse_threshold:
                break

            lag += 1
            iteration += 1

        print(f"Лучшее MSE: {best_mse:.6f}, оптимальное число лагов: {self.best_lag}")

    def fit_best_model(self):
        """Обучение модели с оптимальным числом лагов."""
        self.best_model = AutoReg(self.train, lags=self.best_lag).fit()

    def predict(self, steps=None):
        """
        Предсказание будущих значений.
        steps: Количество шагов вперед для предсказания. Если None, используется оптимальное количество лагов.
        """
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала запустите метод fit_best_model().")

        if steps is None:
            steps = self.best_lag  # Если шаги не указаны, используем лучшее число лагов

        predictions = self.best_model.predict(start=len(self.data), end=len(self.data) + steps - 1)
        return predictions

    def regression(self, steps=None):
        """Основной метод для выполнения авторегрессии и предсказания."""
        # Шаг 1: Разделение данных
        self.split_data()

        # Шаг 2: Поиск оптимального числа лагов
        self.find_best_lag()

        # Шаг 3: Обучение модели с лучшим числом лагов
        self.fit_best_model()

        # Шаг 4: Предсказание
        predictions = self.predict(steps=steps)
        return predictions


###########################################################################################


class PolynomialRegressionModel:
    def __init__(self, data, max_degree=20, train_size=0.9, max_iters=50, mse_threshold=1e-4,
                 no_improvement_threshold=20):
        """
        Инициализация модели полиномиальной регрессии с циклом по степеням полинома.
        data: Временной ряд (вектор-столбец)
        max_degree: Максимальная степень полинома (по умолчанию 10)
        train_size: Доля данных для тренировки (по умолчанию 0.9)
        max_iters: Максимальное количество итераций
        mse_threshold: Порог для минимального значения MSE
        no_improvement_threshold: Число итераций без улучшений перед остановкой
        """
        self.data = data
        self.max_degree = max_degree
        self.train_size = train_size
        self.max_iters = max_iters
        self.mse_threshold = mse_threshold
        self.no_improvement_threshold = no_improvement_threshold

        self.train = None
        self.test = None
        self.best_degree = None
        self.best_model = None
        self.mse_history = []

    def split_data(self):
        """Разделение данных на тренировочные и тестовые выборки."""
        X = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, shuffle=False)
        return X_train, X_test, y_train, y_test

    def fit(self):
        """Обучение модели полиномиальной регрессии с подбором степени полинома."""
        X_train, X_test, y_train, y_test = self.split_data()
        no_improvement_counter = 0
        best_mse = float('inf')
        iteration = 0

        for degree in range(1, self.max_degree + 1):
            # Создаем полиномиальные признаки
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            # Обучаем модель
            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            # Предсказания и оценка MSE
            predictions = model.predict(X_test_poly)
            mse = mean_squared_error(y_test, predictions)
            self.mse_history.append((degree, mse))

            print(f"Степень: {degree}, MSE: {mse:.6f}")

            # Проверка на лучшее MSE
            if mse < best_mse:
                best_mse = mse
                self.best_degree = degree
                self.best_model = model
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Условие выхода из цикла
            if mse < self.mse_threshold:
                print("Достигнут порог MSE, выход из цикла.")
                break
            if no_improvement_counter >= self.no_improvement_threshold:
                print("Отсутствие улучшений, выход из цикла.")
                break

            iteration += 1
            if iteration >= self.max_iters:
                print("Достигнуто максимальное количество итераций.")
                break

    def predict(self, steps=1):
        """
        Предсказание будущих значений.
        steps: Количество шагов вперед для предсказания.
        """
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала запустите метод fit().")

        # Предсказание будущих шагов
        X_future = np.arange(len(self.data), len(self.data) + steps).reshape(-1, 1)
        poly = PolynomialFeatures(degree=self.best_degree)
        X_future_poly = poly.fit_transform(X_future)
        predictions = self.best_model.predict(X_future_poly)
        return predictions


####################################################################


class LSTMModel:
    def __init__(self, data, additional_data=None, sequence_length=30, units = 50, train_size=0.8, dropout_rate=0.2, print_output = False):
        self.data = data
        self.additional_data = additional_data
        self.sequence_length = sequence_length
        self.train_size = train_size
        self.dropout_rate = dropout_rate
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.units = units
        self.print_output = print_output
    def prepare_data(self):
        """Подготовка данных для LSTM с учетом дополнительных временных рядов."""
        X = []
        y = []
        for i in range(self.sequence_length, len(self.data)):
            sequence = [self.data[i - self.sequence_length:i]]  # основной ряд
            if self.additional_data is not None:
                sequence.extend([self.additional_data[:, j][i - self.sequence_length:i]
                                 for j in range(self.additional_data.shape[1])])  # дополнительные ряды
            X.append(np.column_stack(sequence))
            y.append(self.data[i])  # целевой ряд
        return np.array(X), np.array(y)

    def split_data(self):
        """Разделение данных на тренировочные и тестовые выборки."""
        X, y = self.prepare_data()
        return train_test_split(X, y, train_size=self.train_size, shuffle=False)

    def build_model(self):
        """Создание и компиляция LSTM модели."""
        model = Sequential()
        # Добавляем Input слой для явного задания input_shape
        model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, epochs=100, batch_size=32):
        """Обучение модели на тренировочных данных."""
        self.model = self.build_model()
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self):
        """Оценка модели на тестовых данных."""
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        print(f'MSE: {mse}')
        if self.print_output == True: # Если подать True, выведет y_test и predictions для собственных тестов
            print(f'y_test для собственных ошибок:{self.y_test}')
            print(f'predictions для собственных ошибок:{predictions}')
        else: pass
        return mse

    def predict(self, steps=1):
        """Предсказание будущих значений на основе тестовых данных."""
        predictions = []
        last_sequence = self.X_test[-1]  # последняя последовательность из теста
        for _ in range(steps):
            pred = self.model.predict(last_sequence.reshape(1, self.sequence_length, -1))
            predictions.append(pred[0, 0])
            # обновляем последовательность, добавляем новое предсказание
            last_sequence = np.append(last_sequence[1:], pred, axis=0)
        return predictions
