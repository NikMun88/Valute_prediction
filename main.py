from models import AutoRegressionModel, PolynomialRegressionModel, LSTMModel
from Get_data import z_normalize, inverse_z
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Подгрузка данных. Не тревожим API постоянными запросами
file_path = 'dollar_data.xlsx'
dollar_data = pd.read_excel(file_path)
values = dollar_data['close']

file_path = 'euro_data.xlsx'
euro_data = pd.read_excel(file_path)
values_add = euro_data['close']


# Создаем экземпляр класса ЛИНЕЙНОЙ РЕГРЕССИИ, параметры по умолчанию
def Autoreg(values):
    model = AutoRegressionModel(values)
    # Выполняем авторегрессию, предсказываем оптимальное количество шагов вперед
    predictions = model.regression()
    print(f"Предсказания: {predictions}")
    # Выполняем предсказание с регулируемым шагом (steps = колво)
    predictions_x_steps = model.regression(steps=1)
    print(f"Предсказание на несколько шагов вперед: {predictions_x_steps}")


# Создаем экземпляр класса НЕЛИНЕЙНОЙ РЕГРЕССИИ, используем цикл по степеням полинома
def Polynom(values):
    poly_model = PolynomialRegressionModel(values, max_degree=25, no_improvement_threshold=25, train_size=0.99)
    # Обучаем модель и подбираем лучшую степень
    poly_model.fit()
    # Предсказываем 3 значения вперед
    predictions = poly_model.predict(steps=3)
    print(f"Предсказания на 3 шага вперед: {predictions}")


# Модель LSTM(RNN) рекурентной нейронной сети для редсказаний
def neuro(values):
    # Нормализация данных
    values_norm, mean_noth, std_noth = z_normalize(values_add)
    values_add_norm, mean, std = z_normalize(values)
    # Создание модели с нормализованными данными
    model = LSTMModel(
        data=values_norm,
        #additional_data=values_add_norm,
        units=400,
        sequence_length=30,
        train_size=0.98,
        dropout_rate=0.35)
    # Обучение модели
    model.train(epochs=58, batch_size=45)
    # Оценка модели (по желанию)
    model.evaluate()
    # Предсказание будущих значений
    predictions = model.predict(steps=2)
    predictions = pd.Series(predictions)
    print(predictions, mean, std)
    # Обратное преобразование предсказанных значений к исходному масштабу
    clear_value = inverse_z(predictions, mean, std)
    print(clear_value)
