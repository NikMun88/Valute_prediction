import requests
import pandas as pd

# Получить данные с Московской биржи по доллару
def get_data_dollar():
    # Запрос на получение данных с Московской биржи (валютная пара USD/RUB)
    url = 'https://iss.moex.com/iss/engines/currency/markets/selt/securities/USD000UTSTOM/candles.json'

    # Параметры запроса
    params = {
        'from': '2023-12-31',
        'till': '2024-10-10',
        'interval': 24  # Дневной интервал
    }

    response = requests.get(url, params=params)

    # Проверяем успешность запроса
    if response.status_code == 200:
        data = response.json()['candles']['data']
        columns = response.json()['candles']['columns']

        # Создание DataFrame
        df = pd.DataFrame(data, columns=columns)
        # Сохранение в Excel
        df.to_excel('dollar_data.xlsx', index=False, engine='openpyxl')
        print(df.head())
        return df
    else:
        print(f"Error: {response.status_code}")

# Получить данные с Московской биржи по евро
def get_data_euro():
    # Запрос на получение данных с Московской биржи (валютная пара EUR/RUB)
    url = 'https://iss.moex.com/iss/engines/currency/markets/selt/securities/EUR_RUB__TOM/candles.json'

    # Параметры запроса
    params = {
        'from': '2023-12-31',  # Начало периода
        'till': '2024-10-10',  # Конец периода
        'interval': 24  # Дневной интервал
    }

    response = requests.get(url, params=params)

    # Проверяем успешность запроса
    if response.status_code == 200:
        data = response.json()['candles']['data']
        columns = response.json()['candles']['columns']

        # Создание DataFrame
        df = pd.DataFrame(data, columns=columns)
        df.to_excel('euro_data.xlsx', index=False, engine='openpyxl')
        print(df.head())
        return df
    else:
        print(f"Error: {response.status_code}")

# Функция для Z-нормализации с сохранением параметров
def z_normalize(values):
    mean = values.mean()
    std = values.std()
    normalized = (values - mean) / std
    return normalized, mean, std
# Восстановление данных
def inverse_z(values, mean, std):
    return (values * std) + mean

get_data_dollar()
get_data_euro()