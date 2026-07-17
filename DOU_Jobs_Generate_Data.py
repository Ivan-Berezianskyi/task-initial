import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import joblib

def generate_dataset():
    file_path = "2025_june_raw.csv"

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "nazarantoniuk/it-salary-ukraine-2025",
        file_path,
        pandas_kwargs={"sep": ";"}
    )


    df.drop(columns=["Submitted at", "Кількість спеціалістів у вашій компанії в Україні", "Де ви зараз живете?", "Всі бонуси (на місяць)", "Вкажіть вашу основну спеціалізацію", "За яким напрямом ви навчалися / навчаєтеся у виші?", "У якій сфері проєкт, в якому ви зараз працюєте?"], inplace=True)

    df = df.dropna(axis="index", how="all")
    df = df.dropna(axis="columns", how="all")

    condition = df['Чи використовуєте ви у своїй роботі мови програмування (одну чи декілька)?'] == 'Ні, не використовую'
    df.loc[condition, 'Основна мова програмування'] = 'Не використовую'

    condition = df['Основна мова програмування'] != 'Не використовую'
    df.loc[condition, 'Чи використовуєте ви у своїй роботі мови програмування (одну чи декілька)?'] = 'Так, використовую'

    condition = df["Ваша основна зайнятість в ІТ зараз..."] == "Я фрилансер(-ка) в ІТ"
    df.loc[condition, 'Основний напрям роботи компанії, в якій працюєте'] = 'Фріланс'

    df = df.dropna(subset=["ЗАРПЛАТА / СУМАРНИЙ ДОХІД в IT у $$$ за місяць, лише ставка \nЧИСТИМИ - після сплати податків"])


    col_name = 'Основна мова програмування'

    counts = df[col_name].value_counts()
    popular_roles = counts[counts >= 25].index
    df.loc[~df[col_name].isin(popular_roles), col_name] = 'Інше'

    col_name = 'Почніть вводити і оберіть вашу ОСНОВНУ посаду зі списку'

    counts = df[col_name].value_counts()
    popular_roles = counts[counts >= 30].index
    df.loc[~df[col_name].isin(popular_roles), col_name] = 'Інше'


    df = df.rename(columns={
        "Ваша основна зайнятість в ІТ зараз...": "Employment",
        "ЗАРПЛАТА / СУМАРНИЙ ДОХІД в IT у $$$ за місяць, лише ставка \nЧИСТИМИ - після сплати податків": "Salary",
        "Тайтл": "Title",
        "Категорія": "Category",
        "Почніть вводити і оберіть вашу ОСНОВНУ посаду зі списку": "Position",
        "Чи використовуєте ви у своїй роботі мови програмування (одну чи декілька)?": "Uses_Code",
        "Основна мова програмування": "Language",
        "Основний напрям роботи компанії, в якій працюєте": "Company_Type",
        "Загальний стаж роботи за нинішньою ІТ-спеціальністю": "Experience",
        "Знання англійської мови": "English",
        "Ваш вік": "Age"
    })


    df_encoded = pd.get_dummies(df, drop_first=True)


    y = df_encoded['Salary']
    X = df_encoded.drop(columns=['Salary'])

    joblib.dump(X, 'datasets/X.joblib')
    joblib.dump(y, 'datasets/y.joblib')


if __name__ == "__main__":
    generate_dataset()