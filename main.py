import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def replace_with_thresholds(dataframe, variable):
    """
    Verilen değişken için aykırı değerleri sınırlarla değiştirir.
    
    Parameters:
    dataframe (DataFrame): İşlenecek veri seti.
    variable (str): Aykırı değerlerin sınırlarla değiştirileceği değişken adı.
    
    Returns:
    DataFrame: Aykırı değerleri sınırlarla değiştirilmiş veri seti.
    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    
    dataframe[variable] = dataframe[variable].clip(lower=low_limit, upper=up_limit)
    return dataframe

def retail_data_prep(dataframe):
    """
    Perakende veri setini işleyerek temizler ve hazırlar.
    
    Parameters:
    dataframe (DataFrame): İşlenecek perakende veri seti.
    
    Returns:
    DataFrame: Temizlenmiş ve hazırlanmış perakende veri seti.
    """
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0) & (dataframe["Price"] > 0)]
    dataframe = replace_with_thresholds(dataframe, "Quantity")
    dataframe = replace_with_thresholds(dataframe, "Price")
    return dataframe

def create_invoice_product_df(dataframe, id=False):
    """
    Fatura-ürün matrisi oluşturur.
    
    Parameters:
    dataframe (DataFrame): İşlenecek veri seti.
    id (bool): Ürün ID'leri kullanılarak matris oluşturulsun mu?
    
    Returns:
    DataFrame: Fatura-ürün matrisi.
    """
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

def create_rules(dataframe, id=True, country="France"):
    """
    Belirli bir ülkeye göre birliktelik kuralları oluşturur.
    
    Parameters:
    dataframe (DataFrame): İşlenecek veri seti.
    id (bool): Ürün ID'leri kullanılarak kurallar oluşturulsun mu?
    country (str): Ülke adı.
    
    Returns:
    DataFrame: Oluşturulan birliktelik kuralları.
    """
    dataframe = dataframe[(dataframe['Country'] == country) & (dataframe["Quantity"] > 0) & (dataframe["Price"] > 0)]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def arl_recommender(rules_df, product_id, rec_count=1):
    """
    Ürün için önerilen ürünleri listeler.
    
    Parameters:
    rules_df (DataFrame): Birliktelik kuralları.
    product_id (int): Ürün ID'si.
    rec_count (int): Öneri sayısı.
    
    Returns:
    list: Ürün önerileri listesi.
    """
    sorted_rules = rules_df.sort_values("Lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        if product_id in list(product):
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    return recommendation_list[0:rec_count]

df = pd.read_excel("online_retail_II.xlsx", sheet_name='Year 2010-2011')
df = retail_data_prep(df)
rules = create_rules(df)
print(arl_recommender(rules, 22492, 1))
