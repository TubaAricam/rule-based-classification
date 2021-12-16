# KURAL TABANLI SINIFLANDIRMA
# RULE-BASED CLASSİFİCATİON

# Reading libraries
# Kütüphanelerin okutulması
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini okutup df'e atama işlemi
# The process of reading the data set and assigning it to df
df = pd.read_csv("datasets/persona.csv")

# İlk 5 gözlem
# First 5 observations
df.head()

# Genel bilgi
# General information
df.info()

# Son 5 gözlem
# Last 5 observations
df.tail()

# COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar
# Average earnings according to COUNTRY, SOURCE, SEX, AGE
df.groupby("COUNTRY").agg({"PRICE":"mean"})
agg_df=df.groupby("COUNTRY").agg({"PRICE":"mean"})

# df.groupby("COUNTRY")["PRICE"].agg("mean")  ## 2.yol / 2.option
## Sözlük kısmında ,value değerine bir değer daha gelince/ Adding one more value into the dictionary
# df.groupby("COUNTRY").agg({"PRICE":["mean","count"]})

# Sütun bilgisi / Information of columns
agg_df.columns
# İndex bilgisi / Informatıon of ındex
agg_df.index
# İndexleri sıfırlama / Resetting of ındex
agg_df.reset_index()

# Çıktıyı PRICE’a göre azalan olacak şekilde sort_values ile sıralayınız / According to decreasing  PRICE values sort the output
# Çıktıyı agg_df olarak kaydediniz./ Save the output as agg_df
agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"})
agg_df = agg_df.sort_values("PRICE",ascending=False)

# Index’te yer alan isimleri değişken ismine çeviriniz./ Convert the names in the index to variable names
agg_df = agg_df.reset_index()
agg_df.head()

#agg_df["AGE_CAT"] = pd.cut(x=agg_df["AGE"],right=True,
#                      bins=[0,18,23,30,40,agg_df["AGE"].max()])

#agg_df["AGE_CAT"] = pd.cut(x=agg_df["AGE"],right=False,
#                     bins=[0,18,23,30,40,agg_df["AGE"].max()])

#agg_df["AGE_CAT"] = pd.cut(x=agg_df["AGE"],
#                      bins=[0,18,23,30,40,agg_df["AGE"].max()],
#                      labels=["0-18","19-23","24-30","31-40","41-"
#                              + str(agg_df["AGE"].max())])
# agg_df["AGE_CAT"].value_counts()

#Age değişkenini kategorik değişkene çeviriniz ve / Conversion of age variable to categorical variable and
# Age değikeninin hangi değerlerden bölüneceğinin belirlenmesi / Determining which values to divide the age variable into
bins=[0,18,23,30,40,agg_df["AGE"].max()]
labels=["0-18","19-23","24-30","31-40","41-" + str(agg_df["AGE"].max())]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"],bins,False,labels)

agg_df.head()
agg_df.columns

# Örneğin: USA_ANDROID_MALE_0_18 , benzer çıktıyı oluşturmak yapılması gereken adımlar ;
# For example, USA_ANDROID_MALE_0_18 , steps to generate similar output

agg_df["customers_level_based"] = [str(i[0]).upper()+"_"+str(i[1]).upper()+"_"+str(i[2]).upper()+"_"+str(i[5]).upper() for i in agg_df.values]

# customers_level_based sütununu gruplaştırma ile ortalama harcama fiyatları alımı ve index resetlemesi
# Average spend prices by (customers_level_based) columns and resetting index
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE":"mean"})
agg_df = agg_df.reset_index()
agg_df.head()

# Değerlere 1 gelmesi / Getting 1 according to values
agg_df.value_counts()

# Yeni müşterileri  PRICE’a göre 4 segmente ayırınız./ Divide new customers into 4 segments according to PRICE.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz./Add the segments to agg_df as a variable with the SEGMENT naming.
# Segmentlere göre group by yapıp price mean, max, sum’larını alınız./ Group by segments and get price mean, max, sum.
# Veri setinden sadece C segmentini çekip analiz ediniz./ Group by segments and get price mean, max, sum.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"],4,labels=["D","C","B","A"])
agg_df.reset_index()
agg_df.groupby("SEGMENT").agg({"PRICE":["mean","max","sum"]})
agg_df[agg_df["SEGMENT"] == "C"].head()


#agg_df["SEGMENT_2"] = pd.qcut(agg_df["PRICE"],3,labels=["C","B","A"])
#agg_df.groupby("SEGMENT_2").agg({"PRICE":["mean","max","sum"]})
#agg_df["SEGMENT_2"].value_counts()


#Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve /Classify the new customers according to their segments and
# ne kadar gelir getirebileceğini tahmin ediniz./ Estimate how much income it can generate

# 17 yaşında ANDROID kullanan bir Brezilyalı kadın hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# What segment does a 17-year-old Brazilian woman using ANDROID belong to and how much income on average is she expected to earn?
user_ = "BRA_ANDROID_FEMALE_0-18"
agg_df[agg_df["customers_level_based"] == user_]