"""Verilen dosyadaki datayi modele hazir hale getirip tahminler ve sonuclari yeni bir dosyaya yazdiririm.

Kullanim seklim:
python modelim.py <dosyaAdi> <model_yolu> <sonuc_dosyaAdi>
"""

from sklearn import preprocessing, linear_model
import pandas as pd
import numpy as np
import json
import sys
import joblib


def dosyayiOkuyupDataFrameOlustur(dosyaAdi):
    """Verilen ham dosyayi dataframe'e ceviririm."""
    with open(dosyaAdi) as f:
        data = f.readlines()

    data = [json.loads(x) for x in data]
    data = pd.DataFrame(data)
    return data

def onislemleriYap(data):
    """Verilen dataframe uzerinde gerekli on islemleri yapip tahminlemeye hazir hale getiririm.
    
    Ornek olarak yaptigim islemler:
    
    - varsa y kolonunu dusurmek
    - x1'in karesini feature olarak eklemek
    - feature'lari dogru sirada secmek
    - ciktiyi geri gondermek
    """
    if 'y' in data:
        data.drop(['y'], axis=1, inplace=True)
    data = data[['x1']]
    data = data.astype(float)
    data['x1^2'] = data['x1'] ** 2
    data = data[['x1', 'x1^2']]
    return data

def tahminle(data, model_yolu):
    """Verilen dataframe uzerinde modeli kullanarak tahminleme yaparim.
    
    Yaptigim ornek islemler:
    - modeli diskten yuklerim
    - tahminleri yaparim
    - tahmileri dataframe'e ceviririm
    - sonuclari geri dondururum


    """
    model = joblib.load(model_yolu)
    tahmin = model.predict(data)
    tahmin = pd.Series(tahmin, index=data.index).rename("y_pred")
    tahmin = tahmin.to_frame()
    return tahmin

def sonuclariYaz(tahmin, dosyaAdi):
    """Verilen tahminleri dosyaya yazdiririm."""
    tahmin.to_csv(dosyaAdi)
    return

if __name__ == "__main__":
    DOSYA_YOLU = sys.argv[1]
    MODEL_YOLU = sys.argv[2]
    CIKTI_YOLU = sys.argv[3]
    data = dosyayiOkuyupDataFrameOlustur(DOSYA_YOLU)
    data = onislemleriYap(data)
    tahmin = tahminle(data, MODEL_YOLU)
    sonuclariYaz(tahmin, CIKTI_YOLU)
    print("Tahminler yazdirildi.")
    print("Program sonlandirildi.")
    sys.exit(0)

    
    