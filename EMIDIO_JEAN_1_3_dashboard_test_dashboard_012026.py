# TESTS :
# Ouvrir le terminal sous anaconda, et changer de repertoire :
# cd C:\Users\jme1401\Desktop\Openclassrooms\7-Implémentez un modèle de scoring\Datas
# Taper : pytest main_dashboard_test.py : lance toutes les fonctions demarrant par test_

from main_dashboard import get_prediction


def test_get_prediction_accord():
    assert get_prediction(239207) == (0.407, 'Accordé')


def test_get_prediction_refus():
    assert get_prediction(412720) == (0.542, 'Refusé')

