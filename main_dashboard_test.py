# TESTS :
# Ouvrir le terminal sous anaconda, et changer de repertoire :
# cd C:\Users\jme1401\Desktop\Openclassrooms\7-Implémentez un modèle de scoring\Datas
# Taper : pytest projet_7_5_TEST_Dashboard.py : lance toutes les fonctions demarrant par test_

from projet_7_5_Dashboard import get_prediction


def test_get_prediction_accord():
    assert get_prediction(278288) == (0.238, 'Accordé')


def test_get_prediction_refus():
    assert get_prediction(192535) == (0.545, 'Refusé')

