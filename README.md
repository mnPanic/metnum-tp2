# metnum-tp2

TP2 de Metodos Numéricos de Computación, exactas.

El README original de la cátedra se puede encontrar [aqui](README-catedra.md)

## Implementación

Las implementaciones de knn, pca y obtencion de autovalores se encuentran en
`src`, en los archivos provistos por la catedra.

En [notebooks/tests.py](notebooks/tests.py) se encuentran los tests hechos con
`unittest` para todas las implementaciones hechas en cpp. Para correrlos, basta
con hacer

```bash
python3 tests.py
```

## Experimentos

Los experimentos están dentro de [`notebooks`](notebooks). Para cada uno hay un
notebook separado. Y sus datos se encuentran en
[`notebooks/data`](notebooks/data), tambien uno por experimento.

## Kaggle

Los datos para entrenar y testear están en [`data`](data), y dentro están las
corridas que fueron enviadas a kaggle.

Para realizar el submission para kaggle,

1. Compilar

   ```bash
   ./build
   ```

2. Correr el ejecutable con los datos de test y train

    ```bash
    cd build
    $ ./tp2 -m 1 -i ../data/train.csv -q ../data/test.csv -o ../data/kaggle_submission_kNN_6_pow_5_PCA_34.csv
    reading matrix & labels from file: ../data/train.csv
    reading matrix from file: ../data/test.csv
    Running kNN+PCA with k=6 distance weights & alpha=34
    writing vector to csv file: kaggle_submission_kNN_6_pow_5_PCA_34.csv
    done!
    ```

3. Done! Ahora se puede submittear a [kaggle](https://www.kaggle.com/c/digit-recognizer/submit).
