# metnum-tp2

TP2 de Metodos Numéricos de Computación, exactas.

## Kaggle

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
