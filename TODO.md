# TODO

- Metodo potencia & Deflacion OK
  - Falta citerio de corte.
- KNN
  - weighted vote
- PCA
- Cross validation con y analisis de resultados con K-Fold
  - Accuracy
  - Curvas de precision / recall
  - etc.

- Buscar parametros optimos k y $\alpha$ (hill climber / simulated annealing)
- (opcional pero no tanto) Participar en Diggit Recognizer de kaggle

- Recall para ciertos digitos / ver para qué digitos esta metrica es baja (3, 8)
- Analizar si hay que implementar diferente el el método PSA

## Experimentos

dia         manu              tropi           nacho
dom 07      knn weights
lun 08      1.2               2.3             2.3
mar 09                2.4 y 2.5       3, 4
mie 10      graficos, informe, bla  hipotesis y cosas
jue 11      graficos, informe, bla
vie 12

Todos con K-Fold con K = ? (4)

1. agregar evolucion a simm annealing

2. KNN (k)
  1.1 Weights por distancia vs uniforme OK
  1.2 Optimizar con simm ann el mejor k para kNN con weights distance_pow

2. KNN + PCA (k + alpha)

  2.1. Arreglar PCA OK
  2.2. Cache y PCA  08

  2.3. Optimizar con simm annealing obteniendo (k, alpha)

  Con params fijos (k, alpha)
  
  2.4. Entrenando con tamaños variados de imagenes para entrenamiento
      (5000, 10000, 25000, max).
  2.5. Probar con diferentes Ks y ver como varia las metricas

3. Accuracy y F1 para ambos de kNN k opt. y kNN + PCA con (k, alpha)
4. Kappa de Cohen
5. Concluir cual es mejor
