# Projet VSA - Groupe 3 : Understanding Hyperdimensional Computing for Parallel Single-Pass Learning

## Partie 1 : Reproduction des expériences de l'article
Dans cette partie, on clone le GitHub des auteurs de l'article et on réalise ensuite toutes les opérations nécessaires. 

**[Reproduction_of_table_results.ipynb](Reproduction_of_table_results.ipynb) :** Ce fichier est divisé en différentes parties, où nous calculons la précision des différents modèles pour chaque DataSet, pour 1 epoch et pour 10 epochs. Nous stockons les résultats manuellement par la suite dans le fichier vsaresults.txt

**[ReproductionResultsGVSA.ipynb](ReproductionResultsGVSA.ipynb) :** Ce fichier contient la reproduction du graphe de l'article (Figure 1), avec une extension où la production des graphes des autres DataSets a été fait. Pour faire cela, nous avons calculé le CDC et l'avons stocké dans [CDC.txt](CDC.txt). Les résultats de précision des modèles avec les différents DataSets ont eux aussi été utilisés avec le fichier [vsaresults.txt](vsaresults.txt)

**[Results_VSA.xlsx](Results_VSA.xlsx) :** Ce fichier Excel regroupe l'ensemble des résultats obtenus ainsi que le calcul des CDC pour les différents DataSets. 

## Partie 2 : Production de nouveaux résultats (Distance de Manhattan)

**Utilisation du Bundling au lieu de la descente du gradient et testing sur les datasets:** Nous avons remplacé la dernière partie où on utilise la descente du gradient pour prédire les classes des nouveaux vecteurs par un algorithme du bundling où on calcule le vecteur Bundling pour chaque classe sur tous les exemples de training de cette classe. Après on utilise des formules de similarité entre le vecteur à prédire et les vecteur Bundling de chaque classe (10 vecteurs si on a 10 classes) et choisir la classe qui donne la plus grande similarité.
Pour calculer la similarité, on opte pour deux méthodes: 
- La méthode avec les angles.
- La distance de Manhattan.
Implémentation d'un encodeur Manhattan: nous avons 
même implementé un [encodeur Manhattan](encoder_manhattan.py), un Bundling avec Manhattan et calculé la similarité avec la distance Manhattan (Manhattan end-to-end). Les résultats ont été très mauvais par rapport aux autres méthodes.

**Code pour cette partie:** [Bundling et ManhattanEncoding](Bundling_Projet_VSA.ipynb)

**Résultats: Comparaison de la performance selon les différents datasets** 



| DataSet                                              | Isolet  | Ucihar | MNIST  | Fashion-MNIST |
|------------------------------------------------------|---------|--------|--------|---------------|
| **RFF-G(2^3)^VSA with Bundling and RFF Similarity**  | 87.56%  | 80.42% | 84.46% | 73.17%        |
| **RFF-G(2^3)^VSA with Bundling and Manhattan Similarity** | 87.49%  | 80.66% |84.44% | 73.13%        |
| **Manhattan r(8) with Bundling**                     | 3.91%   | 18.32% | 20.19% | 13.87%        |

