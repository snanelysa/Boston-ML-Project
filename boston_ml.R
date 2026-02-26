# Charger le package MASS qui contient le dataset Boston
library(MASS)
library(ggplot2)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(caret)
library(e1071)
library(randomForest)
library(keras)
install.packages("tensorflow")
library(tensorflow)

# Charger le dataset
data("Boston")

# Voir les 6 premières lignes
head(Boston)

# Résumé statistique de toutes les variables
summary(Boston)


# 1. Histogramme du prix des maisons
ggplot(Boston, aes(x = medv)) +
  geom_histogram(bins = 30, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution du prix médian des maisons",
       x = "Prix médian (medv)", y = "Fréquence") +
  theme_minimal()

# 2. Relation entre le prix et le % de ménages défavorisés
ggplot(Boston, aes(x = lstat, y = medv)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Relation entre lstat et medv",
       x = "% ménages défavorisés (lstat)", y = "Prix médian (medv)") +
  theme_minimal()

# 3. Carte des corrélations entre toutes les variables
M = cor(Boston)
corrplot(M, method = 'circle')


# Faire l'ACP
res.pca <- PCA(Boston, scale.unit = TRUE, ncp = 5, graph = FALSE)

# Visualiser le cercle des corrélations
fviz_pca_var(res.pca, col.var = "contrib") +
  theme_minimal() +
  labs(title = "Cercle des corrélations")

# Normalisation des données
boston_scale <- as.data.frame(scale(Boston[,1:13]))

# Séparation en jeu d'entraînement (80%) et de test (20%)
set.seed(123)
indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
trainData <- cbind(boston_scale, medv = Boston$medv)[indices,]
testData <- cbind(boston_scale, medv = Boston$medv)[-indices,]

# Construire le modèle de Régression Linéaire
lm_model <- lm(medv ~ ., data = trainData)
summary(lm_model)

# Prédictions sur le jeu de test
predictions_lm <- predict(lm_model, newdata = testData)

# Visualisation
plot(predictions_lm, testData$medv, main = "Régression Linéaire",
     xlab = "Prédictions", ylab = "Valeurs réelles",
     col = "blue", pch = 16, cex = 0.7)
abline(a = 0, b = 1, col = "red", lwd = 2)

# Évaluation
performance <- as.data.frame(postResample(predictions_lm, testData$medv))
colnames(performance) <- "RegLin"
print(performance)


# Construire le modèle SVM
svm_model <- svm(medv ~ ., data = trainData)

# Prédictions
predictions_svm <- predict(svm_model, newdata = testData)

# Évaluation
performance <- cbind(performance, SVM = postResample(predictions_svm, testData$medv))
print(performance)


# Construire le modèle Random Forest 
rf_model <- randomForest(medv ~ ., data = trainData)
print(rf_model)

# Prédictions
predictions_rf <- predict(rf_model, newdata = testData)

# Évaluation
performance <- cbind(performance, RandFor = postResample(predictions_rf, testData$medv))
print(performance)

# Créer la variable High/Low
trainData$medv_class <- ifelse(trainData$medv > 25, "High", "Low")
testData$medv_class <- ifelse(testData$medv > 25, "High", "Low")

# Convertir en facteur
trainData$medv_class <- factor(trainData$medv_class, levels = c("Low", "High"))
testData$medv_class <- factor(testData$medv_class, levels = c("Low", "High"))

# Vérifier
table(trainData$medv_class)


# Construire le modèle de Régression Logistique
logistic_model <- glm(medv_class ~ . - medv, data = trainData, family = binomial)

# Prédictions
predictions_logistic <- predict(logistic_model, newdata = testData, type = "response")

# Convertir les probabilités en classes
predictions_class <- ifelse(predictions_logistic > 0.5, "High", "Low")

# Évaluation
conf_matrix <- confusionMatrix(as.factor(predictions_class), testData$medv_class)
print(conf_matrix)

precision <- data.frame(reglog = conf_matrix$overall[1])
print(precision)


# Construire le modèle de l'Arbre de decision 
tree_model <- train(medv_class ~ . - medv, data = trainData, method = "rpart")

# Prédictions
predictions_tree <- predict(tree_model, newdata = testData)

# Évaluation
conf_matrix <- confusionMatrix(as.factor(predictions_tree), testData$medv_class)
print(conf_matrix)

precision$Arbre <- conf_matrix$overall[1]
print(precision)

# Construire le modèle KNN 
knn_model <- train(medv_class ~ . - medv, data = trainData, method = "knn", 
                   tuneGrid = data.frame(k = 5))

# Prédictions
predictions_knn <- predict(knn_model, newdata = testData)

# Évaluation
conf_matrix <- confusionMatrix(predictions_knn, testData$medv_class)
print(conf_matrix)

precision$KNN <- conf_matrix$overall[1]
print(precision)

# Construire le modèle  Random Forest
rf_model_class <- randomForest(medv_class ~ . - medv, data = trainData)
print(rf_model_class)

# Prédictions
predictions_rf_class <- predict(rf_model_class, newdata = testData)

# Évaluation
conf_matrix <- confusionMatrix(predictions_rf_class, testData$medv_class)
print(conf_matrix)

precision$RF <- conf_matrix$overall[1]
print(precision)


