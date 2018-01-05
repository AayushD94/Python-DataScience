sum(new3[,15]==0)
sum(new3[,15]== 1)

sum(new3[,15]== 0)/(sum(new3[,15]== 1)+sum(new3[,15]==0))
rf3 <- randomForest(Salary~.-Salary, data = new3, mtry = 5, ntree = 200, importance = TRUE)
importance(rf3)

newsam <- sample(nrow(new2), 0.75*nrow(new2), replace = FALSE) 
new_train <- new3[newsam,]
new_test <- new3[-newsam,]


rfm1 <- randomForest(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, data = new_train, mtry = 2, ntree = 1000)
prdrf <- predict(rfm1, new_test, type = "class")

View(prdrf)
tabrf <- table(new_test$Salary, prdrf)
tabrf
accrf <- sum(diag(tabrf))/sum(tabrf)

library(pROC)
library(ROCR)

str(prdrf)
roc <- roc(as.numeric(new_test$Salary),as.numeric(prdrf))
plot(roc, type = "b")
auc(roc)

rf <- function(ntree = 250, mtry = 4){
  require(randomForest)
  rfm <- randomForest(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, data = new_train, mtry = mtry, ntree = ntree)
  require(pROC)
  
  prd <- predict(rfm,new_test, type = "class")
  tab <- table(new_test$Salary, prd)
  acc <- sum(diag(tab))/sum(tab)
  rc <- roc(as.numeric(new_test$Salary),as.numeric(prd))
  return(print(paste("The Accuracy for Random Forest Model is: ",acc," And the Area under the curve is ", auc(rc))))
}

rf2 <- function(ntree = 250, mtry = 4){
  require(randomForest)
  rfm <- randomForest(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, data = new_train, mtry = mtry, ntree = ntree)
  require(pROC)
  
  prd <- predict(rfm,new_test, type = "class")
  tab <- table(new_test$Salary, prd)
  acc <- sum(diag(tab))/sum(tab)
  rc <- roc(as.numeric(new_test$Salary),as.numeric(prd))
  return(plot(rc, type = "b"))
}

varImpPlot(rfm1)



for (i in 2:5){
  require(randomForest)
  rfm <- randomForest(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, data = new_train, mtry = i, ntree = 500)
  require(pROC)
  prd <- predict(rfm,new_test, type = "class")
  tab <- table(new_test$Salary, prd)
  acc <- sum(diag(tab))/sum(tab)
  rc <- roc(as.numeric(new_test$Salary),as.numeric(prd))
  print(paste("The Accuracy for Random Forest Model is: ",acc," And the Area under the curve is ", auc(rc)))
}

tree <- function(ntree = NULL,mtry= NULL){
  treeModel<-rpart(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, method="class", data=new_train)
  pred <- predict(treeModel, new_test, type = "class")
  tb <- table(new_test$Salary, pred)
  acc <- sum(diag(tb))/sum(tb)
  rc <- roc(as.numeric(new_test$Salary),as.numeric(pred))
  return(c(plot(treeModel),text(treeModel)))
}

tree2 <- function(ntree = NULL,mtry= NULL){
  treeModel<-rpart(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, method="class", data=new_train)
  pred <- predict(treeModel, new_test, type = "class")
  tb <- table(new_test$Salary, pred)
  acc <- sum(diag(tb))/sum(tb)
  rc <- roc(as.numeric(new_test$Salary),as.numeric(pred))
  return(print(paste("The Accuracy for Decision Tree Model is: ",acc," And the Area under the curve is ", auc(rc))))
}

library(e1071)

nb1 <- function(ntree= NULL,mtry= NULL){
  nbModel<-naive_bayes(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, data = new_train)
  pred <- predict(nbModel, new_test, type = "class")
  tb <- table(new_test$Salary, pred)
  acc <- sum(diag(tb))/sum(tb)
  rc <- roc(as.numeric(new_test$Salary),as.numeric(pred))
  return(print(paste("The Accuracy for Naive Bayes Model is: ",acc," And the Area under the curve is ", auc(rc))))
  
}


nb2 <- function(ntree= NULL,mtry= NULL){
  nbModel<-naive_bayes(Salary~age+fnlwgt+marital_status+occupation+relationship+capital_gain, data = new_train)
  pred <- predict(nbModel, new_test, type = "class")
  tb <- table(new_test$Salary, pred)
  acc <- sum(diag(tb))/sum(tb)
  rc <- roc(as.numeric(new_test$Salary),as.numeric(pred))
  return(plot(nbModel))
  
}


library(shiny)
ui <- fluidPage(sliderInput(inputId = "ntree", label = "Enter the Number of Trees", min = 100, max = 1000, value = 250),
                sliderInput(inputId = "mtry", label = "Enter the value of Mtry", min = 2, max = 6, value = 3),
                plotOutput(outputId = "roc"),
                textOutput(outputId = "text"),
                radioButtons("algo", "Choose the Algorithm:",
                             c("Random Forest" = "rf",
                               "Naive Bayes" = "nb",
                               "Decision Trees" = "dt")))
server <- function(input,output){
  output$roc <- renderPlot({ algo <- switch(input$algo,
                                            rf = rf2,
                                            dt = tree, nb = nb2)
  
  algo(input$ntree,input$mtry)})
  output$text <- renderPrint({algo <- switch(input$algo,
                                             rf = rf,
                                             dt = tree2, nb = nb1)
  
  algo(input$ntree,input$mtry)})
}
shinyApp(ui = ui, server = server)
