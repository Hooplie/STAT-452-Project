library(rpart)  # for regression trees
library(MASS)   # For ridge regression
library(glmnet) # For LASSO
library(pls)    # For partial least squares
library(nnet)   # for neural nets
library(randomForest) # for random forest 
library(gbm) # for boosting 
library(dplyr)

###############################
### Import and process data ###
###############################
### Read in data
input.data = read.csv("Data2020.csv") 
test.data = read.csv("Data2020testX.csv") 

########################
### Helper Functions ###
########################
### Create function to compute MSPEs
get.MSPE = function(Y, Y.hat){
  return(mean((Y - Y.hat)^2))
}

### Create function which constructs folds for CV
### n is the number of observations, K is the number of folds
get.folds = function(n, K) {
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold)
  fold.ids = fold.ids.raw[1:n]
  ### Shuffle the fold labels
  folds.rand = fold.ids[sample.int(n)]
  return(folds.rand)
}

### Rescale x1 so that columns of x2 range from 0 to 1
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

###########################
##### INITIAL FITTING #####
###########################

#############
### Setup ###
#############

set.seed(2928893)
K = 10 # Number of folds
### Container for CV MSPEs
all.models = c("LS", "Step", "Ridge", "LASSO-Min", "LASSO-1se", "PLS", "NNET",
               "Full-Tree", "Min-Tree", "1SE-Tree","RF", "Boosting")
CV.MSPEs = array(0, dim = c(length(all.models), K))
rownames(CV.MSPEs) = all.models
colnames(CV.MSPEs) = 1:K

### Get CV fold labels
n = nrow(input.data)
folds = get.folds(n, K)

### Construct candidate lambda values for ridge regression
lambda.vals = seq(from = 0, to = 100, by = 0.05)

### Define parameters for NNET 
best.pars = rep(0, times = K) # Container for tuning parameters
nodes = c(1, 3, 5)
shrinkage = c(0.001, 0.1, 0.5)
all.pars = expand.grid(nodes = nodes, shrink = shrinkage)
n.pars = nrow(all.pars)
K.inner = 5 # Number of folds for inner CV
M = 5 # Number of times to re-fit neural net

######################
### Fitting Models ###
######################

### Perform cross-validation
for (i in 1:K) {
  ### Get training and validation sets
  data.train = input.data[folds != i, ]
  data.valid = input.data[folds == i, ]
  
  # added for neural nets
  X.train.raw = select(data.train, -Y)
  X.train = rescale(X.train.raw, X.train.raw)
  X.valid.raw = select(data.valid, -Y)
  X.valid = rescale(X.valid.raw, X.train.raw)
  ### Container for tuning MSPEs
  tuning.MSPEs = array(0, dim = c(nrow(all.pars), K.inner))
  ### Get inner CV fold labels
  n.inner = nrow(data.train)
  folds.inner = get.folds(n.inner, K.inner)
  # ends here 
  
  Y.train = data.train$Y
  Y.valid = data.valid$Y
  ### We need the data matrix to have an intercept for ridge, and to not have an intercept for LASSO. 
  mat.train.int = model.matrix(Y ~ ., data = data.train)
  mat.train = mat.train.int[,-1]
  mat.valid.int = model.matrix(Y ~ ., data = data.valid)
  mat.valid = mat.valid.int[,-1]
  
  ##########
  ### LS ###
  ##########
  fit.ls = lm(Y ~ ., data = data.train)
  pred.ls = predict(fit.ls, data.valid)
  MSPE.ls = get.MSPE(Y.valid, pred.ls)
  CV.MSPEs["LS", i] = MSPE.ls
  
  ############
  ### Step ###
  ############
  fit.start = lm(Y ~ 1, data = data.train)
  fit.step = step(fit.start, list(upper = fit.ls), trace = 0)
  pred.step = predict(fit.step, data.valid)
  MSPE.step = get.MSPE(Y.valid, pred.step)
  CV.MSPEs["Step", i] = MSPE.step
  
  #############
  ### Ridge ###
  #############
  ### Fit ridge regression
  ### We already defined lambda.vals. No need to re-invent the wheel
  fit.ridge = lm.ridge(Y ~ ., lambda = lambda.vals, data = data.train)
  ### Get optimal lambda value
  ind.min.GCV = which.min(fit.ridge$GCV)
  lambda.min = lambda.vals[ind.min.GCV]
  ### Get coefficients for optimal model
  all.coefs.ridge = coef(fit.ridge)
  coef.min.ridge = all.coefs.ridge[ind.min.GCV,]
  ### Get predictions and MSPE on validation set
  pred.ridge = mat.valid.int %*% coef.min.ridge
  pred.ridge = as.numeric(pred.ridge)
  MSPE.ridge = get.MSPE(Y.valid, pred.ridge)
  CV.MSPEs["Ridge", i] = MSPE.ridge
  
  #############
  ### LASSO ###
  #############
  ### Fit model
  fit.LASSO = cv.glmnet(mat.train, Y.train)
  ### Get optimal lambda values
  lambda.min = fit.LASSO$lambda.min
  lambda.1se = fit.LASSO$lambda.1se
  ### Get predictions
  pred.min = predict(fit.LASSO, mat.valid, lambda.min)
  pred.1se = predict(fit.LASSO, mat.valid, lambda.1se)
  ### Get and store MSPEs
  MSPE.min = get.MSPE(Y.valid, pred.min)
  MSPE.1se = get.MSPE(Y.valid, pred.1se)
  CV.MSPEs["LASSO-Min", i] = MSPE.min
  CV.MSPEs["LASSO-1se", i] = MSPE.1se
  
  ###########
  ### PLS ###
  ###########
  ### Fitting PLS on training data 
  fit.pls = plsr(Y ~ ., data = data.train, validation = "CV", segments=10)
  ### Get the best model from PLS. 
  CV.pls = fit.pls$validation # All the CV information
  PRESS.pls = CV.pls$PRESS    # Sum of squared CV residuals
  CV.MSPE.pls = PRESS.pls / nrow(data.train)  # MSPE for internal CV
  ind.best.pls = which.min(CV.MSPE.pls) # Optimal number of components
  
  ### Get predictions and calculate MSPE on the validation fold
  ### Set ncomps equal to the optimal number of components
  pred.pls = predict(fit.pls, data.valid, ncomp = ind.best.pls)
  MSPE.pls = get.MSPE(Y.valid, pred.pls)
  CV.MSPEs["PLS", i] = MSPE.pls
  
  ############
  ### NNET ###
  ############
  ### Perform inner cross-validation
  for (j in 1:K.inner) {
    print(paste0(i, "-", j, " of ", K, "-", K.inner))
    ### Get training and validation sets, rescale as appropriate
    data.train.inner = data.train[folds.inner != j,]
    X.train.inner.raw = select(data.train.inner, -Y)
    X.train.inner = rescale(X.train.inner.raw, X.train.inner.raw)
    Y.train.inner = data.train.inner$Y
    data.valid.inner = data.train[folds.inner == j,]
    X.valid.inner.raw = select(data.valid.inner, -Y)
    X.valid.inner = rescale(X.valid.inner.raw, X.train.inner.raw)
    Y.valid.inner = data.valid.inner$Y
    ### Fit nnets with each parameter combination
    for(l in 1:n.pars){
      ### Get current parameter values
      this.n.hidden = all.pars[l, 1]
      this.shrink = all.pars[l, 2]
      ### containers to store refit models and their errors.
      all.nnets = list(1:M)
      all.SSEs = rep(0, times = M)
      ### Fit each model multiple times
      for (ii in 1:M) {
        ### Fit model
        fit.nnet = nnet(
          X.train.inner,
          Y.train.inner,
          linout = TRUE,
          size = this.n.hidden,
          decay = this.shrink,
          maxit = 500,
          trace = FALSE
        )
        ### Get model SSE
        SSE.nnet = fit.nnet$value
        ### Store model and its SSE
        all.nnets[[ii]] = fit.nnet
        all.SSEs[ii] = SSE.nnet
      }
      ### Get best fit using current parameter values
      ind.best = which.min(all.SSEs)
      fit.nnet.best = all.nnets[[ind.best]]
      ### Get predictions and MSPE, then store MSPE
      pred.nnet = predict(fit.nnet.best, X.valid.inner)
      MSPE.nnet = get.MSPE(Y.valid.inner, pred.nnet)
      tuning.MSPEs[l, j] = MSPE.nnet 
    }
  }
  
  ### Get best tuning MSPEs by minimizing average
  ave.tune.MSPEs = apply(tuning.MSPEs, 1, mean)
  best.comb = which.min(ave.tune.MSPEs)
  best.pars[i] = best.comb
  ######################
  ### Fit best model ###
  ######################
  ### Get chosen tuning parameter values
  best.n.hidden = all.pars[best.comb, "nodes"]
  best.shrink = all.pars[best.comb, "shrink"]
  ### containers to store refit models and their errors.
  all.nnets = list()
  all.SSEs = rep(0, times = M)
  ### Fit each model multiple times
  for (iii in 1:M) {
    ### Fit model
    fit.nnet = nnet(
      X.train,
      Y.train,
      linout = TRUE,
      size = best.n.hidden,
      decay = best.shrink,
      maxit = 500,
      trace = F
    )
    ### Get model SSE
    SSE.nnet = fit.nnet$value
    ### Store model and its SSE
    all.nnets[[iii]] = fit.nnet
    all.SSEs[iii] = SSE.nnet
  }
  
  ### Get best fit using current parameter values
  ind.best = which.min(all.SSEs)
  fit.best = all.nnets[[ind.best]]
  ### Get predictions and MSPE
  pred = predict(fit.best, X.valid)
  this.MSPE = get.MSPE(Y.valid, pred)
  CV.MSPEs["NNET", i] = this.MSPE
  
  #################
  ### Full Tree ###
  #################
  fit.tree = rpart(Y ~ ., data = data.train, cp = 0)
  ### Get the CP table
  info.tree = fit.tree$cptable
  ### Get predictions
  pred.full = predict(fit.tree, data.valid)
  MSPE.full = get.MSPE(Y.valid, pred.full)
  CV.MSPEs["Full-Tree", i] = MSPE.full
  
  ###################
  ### Min CV Tree ###
  ###################
  ### Get minimum CV error and corresponding CP value
  ind.best = which.min(info.tree[, "xerror"])
  CV.best = info.tree[ind.best, "xerror"]
  CP.best = info.tree[ind.best, "CP"]
  ### Get the geometric mean of best CP with one above it
  if (ind.best == 1) {
    ### If minimum CP is in row 1, store this value
    CP.GM = CP.best
  } else{
    ### If minimum CP is not in row 1, average this with the value from the
    ### row above it.
    ### Value from row above
    CP.above = info.tree[ind.best - 1, "CP"]
    ### (Geometric) average
    CP.GM = sqrt(CP.best * CP.above)
  }
  ### Fit minimum CV error tree
  fit.tree.min = prune(fit.tree, cp = CP.best)
  ### Get predictions and MSPE
  pred.min = predict(fit.tree.min, data.valid)
  MSPE.min = get.MSPE(Y.valid, pred.min)
  CV.MSPEs["Min-Tree", i] = MSPE.min
  
  ########################
  ### 1SE Rule CV Tree ###
  ########################
  ### Get 1se rule CP value
  err.min = info.tree[ind.best, "xerror"]
  se.min = info.tree[ind.best, "xstd"]
  threshold = err.min + se.min
  ind.1se = min(which(info.tree[1:ind.best, "xerror"] < threshold))
  ### Take geometric mean with superior row
  CP.1se.raw = info.tree[ind.1se, "CP"]
  if (ind.1se == 1) {
    ### If best CP is in row 1, store this value
    CP.1se = CP.1se.raw
  } else{
    ### If best CP is not in row 1, average this with the value from the
    ### row above it.
    ### Value from row above
    CP.above = info.tree[ind.1se - 1, "CP"]
    ### (Geometric) average
    CP.1se = sqrt(CP.1se.raw * CP.above)
  }
  ### Prune the tree
  fit.tree.1se = prune(fit.tree, cp = CP.1se)
  ### Get predictions and MSPE
  pred.1se = predict(fit.tree.1se, data.valid)
  MSPE.1se = get.MSPE(Y.valid, pred.1se)
  CV.MSPEs["1SE-Tree", i] = MSPE.1se
  
  #####################
  ### Random Forest ###
  #####################
  fit.rf = randomForest(Y ~ ., data = data.train, importance = F)
  ### Get OOB predictions and MSPE, then store MSPE
  OOB.pred = predict(fit.rf)
  OOB.MSPE = get.MSPE(Y.valid, OOB.pred)
  CV.MSPEs["RF", i] = OOB.MSPE 
  
  ################
  ### Boosting ###
  ################
  fit.gbm = gbm(Y ~ ., data = data.train, distribution = "gaussian")

  ### Get predictions and MSPE, then store MSPE
  pred.gbm = predict(fit.gbm, data.valid)
  MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
  
  CV.MSPEs["Boosting", i] = MSPE.gbm 
}

### Get full-data MSPEs
full.MSPEs = apply(CV.MSPEs, 1, mean)

# table with all MSPEs including full data
all.MSPEs = cbind(CV.MSPEs, full.MSPEs)
colnames(all.MSPEs) = c(1:K, "Full")
print(signif(all.MSPEs), 3)

### MSPE Boxplot
boxplot(t(CV.MSPEs), las=2, main="MSPE Boxplot")

### Compute RMSPEs
CV.RMSPEs = apply(CV.MSPEs, 2, function(W){
  best = min(W)
  return(W/best)
})

### RMSPE Boxplot
boxplot(t(CV.RMSPEs), las=2, ylim=c(1,1.25), main="RMSPE Boxplot")

# Clear winner is Boosting, even beats a semi-tuned NNET. Now we need to
# tune our boosting model. 


#################################
##### Tuning Boosting Model #####
#################################

######################
### Tuning Round 1 ###
######################

set.seed(12345)

### Set parameter values 
max.trees = 10000
all.shrink = c(0.001, 0.01, 0.1) 
all.depth = c(1, 2, 3, 4, 5, 6, 7, 8) 

# Create grid of all combinations of parameter values
all.pars = expand.grid(shrink = all.shrink, depth=all.depth)
n.pars = nrow(all.pars)

# number of folds for CV
K = 5 

### Get CV fold labels
n = nrow(input.data)
folds = get.folds(n, K)

### Create container for CV MSPEs
CV.MSPEs1 = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data
  data.train = input.data[folds != i,]
  data.valid = input.data[folds == i,]
  Y.valid = data.valid$Y
  
  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]
    
    ### Fit model using current parameter values.
    fit.gbm = gbm(Y ~ ., data = data.train, distribution = "gaussian", 
                  n.trees = max.trees, interaction.depth = this.depth, shrinkage = this.shrink, 
                  bag.fraction = 0.8)
    
    ### Choose how many trees to keep using RoT's rule. This will print many
    ### warnings about not just using the number of trees recommended by
    ### gbm.perf(). We have already addressed this problem though, so we can
    ### just ignore the warnings.
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2

    ### Check to make sure that RoT's rule doesn't tell us to use more than 10000
    ### trees (max.trees). If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }

    ### Get predictions and MSPE, then store MSPE
    pred.gbm = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
    
    CV.MSPEs1[i, j] = MSPE.gbm 
  }
}

### Get full-data MSPEs
full.MSPEs1 = apply(CV.MSPEs1, 2, mean)
### best parameters with lowest MSPEs
full.MSPEs1[which.min(full.MSPEs1)]
order(full.MSPEs1)

### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is shrinkage-depth
names.pars = paste0(all.pars$shrink,"-",all.pars$depth)
colnames(CV.MSPEs1) = names.pars

### Make boxplot
boxplot(CV.MSPEs1, las = 2, main = "MSPE Boxplot")

### Get relative MSPEs and make boxplot
CV.RMSPEs1 = apply(CV.MSPEs1, 1, function(W) W/min(W))
boxplot(t(CV.RMSPEs1), las = 2, main = "RMSPE Boxplot")

# From analyzing the MSPE on full data and the boxplots:
# the best shrinkage is 0.01, with 0.001 in second and 0.1 last. 
# the best depth seems to be 8 followed by 5, 6 and 7.


######################
### Tuning Round 2 ###
######################

set.seed(123456)

### Set parameter values 
max.trees = 10000
all.shrink = c(0.0075, 0.01, 0.0125) 
all.depth = c(5, 6, 7, 8, 9) 

# Create grid of all combinations of parameter values
all.pars = expand.grid(shrink = all.shrink, depth=all.depth)
n.pars = nrow(all.pars)

# number of folds for CV
K = 5 

### Get CV fold labels
n = nrow(input.data)
folds = get.folds(n, K)

### Create container for CV MSPEs
CV.MSPEs1 = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data
  data.train = input.data[folds != i,]
  data.valid = input.data[folds == i,]
  Y.valid = data.valid$Y
  
  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]
    
    ### Fit model using current parameter values.
    fit.gbm = gbm(Y ~ ., data = data.train, distribution = "gaussian", 
                  n.trees = max.trees, interaction.depth = this.depth, shrinkage = this.shrink, 
                  bag.fraction = 0.8)
    
    ### Choose how many trees to keep using RoT's rule. This will print many
    ### warnings about not just using the number of trees recommended by
    ### gbm.perf(). We have already addressed this problem though, so we can
    ### just ignore the warnings.
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2
    
    ### Check to make sure that RoT's rule doesn't tell us to use more than 10000
    ### trees (max.trees). If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }
    
    ### Get predictions and MSPE, then store MSPE
    pred.gbm = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
    
    CV.MSPEs1[i, j] = MSPE.gbm 
  }
}

### Get full-data MSPEs
full.MSPEs1 = apply(CV.MSPEs1, 2, mean)
### best parameters with lowest MSPEs
full.MSPEs1[which.min(full.MSPEs1)]
order(full.MSPEs1)

### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is shrinkage-depth
names.pars = paste0(all.pars$shrink,"-",all.pars$depth)
colnames(CV.MSPEs1) = names.pars

### Make boxplot
boxplot(CV.MSPEs1, las = 2, main = "MSPE Boxplot")

### Get relative MSPEs and make boxplot
CV.RMSPEs1 = apply(CV.MSPEs1, 1, function(W) W/min(W))
boxplot(t(CV.RMSPEs1), las = 2, main = "RMSPE Boxplot")

# From analyzing the MSPE on full data and the boxplots:
# the best shrinkage is 0.0125 
# the best depth seems to be 9 followed by 8. 


######################
### Tuning Round 3 ###
######################

set.seed(123)

### Set parameter values 
max.trees = 10000
all.shrink = c(0.0125,0.02,0.03) 
all.depth = c(5, 6, 7, 8, 9) 

# Create grid of all combinations of parameter values
all.pars = expand.grid(shrink = all.shrink, depth=all.depth)
n.pars = nrow(all.pars)

# number of folds for CV
K = 5 

### Get CV fold labels
n = nrow(input.data)
folds = get.folds(n, K)

### Create container for CV MSPEs
CV.MSPEs1 = array(0, dim = c(K, n.pars))

for(i in 1:K){
  ### Print progress update
  print(paste0(i, " of ", K))
  
  ### Split data
  data.train = input.data[folds != i,]
  data.valid = input.data[folds == i,]
  Y.valid = data.valid$Y
  
  ### Fit boosting models for each parameter combination
  for(j in 1:n.pars){
    ### Get current parameter values
    this.shrink = all.pars[j,"shrink"]
    this.depth = all.pars[j,"depth"]
    
    ### Fit model using current parameter values.
    fit.gbm = gbm(Y ~ ., data = data.train, distribution = "gaussian", 
                  n.trees = max.trees, interaction.depth = this.depth, shrinkage = this.shrink, 
                  bag.fraction = 0.8)
    
    ### Choose how many trees to keep using RoT's rule. This will print many
    ### warnings about not just using the number of trees recommended by
    ### gbm.perf(). We have already addressed this problem though, so we can
    ### just ignore the warnings.
    n.trees = gbm.perf(fit.gbm, plot.it = F) * 2
    
    ### Check to make sure that RoT's rule doesn't tell us to use more than 10000
    ### trees (max.trees). If it does, add extra trees as necessary
    if(n.trees > max.trees){
      extra.trees = n.trees - max.trees
      fit.gbm = gbm.more(fit.gbm, extra.trees)
    }
    
    ### Get predictions and MSPE, then store MSPE
    pred.gbm = predict(fit.gbm, data.valid, n.trees)
    MSPE.gbm = get.MSPE(Y.valid, pred.gbm)
    
    CV.MSPEs1[i, j] = MSPE.gbm 
  }
}

### Get full-data MSPEs
full.MSPEs1 = apply(CV.MSPEs1, 2, mean)
### best parameters with lowest MSPEs
full.MSPEs1[which.min(full.MSPEs1)]
order(full.MSPEs1)

### We can now make an MSPE boxplot. First, add column names to indicate
### which parameter combination was used. Format is shrinkage-depth
names.pars = paste0(all.pars$shrink,"-",all.pars$depth)
colnames(CV.MSPEs1) = names.pars

### Make boxplot
boxplot(CV.MSPEs1, las = 2, main = "MSPE Boxplot")

### Get relative MSPEs and make boxplot
CV.RMSPEs1 = apply(CV.MSPEs1, 1, function(W) W/min(W))
boxplot(t(CV.RMSPEs1), las = 2, main = "RMSPE Boxplot")

# From analyzing the MSPE on full data and the boxplots:
# the best shrinkage is 0.03
# the best depth 8


#######################
##### Final model #####
#######################

library(gbm)

input.data = read.csv("Data2020.csv") 
test.data = read.csv("Data2020testX.csv") 

set.seed(98765)

max.trees = 10000

fit.gbm.final = gbm(Y ~ ., data = input.data, distribution = "gaussian", 
              n.trees = max.trees, interaction.depth = 8, shrinkage = 0.03, 
              bag.fraction = 0.8)

### Choose how many trees to keep using RoT's rule. This will print many
### warnings about not just using the number of trees recommended by
### gbm.perf(). We have already addressed this problem though, so we can
### just ignore the warnings.
n.trees.best = gbm.perf(fit.gbm.final, plot.it = F) * 2

### Check to make sure that RoT's rule doesn't tell us to use more than 10000  
### trees (max.trees). If it does, add extra trees as necessary
if(n.trees.best > max.trees){
  extra.trees = n.trees.best - max.trees
  fit.gbm.final = gbm.more(fit.gbm, extra.trees)
}

### Get predictions and output to file
pred.gbm.final = predict(fit.gbm.final, test.data, n.trees.best)

write.table(pred.gbm.final, file="final prediction", sep = ",", row.names = F, col.names = F)
