
# ------------------------------------------------------------------------------------
# Simulação com inclusão do modelo exponencial por partes (PEM) com xgboost (Bender 2021)
# ------------------------------------------------------------------------------------

# Limpeza e bibliotecas
rm(list=ls())
library(flexsurv)
library(splines)
library(survival)
library(rstpm2)
library(pec)
library(riskRegression)
library(randomForestSRC)
library(simsurv)
library(mboost)
library(timereg)
library(mltools)
library(data.table)
library(tictoc)
library(survivalmodels)
library(mlr3)
library(mlr3proba)
library(paradox)
library(mlr3tuning)
library(mlr3extralearners)
library(xgboost)

# Dados
colon2 <- subset(survival::colon, subset=etype==1)
colon2$months <- colon2$time/30.4167
colon2$inv.serosa <- 1*(colon2$extent %in% c(3,4))
colon2$differ <- as.factor(colon2$differ)

colon3 <- na.omit(colon2)
set.seed(1)
n <- nrow(colon3)
split <- sample.int(n, size = round(3*n/4))
colon3$id2 <- 1:n
colon3$train <- colon3$id2 %in% split
colon2.exp <- colon3[rep(seq(nrow(colon3)), 20),]

D <- 3
knots <- 50
age.sp <- outer(colon2.exp$age, knots, ">") * outer(colon2.exp$age, knots, "-")^D
colon2.exp$age2 <- age.sp[,1]
knots <- 2
nodes.sp <- outer(colon2.exp$nodes, knots, ">") * outer(colon2.exp$nodes, knots, "-")^D
colon2.exp$nodes2 <- nodes.sp[,1]

true_mod <- stpm2(Surv(time, status)~(age + age2) * rx + sex + obstruct + perfor + adhere +
                  nodes + nodes2 + node4 + differ + inv.serosa + surg,
                  data=colon2.exp, smooth.formula=~nsx(log(time), df=3))

logcumhaz <- function(t, x, betas) {
  basis <- rstpm2::nsx(log(t), knots = c(5.5606816, 6.3595739),
                       Boundary.knots = c(2.0794415, 7.8991535), intercept = FALSE,
                       derivs = c(2, 2), centre = FALSE, log = FALSE)
  res <- betas[["(Intercept)"]] +
    betas[["nsx(log(time), df = 3)1"]] * basis[,1] + 
    betas[["nsx(log(time), df = 3)2"]] * basis[,2] +
    betas[["nsx(log(time), df = 3)3"]] * basis[,3] +
    betas[["age"]] * x[["age"]] +
    betas[["age2"]] * x[["age2"]] +
    betas[["rxLev"]] * x[["rxLev"]] +
    betas[["rxLev+5FU"]] * x[["rxLev+5FU"]] + 
    betas[["sex"]] * x[["sex"]] +
    betas[["obstruct"]] * x[["obstruct"]] +
    betas[["perfor"]] * x[["perfor"]] +
    betas[["adhere"]] * x[["adhere"]] + 
    betas[["nodes"]] * x[["nodes"]] + 
    betas[["nodes2"]] * x[["nodes2"]] + 
    betas[["node4"]] * x[["node4"]] +
    betas[["differ2"]] * x[["differ2"]] +
    betas[["differ3"]] * x[["differ3"]] +
    betas[["inv.serosa"]] * x[["inv.serosa"]] +
    betas[["surg"]] * x[["surg"]] +
    betas[["age:rxLev"]] * x[["age:rxLev"]] +
    betas[["age:rxLev+5FU"]] * x[["age:rxLev+5FU"]] +
    betas[["age2:rxLev"]] * x[["age2:rxLev"]] +
    betas[["age2:rxLev+5FU"]] * x[["age2:rxLev+5FU"]] 
  return(res)
}

gen_data <- function(true_mod) {
  cov <- as.data.frame(true_mod@x)
  dat <- simsurv(betas = coef(true_mod), x = cov, logcumhazard = logcumhaz,
                 maxt = 3650, interval = c(1E-8,1E5))
  dat <- cbind(colon2.exp[,c(3:9, 11, 13:14, 18, 20:22)], dat[, c(2:3)])
  dat <- as.data.frame(dat)
  dat$train <- colon2.exp$train
  return(dat)
}

create_pem_data <- function(data, time_var = "eventtime", status_var = "status", cutpoints = NULL) {
  if (is.null(cutpoints)) {
    cutpoints <- sort(unique(data[[time_var]][data[[status_var]] == 1]))
  }
  intervals <- c(0, cutpoints, max(data[[time_var]]) + 1)
  pem_list <- list()
  for (i in 1:nrow(data)) {
    t <- data[[time_var]][i]
    d <- data[[status_var]][i]
    id_row <- data[i, ]
    for (j in 1:(length(intervals)-1)) {
      start <- intervals[j]
      end <- intervals[j+1]
      if (t > start) {
        t_ij <- ifelse(t < end, t - start, end - start)
        delta_ij <- ifelse(d == 1 && t <= end && t > start, 1, 0)
        row <- cbind(id_row, interval_start = start, interval_end = end,
                     t_ij = t_ij, delta_ij = delta_ij, interval_mid = (start + end)/2)
        pem_list[[length(pem_list) + 1]] <- row
      }
    }
  }
  pem_data <- do.call(rbind, pem_list)
  pem_data$log_tij <- log(pem_data$t_ij)
  return(pem_data)
}

fit_pem_model <- function(pem_data, features) {
  dtrain <- xgb.DMatrix(data = as.matrix(pem_data[, features]),
                        label = pem_data$delta_ij,
                        weight = pem_data$t_ij,
                        base_margin = pem_data$log_tij)
  params <- list(objective = "count:poisson", base_score = 1, eval_metric = "logloss")
  model <- xgb.train(params = params, data = dtrain, nrounds = 100)
  return(model)
}

predictRisk.xgb.Booster <- function(object, newdata, times, ...) {
  args <- list(...)
  cutpoints <- args$cutpoints
  features <- args$features

  if (is.null(cutpoints) || is.null(features)) {
    stop("Argumentos 'cutpoints' e 'features' são obrigatórios para predictRisk.xgb.Booster.")
  }
  intervals <- c(0, cutpoints, max(times) + 1)
  newdata$id2 <- 1:nrow(newdata)
  preds <- matrix(0, nrow = nrow(newdata), ncol = length(times))
  for (k in seq_along(times)) {
    t_cut <- times[k]
    pem_pred <- create_pem_data(newdata, time_var = "eventtime", status_var = "status", cutpoints = cutpoints)
    pem_pred <- pem_pred[pem_pred$interval_start < t_cut, ]
    pem_pred <- pem_pred[pem_pred$interval_start + pem_pred$t_ij <= t_cut, ]
    pem_pred$rx <- as.numeric(as.factor(pem_pred$rx))
    pem_pred$differ <- as.numeric(as.factor(pem_pred$differ))
    dtest <- xgb.DMatrix(as.matrix(pem_pred[, features]),
                         base_margin = pem_pred$log_tij)
    lambda <- predict(object, dtest)
    cum_hazard <- rowsum(lambda * pem_pred$t_ij, group = pem_pred$id2)
    surv <- exp(-cum_hazard[, 1])
    preds[, k] <- 1 - surv
  }
  return(preds)
}

sim_run <- function(true_mod) {
  tic("data generation")
  dat <- gen_data(true_mod)
  toc()
  dataleft <- dat[dat$train,]
  dataright <- dat[!dat$train,]
  dataleft$id2 <- 1:nrow(dataleft)
  dataright$id2 <- (1 + nrow(dataleft)):(nrow(dataleft) + nrow(dataright))
  cutpoints <- c(365, 730, 1095, 1460, 1825, 2190)
  pem_data <- create_pem_data(dataleft, cutpoints = cutpoints)
  pem_data$rx <- as.numeric(as.factor(pem_data$rx))
  pem_data$differ <- as.numeric(as.factor(pem_data$differ))
  pem_features <- c("age", "nodes", "rx", "differ", "sex", "obstruct", "perfor", "adhere", "node4", "inv.serosa", "surg")
  pem_model <- fit_pem_model(pem_data, pem_features)
  tic("Scoring")
  res <- Score(list("xgb.Booster" = pem_model),
               formula = Surv(eventtime, status) ~ 1,
               data = dataright,
               conf.int = FALSE,
               times = c(1460, 1825, 2190),
               summary = c("risks", "IPA", "ibs"),
               predictRisk.args = list("xgb.Booster" = list(cutpoints = cutpoints, features = pem_features)))
  toc()
  return(list(res = res))
}

# Rodar simulação
res <- sim_run(true_mod)
print(res$res$Brier$score)
