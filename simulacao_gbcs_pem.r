# ------------------------------------------------------------------------------------
# Simulação com inclusão do modelo exponencial por partes (PEM) com xgboost para dados GBCS
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

# Carregar dados GBCS
data(gbcsCS, package="condSURV")

# Preparação dos dados
gbcsCS$months <- gbcsCS$survtime/30.4167
gbcsCS$grade <- as.factor(gbcsCS$grade)

# Divisão treino/teste
set.seed(1)
n <- nrow(gbcsCS)
split <- sample.int(n, size = round(3*n/4))
gbcsCS$id2 <- 1:n
gbcsCS$train <- gbcsCS$id2 %in% split
gbcs.exp <- gbcsCS[rep(seq(nrow(gbcsCS)), 20),]

# Definir variáveis de tempo e status para consistência
gbcs.exp$eventtime <- gbcs.exp$survtime
gbcs.exp$status <- gbcs.exp$censdead

# Modelo verdadeiro para GBCS (similar ao do código original)
true_mod <- stpm2(Surv(eventtime, status)~nsx(age,3) + menopause + hormone + nsx(size,3) * grade + nodes + 
                  nsx(prog_recp,3) + nsx(estrg_recp, 3), data=gbcs.exp,
                  smooth.formula=~nsx(log(eventtime), df=3) + nsx(estrg_recp, 3) : log(eventtime))

# Função de log-cumulative hazard para GBCS
logcumhaz <- function(t, x, betas) {
  basis <- rstpm2::nsx(log(t), knots = c(6.50328867253789, 7.05701036485508),
                       Boundary.knots = c(4.27666611901606, 7.80384330353877), 
                       intercept = FALSE, derivs = c(2, 2), centre = FALSE, log = FALSE)
  
  res <- betas[["(Intercept)"]] +
    betas[["nsx(log(eventtime), df = 3)1"]] * basis[,1] + 
    betas[["nsx(log(eventtime), df = 3)2"]] * basis[,2] +
    betas[["nsx(log(eventtime), df = 3)3"]] * basis[,3] +
    betas[["nsx(age, 3)1"]] * x[["nsx(age, 3)1"]] +
    betas[["nsx(age, 3)2"]] * x[["nsx(age, 3)2"]] +
    betas[["nsx(age, 3)3"]] * x[["nsx(age, 3)3"]] +
    betas[["menopause"]] * x[["menopause"]] +
    betas[["hormone"]] * x[["hormone"]] + 
    betas[["nsx(size, 3)1"]] * x[["nsx(size, 3)1"]] +
    betas[["nsx(size, 3)2"]] * x[["nsx(size, 3)2"]] +
    betas[["nsx(size, 3)3"]] * x[["nsx(size, 3)3"]] +
    betas[["nodes"]] * x[["nodes"]] + 
    betas[["grade2"]] * x[["grade2"]] + 
    betas[["grade3"]] * x[["grade3"]] + 
    betas[["nsx(prog_recp, 3)1"]] * x[["nsx(prog_recp, 3)1"]] +
    betas[["nsx(prog_recp, 3)2"]] * x[["nsx(prog_recp, 3)2"]] +
    betas[["nsx(prog_recp, 3)3"]] * x[["nsx(prog_recp, 3)3"]] +
    betas[["nsx(estrg_recp, 3)1"]] * x[["nsx(estrg_recp, 3)1"]] +
    betas[["nsx(estrg_recp, 3)2"]] * x[["nsx(estrg_recp, 3)2"]] +
    betas[["nsx(estrg_recp, 3)3"]] * x[["nsx(estrg_recp, 3)3"]] +
    betas[["nsx(size, 3)1:grade2"]] * x[["nsx(size, 3)1:grade2"]] +
    betas[["nsx(size, 3)2:grade2"]] * x[["nsx(size, 3)2:grade2"]] +
    betas[["nsx(size, 3)3:grade2"]] * x[["nsx(size, 3)3:grade2"]] +
    betas[["nsx(size, 3)1:grade3"]] * x[["nsx(size, 3)1:grade3"]] +
    betas[["nsx(size, 3)2:grade3"]] * x[["nsx(size, 3)2:grade3"]] +
    betas[["nsx(size, 3)3:grade3"]] * x[["nsx(size, 3)3:grade3"]] +
    betas[["nsx(estrg_recp, 3)1:log(eventtime)"]] * x[["nsx(estrg_recp, 3)1"]] * log(t) +
    betas[["nsx(estrg_recp, 3)2:log(eventtime)"]] * x[["nsx(estrg_recp, 3)2"]] * log(t) +
    betas[["nsx(estrg_recp, 3)3:log(eventtime)"]] * x[["nsx(estrg_recp, 3)3"]] * log(t)
  
  return(res)
}

# Função para gerar dados simulados
gen_data <- function(true_mod) {
  cov <- as.data.frame(true_mod@x)
  dat <- simsurv(betas = coef(true_mod), x = cov, logcumhazard = logcumhaz,
                 maxt = 3650, interval = c(1E-8,1E5))
  dat <- cbind(gbcs.exp[,c("age", "menopause", "hormone", "size", "grade", "nodes", 
                          "prog_recp", "estrg_recp", "train")], dat[, c("eventtime", "status")])
  dat <- as.data.frame(dat)
  dat$train <- gbcs.exp$train
  return(dat)
}

# Função para criar dados PEM (mantida igual ao original)
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

# Função para ajustar modelo PEM (mantida igual ao original)
fit_pem_model <- function(pem_data, features) {
  dtrain <- xgb.DMatrix(data = as.matrix(pem_data[, features]),
                        label = pem_data$delta_ij,
                        weight = pem_data$t_ij,
                        base_margin = pem_data$log_tij)
  params <- list(objective = "count:poisson", base_score = 1, eval_metric = "logloss")
  model <- xgb.train(params = params, data = dtrain, nrounds = 100)
  return(model)
}

# Função de predição para XGBoost (mantida igual ao original)
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
    pem_pred$grade <- as.numeric(as.factor(pem_pred$grade))
    dtest <- xgb.DMatrix(as.matrix(pem_pred[, features]),
                         base_margin = pem_pred$log_tij)
    lambda <- predict(object, dtest)
    cum_hazard <- rowsum(lambda * pem_pred$t_ij, group = pem_pred$id2)
    surv <- exp(-cum_hazard[, 1])
    preds[, k] <- 1 - surv
  }
  return(preds)
}

# Função principal de simulação para GBCS
sim_run_gbcs <- function(true_mod) {
  tic("data generation")
  dat <- gen_data(true_mod)
  toc()
  
  dataleft <- dat[dat$train,]
  dataright <- dat[!dat$train,]
  dataleft$id2 <- 1:nrow(dataleft)
  dataright$id2 <- (1 + nrow(dataleft)):(nrow(dataleft) + nrow(dataright))
  
  # Definir pontos de corte para PEM
  cutpoints <- c(365, 730, 1095, 1460, 1825, 2190)
  
  # Criar dados PEM
  pem_data <- create_pem_data(dataleft, cutpoints = cutpoints)
  pem_data$grade <- as.numeric(as.factor(pem_data$grade))
  
  # Definir features para o modelo PEM
  pem_features <- c("age", "menopause", "hormone", "size", "grade", "nodes", 
                   "prog_recp", "estrg_recp")
  
  # Ajustar modelo PEM
  pem_model <- fit_pem_model(pem_data, pem_features)
  
  tic("Scoring")
  res <- Score(list("xgb.Booster" = pem_model),
               formula = Surv(eventtime, status) ~ 1,
               data = dataright,
               conf.int = FALSE,
               times = c(1460, 1825, 2190),  # ~4, 5 e 6 anos
               summary = c("risks", "IPA", "ibs"),
               predictRisk.args = list("xgb.Booster" = list(cutpoints = cutpoints, 
                                      features = pem_features)))
  toc()
  
  return(list(res = res))
}

# Rodar simulação para GBCS
res_gbcs <- sim_run_gbcs(true_mod)
print(res_gbcs$res$Brier$score)