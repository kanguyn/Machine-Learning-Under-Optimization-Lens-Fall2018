using DataFrames
using MLDataUtils
using OptimalTrees

function tune_train_OCT(train_X, train_y, valid_X, valid_y, test_X, test_y)
    println("\n##### Tuning and training the OCT #####")
    # Tuning parameters for an OCT
    #  - If we do not specify the cp value, then it is autotuned (preferred)
    #  - We use validation to determine the best parameter
    #    combination (max_depth, criterion, minbucket)
    #  - Because we are validating over the criterion, we
    #    select one criterion (in this case entropy) to be used
    #    as the evaluation metric.
    #  - We specify N=100 random starts, and set the random seed = 1.

    lnr = OptimalTrees.OptimalTreeClassifier(ls_random_seed=1,
                                             ls_num_tree_restarts=100)
    grid1 = OptimalTrees.GridSearch(lnr, Dict(
        :max_depth => 1:8,
        :minbucket => [5, 10, 15, 20, 25]
    ));

    OptimalTrees.fit!(grid1, train_X, train_y, valid_X, valid_y,
        validation_criterion = :entropy);

    # Print out the best parameters that were selected
    println("\nBest Parameters")
    println("----------------------")
    #println(grid1.best_params)
    println("cp = ", grid1.best_params[:cp])
    println("Max Depth = ", grid1.best_params[:max_depth])
    println("Minbucket = ", grid1.best_params[:minbucket])
    #println("Criterion = ", grid1.best_params[:criterion])
    println("----------------------\n")

    println("Tree:")
    println(grid1.best_lnr)
    println()

    #println(OptimalTrees.getnodefields(grid1.best_lnr, [:class, :probs, :split_type]))
    #println(OptimalTrees.getnodefields(grid1.best_lnr, [:class, :probs, :depth]))
    # Here are the node fields that we can display
    #println(fieldnames(grid1.best_lnr.tree_.nodes[1]))

    # Compute the in-sample and out-of-sample accuracy
    # Note: This problem has K = 3 classes, so we cannot compute AUC.
    train_acc = OptimalTrees.score(grid1.best_lnr, train_X, train_y, criterion=:misclassification)
    valid_acc = OptimalTrees.score(grid1.best_lnr, valid_X, valid_y, criterion=:misclassification)
    test_acc = OptimalTrees.score(grid1.best_lnr, test_X, test_y, criterion=:misclassification)
    println("OCT-Parallel Results")
    println("----------------------")
    println("Training accuracy = ", round(100 * train_acc, 2), "%")
    println("Validation accuracy = ", round(100 * valid_acc, 2), "%")
    println("Testing accuracy = ", round(100 * test_acc, 2), "%")

    return(grid1.best_lnr)

end


##############################
###### QUESTION 1 #############

data = readtable("iris.csv", makefactors = true)
X = data[:,1:(end-1)]
y = data[:,end]

srand(1)
(big_X, big_y), (test_X, test_y) = stratifiedobs((X, y), p=0.75);
(train_X, train_y), (valid_X, valid_y) = stratifiedobs((big_X, big_y), p=0.67);

lnr1 = tune_train_OCT(train_X, train_y, valid_X, valid_y, test_X, test_y)

###### QUESTION 2 #############
println("\nWhat is the most important variables in this dataset?")
println(OptimalTrees.variable_importance(lnr1))





######## question 4: necessary functions ########

println("\n\n**** Data imputation, leaving the outcome variable out ****\n")

# load missing data
missing = readtable("iris-missing.csv", makefactors = true)

#### Function to compute the MAE ###
function getMAE(X_real, X_imputed, X_missing)
    n, p = size(X_real)
    missing_count = 0
    totalError = 0.0
    for i in 1:n
        for d in 1:p
            if isna( X_missing[i,d] )
                missing_count += 1
                totalError += abs(X_real[i,d] - X_imputed[i,d])
            end
        end
    end
    return(totalError / missing_count)
end


function impute_missing(X_real, X_missing)
    # Data imputation with MEAN method
    X_mean = OptImpute.impute(X_missing, :mean);
    println("\n**** Mean method ****\nMAE : ", getMAE(X_real, X_mean, X_missing))

    println("\n\n***** Cross-Validation *****")
    # Use cross-validation to select the parameters
    #  - knn_k is the number of neighbors for K-NN
    #  - algorithm is either Block Coordinate Decent (BCD)
    #    or Coordinate Descent (CD)
    #  - norm is either L_1 (l1) or L_2 (l2).  L_2 uses the
    #    standard Euclidean distance, L_1 uses the absolute
    #    value distance.

    imputer = OptImpute.Imputer(:opt_knn)
    grid2 = OptImpute.GridSearch(imputer, Dict(
        :knn_k => [10, 20, 30, 40, 50],
        :algorithm => [:CD, :BCD],
        :norm => [:l1, :l2]
    ))
    X_opt_knn_cv = OptImpute.fit!(grid2, X_missing)

    println("Best Parameters")
    println("----------------------")
    println(grid2.best_params)
    println("Number of neighbors k = ", grid2.best_params[:knn_k])
    println("Algorithm = ", grid2.best_params[:algorithm])
    println("Norm = ", grid2.best_params[:norm])

    println("**** opt.knn method ****\nMAE: ", getMAE(X_real, X_opt_knn_cv, X_missing))

    return(X_mean, X_opt_knn_cv)
end



############ QUESTION 4 answers ########################
X_missing1 = missing[:,1:(end-1)]
X_real = data[:,1:(end-1)]

println("\n#################### Question 4 ####################\n")
X_mean1, X_opt_knn_cv1 = impute_missing(X_real, X_missing1)

println("\n**** Retrain OCT on imputed data with MEAN METHOD | q.4****\n")
y = missing[:,end]

srand(1)
(big_X2, big_y2), (test_X2, test_y2) = stratifiedobs((X_mean1, y), p=0.75);
(train_X2, train_y2), (valid_X2, valid_y2) = stratifiedobs((big_X2, big_y2), p=0.67);

lnr_mean1 = tune_train_OCT(train_X2, train_y2, valid_X2, valid_y2, test_X2, test_y2)
println("\nWhat is the most important variables in this dataset?")
println(OptimalTrees.variable_importance(lnr_mean1))

println("\n**** Retrain OCT on imputed data with OPT.KNN METHOD |4 ****\n")

srand(1)
(big_X3, big_y3), (test_X3, test_y3) = stratifiedobs((X_opt_knn_cv1, y), p=0.75);
(train_X3, train_y3), (valid_X3, valid_y3) = stratifiedobs((big_X3, big_y3), p=0.67);

lnr_knn1 = tune_train_OCT(train_X3, train_y3, valid_X3, valid_y3, test_X3, test_y3)
println("\nWhat is the most important variables in this dataset?")
println(OptimalTrees.variable_importance(lnr_knn1))


println("\nCORRELATION MATRIX of X_mean1")
println(cor(Matrix( X_mean1 )))

println("\nCORRELATION MATRIX of X_opt_knn_cv1")
println(cor(Matrix( X_opt_knn_cv1)))

println("\nCORRELATION MATRIX of X")
println(cor(Matrix( X)))







############ QUESTION 5 answers ########################
println("\n############## Question 5 ######################\n")

mean2, opt_knn_cv2 = impute_missing(data, missing)
X_mean2, X_opt_knn_cv2 = mean2[:,1:(end-1)], opt_knn_cv2[:,1:(end-1)]



println("\n**** Retrain OCT on imputed data with MEAN METHOD **** | q5")
srand(1)
(big_X4, big_y4), (test_X_missing, test_y_missing) = stratifiedobs((X_missing1, y), p=0.75);
(big_X5, big_y5), (test_X5, test_y5) = stratifiedobs((X_mean2, y), p=0.75);
(train_X4, train_y4), (valid_X4, valid_y4) = stratifiedobs((big_X5, big_y5), p=0.67);

test_X_mean2, test_X_opt_knn_cv2 = impute_missing(test_X, test_X_missing)

lnr_mean2 = tune_train_OCT(train_X4, train_y4, valid_X4, valid_y4, test_X_mean2, test_y)
println("\nWhat is the most important variables in this dataset?")
println(OptimalTrees.variable_importance(lnr_mean2))

println("\n\n**** Retrain OCT on imputed data with OPT.KNN METHOD ****| q5")
srand(1)
(big_X6, big_y6), (test_X6, test_y6) = stratifiedobs((X_opt_knn_cv2, y), p=0.75);
(train_X6, train_y6), (valid_X6, valid_y6) = stratifiedobs((big_X6, big_y6), p=0.67);

lnr_knn2 = tune_train_OCT(train_X6, train_y6, valid_X6, valid_y6, test_X_opt_knn_cv2, test_y)
println("\nWhat is the most important variables in this dataset?")
println(OptimalTrees.variable_importance(lnr_knn2))


println("\nCORRELATION MATRIX of X_mean2")
println(cor(Matrix( vcat(train_X4, valid_X4, test_X_mean2) )))

println("\nCORRELATION MATRIX of X_opt_knn_cv2")
println(cor(Matrix( vcat(train_X6, valid_X6, test_X_opt_knn_cv2) )))

println("\nCORRELATION MATRIX of X")
println(cor(Matrix( X)))
