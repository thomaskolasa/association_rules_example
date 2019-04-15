library(arules)
library(ggplot2)

setwd("~/Downloads/association_rules_example/")

raw_df <- read.csv("2017-2014 CP WBS SALES Expanded.csv", sep = "\t")

# only look at product data without quantity
raw_df <- raw_df[, c(30, 32, 34, 36, 38)]

write.table(raw_df, "transactions.csv", sep = ",", 
            col.names = FALSE, row.names = FALSE)

transactions <- read.transactions("transactions.csv", sep = ",", rm.duplicates = TRUE)

transactions
# transactions in sparse format with
# 505 transactions (rows) and
# 14 items (columns)

dim(transactions)
# [1] 505  14

colnames(transactions)
# [1] "Case of Honey"        "Chocolate Truffles"   "Cocktail Rimmers"     "Gift Set"            
# [5] "Honey"                "Honey & Syrup"        "Honey Wand"           "Ketchup"             
# [9] "Maple Sugar"          "Maple Syrup"          "Marinating Solutions" "Pancake & Waffle Mix"
# [13] "Raw Honey Comb"       "Syrup"   

summary(transactions)
# transactions as itemMatrix in sparse format with
# 505 rows (elements/itemsets/transactions) and
# 14 columns (items) and a density of 0.09363508 
# 
# most frequent items:
#   Honey   Chocolate Truffles             Gift Set Pancake & Waffle Mix              Ketchup 
# 178                  110                   95                   59                   54 
# (Other) 
# 166 
# 
# element (itemset/transaction) length distribution:
#   sizes
# 0   1   2   3   4   5 
# 3 383  88  24   4   3 
# 
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   1.000   1.000   1.311   1.000   5.000 
# 
# includes extended item information - examples:
#   labels
# 1      Case of Honey
# 2 Chocolate Truffles
# 3   Cocktail Rimmers

ggplot(data.frame(size(transactions)), aes(x=size.transactions.)) + 
  geom_histogram(bins = 6) + xlim(0,6) + ggtitle("Number of Items per Transaction")

itemFrequencyPlot(transactions)

catskill_rules <- apriori(transactions,
                        parameter = list(support = .02, #pretty low threshold  
                                         confidence = .5,
                                         minlen = 1,
                                         maxlen = 10))

inspect(catskill_rules)
# lhs              rhs     support    confidence lift    
# 1 {Honey Wand}  => {Honey} 0.02178218 0.8461538  2.400605
# 2 {Maple Syrup} => {Honey} 0.02178218 0.7333333  2.080524
