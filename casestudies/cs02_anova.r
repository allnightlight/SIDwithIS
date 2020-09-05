
df = read.csv("cs02a-01_score.csv")
df = df[df$epoch > 0, ]

target_criteria = "RMSE_TOTAL_AVERAGE"
nAnova = 2**3

dfEvOff = df[df$criteria == target_criteria
    & df$use_imbalanced_sampling == 0 ,]
dfEvOff = dfEvOff[dfEvOff$key %in% sample(unique(dfEvOff$key), nAnova),]

dfEvOn = df[df$criteria == target_criteria
    & df$use_imbalanced_sampling == 1 ,]
dfEvOn = dfEvOn[dfEvOn$key %in% sample(unique(dfEvOn$key), nAnova),]

resEvOn = aov( 
    log10(score) ~ 
        + log10(epoch)
        + log10(sampling_balance)
        + log10(NhiddenAgent)
        + log10(N0)
        + use_offset_compensate
        , data = dfEvOn)

resEvOff = aov( 
    log10(score) ~ 
        + log10(epoch)
        + log10(NhiddenAgent)
        + log10(N0)
        + use_offset_compensate
        , data = dfEvOff)

print("===============================")
print(summary.lm(resEvOn))
print("===============================")
print(summary.lm(resEvOff))
print("===============================")

