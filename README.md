# Bitcoin-Crash-Prediction-or-Forecasting


Approaches
1) FB Prophet
Prophet developed by Facebook, is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with daily periodicity data with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers.

RESULTS:
Following are the graphs for Bitcoin and Ethereum Open and Close feature values prediction using fbProphet. The graph represents the actual Open and Close feature trend over the time, with the lower bound, upper bound and mean predicted values for that time series data. The red line in the graph represents the mean predicted values bounded by the lower and upper bound.

2) HMM
When the opening and closing price is plotted on graph, we could see that there is close relation between them.

To leverage this relationship, we tried using HMM. We created observation sequence using historical data as follows:
Calculate change in opening and closing price as close-openopen
Divide all obtained values in ranges and label them starting 0.

We trained multiple HMM models using various number of states and various number of observation labels.
For prediction, we created list of possible predictions using all the label created while obtaining observation sequence.

We used solution to classic HMM problem that given a model and observation sequence, score the sequence in terms of likelihood. Hence, the problem becomes given HMM model and bitcoin values for d days along with open value for d+1st day, we need to compute close value for d+1st day.

TRYING TO MAKE SENSE OUT OF HMM: We can see even though, these models converge pretty good and we can take inferences from the transition matrix and observation matrix, the prediction part is not working well. Every observation gets the score very close to 1, so it does not help us predict correctly. The variance is so minute that while printing every score is printed as 1, but as we can see in last result (N=6), some scores are better than others by a very tiny margin.

Another way of predicting values for (d+1)st day, would have been by looking at the B matrix and trying to predict the next best value. Such an approach would be like flipping a coin or randomly selecting a state for dth day. Based on the selected state we can look at the transition matrix to find the most probable state for (d+1)st day. On finding the most probable state for the (d+1)st day we can look at the B matrix and find the most probable observation range. Based on this range we can calculate closing value, when given opening value for (d+1)st day. Here is the catch ☹. If we look closely at the B matrix, irrespective of the value of N. The highest probabilities are concentrated at observation c (range -10% to 0%) and d (range 0% to 10%). So, irrespective of the whichever state we select for the (d+1)st day we will always land either on c or d observation, which is like flipping a coin.

3) GMM-HMM (Gaussian Mixture Model)
After getting very low accuracy in simple HMM, we moved on to Gaussian Mixture based HMM. Earlier in simple HMM approach we were dividing the observation (close-openopen) into different discrete ranges. The intuition behind trying GMM-HMM approach was moving from discrete observations to continuous observations. A C library with python wrappings ‘hmmlearn’ has GMMHMM capabilities. We trained the GMMHMM models as follows: Observation sequence was created using vector observations. Observation at time t was given as Ot= (close-open/open, high-open/open, open-low/open)

Hence Ot= (fractional-Close, fractional-High, fractional-Low)

We used Maximum a Posteriori algorithm to train the model. We used different number of states and different number of mixtures to train the models. For prediction, we used following point as possible observations.
The values between the brackets [] are the observation being scored (fractional-Close, fractional-High, fractional-Low) and the next value is the predicted score for that observation

4) SVM-PCA
In this technique, we tried to apply ML to answer specific business query i.e. will I earn profit selling the bitcoin tomorrow. In this question, a user has to specify the price at which the user wants to sell the bitcoin.
To train the model, we used different features than earlier models.
For bitcoin:
btc_avg_block_size, btc_n_transactions, btc_n_transactions_total, btc_n_transactions_excluding_popular, btc_n_transactions_excluding_chains_longer_than_100, btc_output_volume
For Ethereum:
eth_supply, eth_hashrate, eth_difficulty, eth_blocks, eth_blocksize, eth_blocktime, eth_ethersupply
Since we did not know the significance of each feature, we used PCA to narrow the dimensionality of the data. We found that, there are 3 most prominent eigen values. So, our experiments were focused on 3 and 2 most prominent bases.

Then we label this data using known close and open price if more than user give price as +1 or -1. The next step was to train SVM based on this labeled data. After training the model on given historical data, we predicted the if the price would be higher than user specified one. 

To predict the future price of cryptocurrencies is tougher than it looks. We tried Facebook Prophet, HMM, GMM-HMM, PCA-SVM, and PCA-SVM-FBProphet. In some cases, the results look promising. But, there are a lot we can improve.
