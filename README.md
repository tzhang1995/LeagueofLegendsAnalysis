# LeagueofLegendsAnalysis
Project detailing exploratory data analysis of League of Legends match data

# Run Order
I. Part 1
  1. DataExtraction
  2. BasicStatisticalModels
  3. ChapmionStatistics
 
II. Part 2

  4. TreeNN


III. Part 3

  5. NeuralNetworks

Item's 1, 2, and 3 are in "Part 1", item 4 in "Part 2" and item 5 in "Part3". This is mainly due to memory constraints - the match files are quite large and storing them in memeory is too taxing for my computer.

Note, the RDATA and match data .csv files are not uploaded to git for space reasons. You can message me if you'd like access to those files. It is not reccomended to run DataExtraction.RMD as this will take a minimum of 20 or so hours to run because of the limits on the personal API key. 

Conclusions:
1. Vision Score:
  A. As a suppport, your vision score should follow a 25-40-80 rule, corresponding to the vision score at 20-30-40 minutes to have a 50% chance of victory.
  B. As any other role, your vision score should roughly follow a 10-20-30 rule for the same timestamps
2. Champion Selection and Gold Distribution:
  A. Play easy champions, even in higher elos. 
  B. Try to get your ADC gold, same with junglers in higher elos.
  C. Mid needs to get the team ahead, they typically drop in winrate the more gold they have above 20% of team gold, especially in higher elo.
  
The .html files should be easiest to view any results with the exception of the neural networks which will entail a more involved process.

A few basic Neural Network models were explored, likely will need more data from higher elo games to make any stronger conclusions regarding the predictive power of models. When just looking at one team, the validation accuracy of the neural network was around 52%. When including both teams, the validation accuracy rose to around 53% which I consider a decent result for a game with such high variance as league of legends. 
