# RL Strategy for Scalp Trading using Time Series Generation with GANs on Moscow Stock Exchange

Scalp trading (scalping) is a fast-paced trading strategy that involves making quick, short-term trades within a single trading session in order to profit from small price fluctuations in financial markets. This study explores the application of Generative Adversarial Networks (GANs) and Reinforcement Learning (RL) in financial time series. Scalping has the potential for higher profitability due to frequent intraday trades. It fills the gap between traditional day trading and high-frequency trading, which is resource-intensive. The high frequency of transactions in Scalping produces more data than traditional day trading that allows us to apply complex deep learning method that can model non-linear temporal dependencies in financial data. GANs can simulate a realistic time series data providing a reliable learning environment for RL strategies. Our core contribution lies in demonstrating that the synergistic integration of RL, Scalp trading, and diverse GAN architectures (e.g., RNN-GANs, TCN-GANs) significantly enhances trading performance metrics: profitability, win rate, and sharp ratio. This study proposes the use of scalp trading based on neural network models, which yields an increase of over 10\% compared to manual trading or strategies using linear models.

Git repository consists of 2 branches:
1) main branch with RL strategy and EDA of Lukoil stock market data
2) gan branch with RNN and TCN GANs

You are now in **main** branch

## EDA
To see Exploratory Data Analysis for Lukoil stocks simply download ```EDA.ipynb``` and open it in an environment of your choice (Google Colab, Jupyter Notebook, etc)

## RL setup
Clone the main branch  ```git clone```

To run Reinforcement learning agent training simply run the following command from cloned repo
```python
cd RL
pip install -r rl_requirements.txt
python3 grid_search.py
```
All the results (total profit, sharp ratio, etc) on validation set you will see in file logs.txt every 64k timestemps of training. 

In eval.py you can find an option in line 152 
```python
 val_reward_true, za_val_true, nza_val_true, wa_val_true, info = evaluate_agent(self.model, create_masked_env(TestingTradingEnv(**self.val_env_config)), max_steps=min(20000, len(self.df_val)) - 1, with_action_logs=True)
 ```
You can disable with_action_logs (set it to False) if you don't want to see actual descisions the were made by agent during evaluation.  

If you want you can login into your wandb with your token and track your progress during training. 

Our results 

![image](https://github.com/user-attachments/assets/d5c0576f-127b-4c59-aca2-3385112d41bd)



## GAN training
Data for training can be downloaded [here](https://drive.google.com/file/d/1OHlZT5b5a9qAnQxwF8-x6UO3OO8Lv31N/view?usp=sharing)

To run Generative Adversarial Network training follow these steps below:
1) swtich to a gan branch ```git checkout gan```
2) clone the branch with ```git clone```
3) install dependencies with
```python
pip install -r gan_requirements.txt
```
4) drop data csv into tcngan folder
5) choose type of GAN to train and run training in its folder with
```python
   python main.py
   ``` 
