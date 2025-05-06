**RL Strategy for Scalp Trading using Time Series Generation with GANs on Moscow Stock Exchange**

Scalp trading (scalping) is a fast-paced trading strategy that involves making quick, short-term trades within a single trading session in order to profit from small price fluctuations in financial markets. This study explores the application of Generative Adversarial Networks (GANs) and Reinforcement Learning (RL) in financial time series. Scalping has the potential for higher profitability due to frequent intraday trades. It fills the gap between traditional day trading and high-frequency trading, which is resource-intensive. The high frequency of transactions in Scalping produces more data than traditional day trading that allows us to apply complex deep learning method that can model non-linear temporal dependencies in financial data. GANs can simulate a realistic time series data providing a reliable learning environment for RL strategies. Our core contribution lies in demonstrating that the synergistic integration of RL, Scalp trading, and diverse GAN architectures (e.g., RNN-GANs, TCN-GANs) significantly enhances trading performance metrics: profitability, win rate, and sharp ratio. This study proposes the use of scalp trading based on neural network models, which yields an increase of over 10\% compared to manual trading or strategies using linear models.

Git repository consists of 2 branches:
1) main branch with RL strategy and EDA of Lukoil stock market data
2) gan branch with RNN and TCN GANs

**RL setup**
To run Reinforcement learning agent training simply run the following command inside RL folder
```python
python3 grid_search.py
```
**GAN training**
1) swtich to a gan branch ```git checkout gan```
2) clone the branch with ```git clone```
3) drop data csv into tcngan folder
4) choose type of GAN to train and run training in its folder with
```python
   python main.py
   ``` 
