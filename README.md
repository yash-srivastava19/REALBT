# realbt

## What exactly is Back-Testing?

*"The idea that the future is predictable is a very foolish idea."*

Finance is both stochastic and logical, and if you want to make a fortune, you need to play your cards well. Markets are very unpredictable, and placing your bets on stocks is comparable to gambling, but that is not always true if you have the right tools in your arsenal - and that's exactly what a back-testing engine does. An analogy to better understand what exactly back-testing engine does is this: 

*Imagine before buying a real car(a big investment), you buy a toy car to know whether your investment is worth it or not. A back-test is basically testing your toy car on a pretend road to see how well it drives. Now, a good condition test for the toy car would be when the pretend road is like a real road - with bumps, traffic, gas stations etc. In this way, when your toy car does well on the pretend road, you can be sure that the real car will also do well.*

That's exactly a backtesting engine like REALBT does. You can play with your trading strategy on past market data to see how well it performs. This gives you confidence about your trading strategy in the real world, so instead of playing a gamble, you make an educated choice about which stock to invest in or not.  


## How to use REALBT?

1. **Clone the Repository:** First, clone the repository to your local machine using Git.
```bash
git clone https://github.com/username/realbt.git && cd realbt
```

2. **Set Up the Environment**

REALBT has dependencies that need to be installed. It is recommended to use a virtual environment to avoid conflicts with other Python packages.

2.1 **Create a Virtual Environment**:
```python
python -m venv venv
```

2.2 **Activate the Virtual Environment**:
- On Windows:
```bash
venv\Scripts\activate
```

- On macos/Linux
```bash
source venv/bin/activate
```

2.3 **Install Dependencies**
```bash
pip install -r requirements.txt
```
    
3. **Verify Installation**
Run the CLI help command to ensure the package is installed and working correctly:

```bash
python realbt/cli.py --help
```

You should see a list of available commands, such as `new`, `fetch-data`, and `run`.

4. **Create a New Project:** Use the `new` command to create a new backtesting project:

```bash
python realbt/cli.py new my_project -d /path/to/directory
```


This will generate the following folder structure:
```
my_project/
├── data/
├── results/
├── strategies/
│   └── sample_strategy.py
└── config.yaml
```


 5. **Fetch Historical Data**

Fetch historical stock data using the `fetch-data` command. For example, to fetch Apple stock data for 2024:

```bash
python realbt/cli.py fetch-data AAPL 2024-01-01 2024-12-31 my_project/data/apple.csv
```


---

6. **Define a Strategy**

Edit the `sample_strategy.py` file in the `strategies` folder to define your trading strategy. For example:

```python

from realbt.src.engine import BacktestEngine
def my_strategy(data):

    # Example: Moving Average Crossover Strategy
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    signals = data['SMA_50'] > data['SMA_200']
    return signals

engine = BacktestEngine()
engine.run(data_path="data/apple.csv", strategy=my_strategy)
```


---

7. **Run the Backtest**

Run the Backtest using the `run` command:

```bash
python realbt/cli.py run my_project/config.yaml
```


The results will be saved in the `results` folder, and you can visualize them using the built-in visualization tools.

---

8. **Extend the Framework**

REALBT is modular and extensible. You can:

- Add custom cost models in the `costs` directory.
- Create new strategies in the `strategies` folder.
- Modify the backtesting engine to suit specific requirements.

For example, to add a custom transaction cost model:

```python
def custom_transaction_cost(volume, price):
    return 0.001 * volume * price  # Example: 0.1% transaction cost
```

Integrate it into your strategy:

```python
from realbt.costs.custom_transaction_cost import custom_transaction_cost
def my_strategy_with_costs(data):
    # Define strategy logic
    ...
    # Apply custom transaction costs
    costs = custom_transaction_cost(volume, price)
    ...
```

## Future Works
- What I'm thinking with REALBT strategy is to provide a lot of boilerplate Strategies that everyone can use and extend. The other advantage is the running from YAML, which can branch the process into multiple runs(multithreaded).

- Another thing in this that I have figure out is how to interface this library to the users. Do they clone it and add strategies - how are they going to build the config.YAML? How do we provide it as a package?