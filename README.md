# realbt

## Analogy
Imagine you have a toy car (your trading strategy) and you want to know if it's good before buying a real car. A backtest is like playing with your toy car on a pretend road (past market data) to see how well it drives.
But many toy roads are too smooth and perfect - no bumps, no traffic, no waiting at gas stations. This gives you the wrong idea about how your car will work in the real world.

REALBT makes the toy road much more like a real road - with bumps (price slippage), traffic (market impact), and gas stations (transaction costs). This way, when your toy car does well, you can be more confident your real car (actual trading strategy) will also do well.

## Research
- What the framework should provide is a way to write re-suable trading strategies instead of having to build the everything.

- There is this thing called backtesting.py, which is build on backtrader. The thing with backtesting.py is that it is tested well and offere less boilerplate compared to backtrader. This is the entire workflow for backtrader.

```python
from datetime import datetime
import backtrader as bt

class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)

data0 = bt.feeds.YahooFinanceData(dataname='MSFT', fromdate=datetime(2011, 1, 1),
                                  todate=datetime(2012, 12, 31))
cerebro.adddata(data0)

cerebro.run()
cerebro.plot()
```

- The flow of backtesting.py is like this:

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)  # the self.sma is just an indicator(an array of values), which is revealed in the next() function. This is init for 
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        # main strategy runtime method, called on each new Data Instance.
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()


bt = Backtest(GOOG, SmaCross,
              cash=10000, commission=.002,
              exclusive_orders=True)

output = bt.run()
bt.plot()
```

- What I'm thinking with REALBT strategy is to provide a lot of boilerplate Strategies that everyone can use and extend. The other advantage is the running from YAML, which can branch the process into multiple runs(multithreaded).

Another thing in this that I have figure out is how to interface this library to the users. Do they clone it and add strategies - how are they going to build the config.YAML? Do we provide it as a package?

